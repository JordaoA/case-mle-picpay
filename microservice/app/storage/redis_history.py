"""
storage/redis_history.py
-------------------------
Redis-backed implementation of the prediction history store.

Data model in Redis:
    prediction:{id}     → Hash  — all fields of a single PredictionRecord
    predictions:index   → List  — ordered list of IDs (LPUSH = newest first)
    predictions:counter → String (integer) — atomic ID counter via INCR

Why this structure:
    - INCR is atomic, so concurrent requests can never get the same ID
    - LPUSH + LRANGE gives efficient ordered retrieval without sorting
    - Storing each record as a Hash allows partial field access if needed
    - TTL is set per-record so old predictions expire automatically

The public interface (add / all / count) is identical to PredictionHistory
so ner_service.py requires zero changes.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

import redis

from app.config import settings
from app.schemas.requests import EntityResult, PredictionRecord

logger = logging.getLogger("ner_service.redis_history")

# Redis key constants — centralised to avoid typos across methods
KEY_COUNTER = "predictions:counter"
KEY_INDEX   = "predictions:index"


def _record_key(prediction_id: int) -> str:
    return f"prediction:{prediction_id}"


class RedisHistory:
    """
    Thread-safe prediction history backed by Redis.
    Each instance shares the same connection pool.
    """

    def __init__(self, client: Optional[redis.Redis] = None) -> None:
        """
        Args:
            client: Optional pre-built Redis client (useful for testing).
                    If None, a client is created from settings.
        """
        if client is not None:
            self._redis = client
        else:
            pool = redis.ConnectionPool(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password or None,
                decode_responses=True,      # always return str, never bytes
                max_connections=20,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            self._redis = redis.Redis(connection_pool=pool)

        self._ttl = settings.redis_ttl_seconds
        logger.info(
            f"RedisHistory initialised — "
            f"{settings.redis_host}:{settings.redis_port}/db{settings.redis_db} | "
            f"TTL: {self._ttl}s"
        )


    def add(
        self,
        input_text: str,
        output: list[EntityResult],
        model: str,
        timestamp: datetime,
    ) -> PredictionRecord:
        """
        Persists a prediction to Redis and returns the stored record.

        Steps:
            1. INCR predictions:counter  → get a unique, atomic ID
            2. HSET prediction:{id}      → store all record fields as a Hash
            3. LPUSH predictions:index   → prepend ID to the ordered index
            4. EXPIRE prediction:{id}    → set TTL if configured
        """
        prediction_id: int = self._redis.incr(KEY_COUNTER)

        record = PredictionRecord(
            id=prediction_id,
            input_text=input_text,
            output=output,
            model=model,
            timestamp=timestamp,
        )

        # Serialise output (list of EntityResult) to JSON for Redis storage
        serialised = {
            "id":         str(prediction_id),
            "input_text": input_text,
            "output":     json.dumps([e.model_dump() for e in output]),
            "model":      model,
            "timestamp":  timestamp.isoformat(),
        }
        key = _record_key(prediction_id)

        pipe = self._redis.pipeline()
        pipe.hset(key, mapping=serialised)
        pipe.lpush(KEY_INDEX, prediction_id)
        if self._ttl > 0:
            pipe.expire(key, self._ttl)
        pipe.execute()

        logger.debug(f"Stored prediction id={prediction_id} in Redis (key={key})")
        return record

    def all(self) -> list[PredictionRecord]:
        """
        Returns all stored predictions, newest first.

        Uses LRANGE to fetch the full ID index, then retrieves each
        record's Hash in a single pipeline to minimise round-trips.
        """
        ids = self._redis.lrange(KEY_INDEX, 0, -1)  # all IDs, newest first
        if not ids:
            return []

        pipe = self._redis.pipeline()
        for prediction_id in ids:
            pipe.hgetall(_record_key(int(prediction_id)))
        raw_records = pipe.execute()

        records = []
        for raw in raw_records:
            if not raw:
                continue  # record may have expired (TTL)
            try:
                records.append(self._deserialise(raw))
            except Exception as exc:
                logger.warning(f"Failed to deserialise prediction record: {exc}")

        return records

    def count(self) -> int:
        """
        Returns the total number of predictions ever recorded.
        Uses the atomic counter — accurate even after TTL-based expiry.
        """
        value = self._redis.get(KEY_COUNTER)
        return int(value) if value else 0

    def ping(self) -> bool:
        """Health check — returns True if Redis is reachable."""
        try:
            return self._redis.ping()
        except redis.RedisError:
            return False

    def _deserialise(self, raw: dict) -> PredictionRecord:
        """Reconstructs a PredictionRecord from a Redis Hash."""
        output = [
            EntityResult(**entity)
            for entity in json.loads(raw["output"])
        ]
        return PredictionRecord(
            id=int(raw["id"]),
            input_text=raw["input_text"],
            output=output,
            model=raw["model"],
            timestamp=datetime.fromisoformat(raw["timestamp"]),
        )