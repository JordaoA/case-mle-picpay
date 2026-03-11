"""
storage/history.py
------------------
Factory module — returns the active prediction history backend.

The rest of the application always imports from here:

    from app.storage.history import prediction_history

Swapping backends requires changing only this file.

Strategy:
    - Try to connect to MongoDB on startup
    - If Mongo is reachable → use MongoHistory
    - If Mongo is unavailable → fall back to in-memory PredictionHistory
      and log a clear warning (so it's obvious in production logs)
"""

import logging
import threading
from datetime import datetime
from typing import Union

from app.schemas.requests import EntityResult, PredictionRecord

logger = logging.getLogger("ner_service.history_factory")


class PredictionHistory:
    """Thread-safe in-memory prediction log. Used as fallback if Mongo is down."""

    def __init__(self) -> None:
        self._records: list[PredictionRecord] = []
        self._lock = threading.Lock()
        self._counter = 0

    def add(
        self,
        input_text: str,
        output: list[EntityResult],
        model: str,
        timestamp: datetime,
    ) -> PredictionRecord:
        with self._lock:
            self._counter += 1
            record = PredictionRecord(
                id=str(self._counter),
                input_text=input_text,
                output=output,
                model=model,
                timestamp=timestamp,
            )
            self._records.append(record)
            return record

    def all(self) -> list[PredictionRecord]:
        with self._lock:
            return list(self._records)

    def count(self) -> int:
        with self._lock:
            return self._counter

    def ping(self) -> bool:
        return True


def _build_history() -> Union["MongoHistory", PredictionHistory]:
    """
    Attempts to initialise MongoHistory.
    Falls back to PredictionHistory if Mongo is unreachable.
    """
    try:
        from app.storage.mongo_history import MongoHistory

        instance = MongoHistory()
        if instance.ping():
            logger.info("Using MongoDB-backed prediction history.")
            return instance
        else:
            raise ConnectionError("MongoDB ping failed.")
    except Exception as exc:
        logger.warning(
            f"MongoDB unavailable ({exc}). "
            f"Falling back to in-memory history — data will NOT persist across restarts."
        )
        return PredictionHistory()