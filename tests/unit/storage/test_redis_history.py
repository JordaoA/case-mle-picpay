"""
unit/storage/test_redis_history.py
------------------------------------
RedisHistory backed by fakeredis — add, all, count, TTL, thread safety.
No real Redis connection is made.
"""

import json
import threading
from datetime import datetime, timezone

import fakeredis
import pytest

from app.schemas.requests import EntityResult
from app.storage.redis_history import RedisHistory


def _ts():
    return datetime.now(tz=timezone.utc)


def _client():
    server = fakeredis.FakeServer()
    return fakeredis.FakeRedis(server=server, decode_responses=True)


def _make() -> RedisHistory:
    return RedisHistory(client=_client())


def _money():
    return [EntityResult(label="MONEY", text="$45", start=13, end=16)]


@pytest.mark.unit
class TestInitialState:

    def test_count_zero(self):
        assert _make().count() == 0

    def test_all_empty(self):
        assert _make().all() == []

    def test_ping_returns_true(self):
        assert _make().ping() is True


@pytest.mark.unit
class TestAdd:

    def test_first_id_is_one(self):
        h = _make()
        r = h.add("text", _money(), "en_core_web_sm", _ts())
        assert r.id == 1

    def test_ids_are_sequential(self):
        h = _make()
        ids = [h.add(f"t{i}", [], "en_core_web_sm", _ts()).id for i in range(5)]
        assert ids == [1, 2, 3, 4, 5]

    def test_record_fields_stored(self):
        h = _make()
        ts = _ts()
        r = h.add("Send $45", _money(), "en_core_web_sm", ts)
        assert r.input_text == "Send $45"
        assert r.model == "en_core_web_sm"
        assert r.id == 1
        assert len(r.output) == 1

    def test_empty_entities_stored(self):
        h = _make()
        r = h.add("Hello", [], "en_core_web_sm", _ts())
        assert r.output == []

    def test_multiple_entities(self):
        h = _make()
        entities = [
            EntityResult(label="MONEY",  text="$45",     start=5,  end=8),
            EntityResult(label="PERSON", text="Michael", start=12, end=19),
        ]
        r = h.add("text", entities, "en_core_web_sm", _ts())
        assert len(r.output) == 2


@pytest.mark.unit
class TestAll:

    def test_newest_first(self):
        h = _make()
        for text in ["first", "second", "third"]:
            h.add(text, [], "en_core_web_sm", _ts())
        texts = [r.input_text for r in h.all()]
        assert texts == ["third", "second", "first"]

    def test_returns_list_of_prediction_records(self):
        h = _make()
        h.add("text", _money(), "en_core_web_sm", _ts())
        from app.schemas.requests import PredictionRecord
        records = h.all()
        assert all(isinstance(r, PredictionRecord) for r in records)

    def test_empty_when_no_records(self):
        assert _make().all() == []

    def test_entities_deserialised_correctly(self):
        h = _make()
        h.add("text", _money(), "en_core_web_sm", _ts())
        record = h.all()[0]
        assert record.output[0].label == "MONEY"
        assert record.output[0].text == "$45"
        assert record.output[0].start == 13
        assert record.output[0].end == 16


@pytest.mark.unit
class TestCount:

    def test_count_increments(self):
        h = _make()
        for _ in range(7):
            h.add("t", [], "en_core_web_sm", _ts())
        assert h.count() == 7

    def test_count_is_zero_initially(self):
        assert _make().count() == 0

    def test_count_survives_ttl_expiry(self):
        """
        Counter is a separate Redis key — it should survive even if individual
        record hashes expired (simulated by direct key deletion).
        """
        client = _client()
        h = RedisHistory(client=client)
        h.add("text", [], "en_core_web_sm", _ts())

        # Simulate TTL expiry by deleting the record hash directly
        client.delete("prediction:1")

        # Counter must still report 1
        assert h.count() == 1


@pytest.mark.unit
class TestTTL:

    def test_ttl_is_set_on_record_key(self):
        client = _client()
        h = RedisHistory(client=client)
        h.add("text", [], "en_core_web_sm", _ts())
        ttl = client.ttl("prediction:1")
        # fakeredis returns -1 if no TTL — assert a real TTL was set
        assert ttl > 0

    def test_zero_ttl_means_no_expiry(self, monkeypatch):
        monkeypatch.setenv("REDIS_TTL_SECONDS", "0")
        from app.config import Settings
        s = Settings()
        client = _client()
        h = RedisHistory(client=client)
        h._ttl = 0  # override directly
        h.add("text", [], "en_core_web_sm", _ts())
        ttl = client.ttl("prediction:1")
        # -1 means key exists with no TTL
        assert ttl == -1

    def test_missing_hash_skipped_in_all(self):
        client = _client()
        h = RedisHistory(client=client)
        h.add("keep",   [], "m", _ts())
        h.add("expire", [], "m", _ts())
        client.delete("prediction:2")  # simulate expiry
        records = h.all()
        assert len(records) == 1
        assert records[0].input_text == "keep"


@pytest.mark.unit
class TestThreadSafety:

    def test_concurrent_adds_unique_ids(self):
        """INCR in Redis is atomic — every thread must get a unique ID."""
        server = fakeredis.FakeServer()
        h = RedisHistory(
            client=fakeredis.FakeRedis(server=server, decode_responses=True)
        )
        n = 50
        threads = [
            threading.Thread(target=lambda: h.add("t", [], "m", _ts()))
            for _ in range(n)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert h.count() == n
        ids = {r.id for r in h.all()}
        assert len(ids) == n