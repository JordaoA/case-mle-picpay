"""
unit/storage/test_memory_history.py
-------------------------------------
In-memory PredictionHistory — add, all, count, thread safety.
"""

import threading
from datetime import datetime, timezone

import pytest

from app.schemas.requests import EntityResult
from app.storage.history import PredictionHistory


def _ts():
    return datetime.now(tz=timezone.utc)


def _money():
    return [EntityResult(label="MONEY", text="$45", start=13, end=16)]


@pytest.mark.unit
class TestInitialState:

    def test_empty_on_creation(self):
        h = PredictionHistory()
        assert h.all() == []

    def test_count_zero(self):
        assert PredictionHistory().count() == 0

    def test_ping_always_true(self):
        assert PredictionHistory().ping() is True


@pytest.mark.unit
class TestAdd:

    def test_first_id_is_one(self):
        h = PredictionHistory()
        r = h.add("text", _money(), "en_core_web_sm", _ts())
        assert r.id == '1'

    def test_ids_are_sequential(self):
        h = PredictionHistory()
        ids = [h.add(f"t{i}", [], "en_core_web_sm", _ts()).id for i in range(5)]
        assert ids == ['1', '2', '3', '4', '5']

    def test_record_stores_all_fields(self):
        h = PredictionHistory()
        ts = _ts()
        r = h.add("Send $45", _money(), "en_core_web_sm", ts)
        assert r.input_text == "Send $45"
        assert r.model == "en_core_web_sm"
        assert r.timestamp == ts
        assert len(r.output) == 1
        assert r.output[0].label == "MONEY"

    def test_empty_entities_stored(self):
        h = PredictionHistory()
        r = h.add("Hello", [], "en_core_web_sm", _ts())
        assert r.output == []

    def test_multiple_entities(self):
        h = PredictionHistory()
        entities = [
            EntityResult(label="MONEY",  text="$45",     start=5,  end=8),
            EntityResult(label="PERSON", text="Michael", start=12, end=19),
        ]
        r = h.add("text", entities, "en_core_web_sm", _ts())
        assert len(r.output) == 2


@pytest.mark.unit
class TestAll:

    def test_preserves_insertion_order(self):
        h = PredictionHistory()
        for text in ["alpha", "beta", "gamma"]:
            h.add(text, [], "en_core_web_sm", _ts())
        assert [r.input_text for r in h.all()] == ["alpha", "beta", "gamma"]

    def test_returns_independent_copy(self):
        h = PredictionHistory()
        h.add("text", [], "en_core_web_sm", _ts())
        copy = h.all()
        copy.clear()
        assert h.count() == 1

    def test_empty_list_when_empty(self):
        assert PredictionHistory().all() == []

    def test_count_matches_all_length(self):
        h = PredictionHistory()
        for _ in range(4):
            h.add("t", [], "m", _ts())
        assert h.count() == len(h.all())


@pytest.mark.unit
class TestThreadSafety:

    def test_concurrent_adds_produce_unique_ids(self):
        h = PredictionHistory()
        n = 100
        threads = [
            threading.Thread(target=lambda: h.add("t", [], "m", _ts()))
            for _ in range(n)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert h.count() == n
        assert len({r.id for r in h.all()}) == n

    def test_concurrent_reads_during_writes(self):
        h = PredictionHistory()
        errors = []

        def writer():
            for _ in range(20):
                h.add("t", [], "m", _ts())

        def reader():
            try:
                for _ in range(20):
                    h.all()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []