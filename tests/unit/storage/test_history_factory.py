"""
unit/storage/test_history_factory.py
--------------------------------------
Factory logic — Redis-first, in-memory fallback.
Uses fakeredis for the happy path; patch redis.Redis for failure paths.
"""

import pytest
import fakeredis
from unittest.mock import MagicMock, patch

from app.storage.history import PredictionHistory, _build_history


@pytest.mark.unit
class TestBuildHistory:

    def test_returns_redis_when_reachable(self):
        fake_client = fakeredis.FakeRedis(decode_responses=True)
        with patch("app.storage.redis_history.redis.Redis", return_value=fake_client), \
             patch("app.storage.redis_history.redis.ConnectionPool"):
            history = _build_history()
        from app.storage.redis_history import RedisHistory
        assert isinstance(history, RedisHistory)

    def test_falls_back_when_redis_unreachable(self):
        with patch("app.storage.redis_history.RedisHistory") as MockRedis:
            MockRedis.side_effect = Exception("Connection refused")
            history = _build_history()
        assert isinstance(history, PredictionHistory)

    def test_falls_back_when_ping_fails(self):
        mock_instance = MagicMock()
        mock_instance.ping.return_value = False
        with patch("app.storage.redis_history.RedisHistory", return_value=mock_instance):
            history = _build_history()
        assert isinstance(history, PredictionHistory)

    def test_fallback_has_correct_interface(self):
        with patch("app.storage.redis_history.RedisHistory") as MockRedis:
            MockRedis.side_effect = Exception("down")
            history = _build_history()
        # Must expose the same interface as RedisHistory
        assert hasattr(history, "add")
        assert hasattr(history, "all")
        assert hasattr(history, "count")
        assert hasattr(history, "ping")

    def test_fallback_is_functional(self):
        from datetime import datetime, timezone
        with patch("app.storage.redis_history.RedisHistory") as MockRedis:
            MockRedis.side_effect = Exception("down")
            history = _build_history()

        ts = datetime.now(tz=timezone.utc)
        history.add("Hello", [], "en_core_web_sm", ts)
        assert history.count() == 1