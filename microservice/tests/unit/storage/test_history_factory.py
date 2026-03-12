"""
unit/storage/test_history_factory.py
--------------------------------------
Factory logic — MongoDB-first, in-memory fallback.
"""

import pytest
from unittest.mock import patch

from app.storage.history import PredictionHistory, _build_history

@pytest.mark.unit
class TestBuildHistory:

    @patch("app.storage.mongo_history.MongoHistory")
    def test_returns_mongo_when_reachable(self, MockMongo):
        mock_instance = MockMongo.return_value
        mock_instance.ping.return_value = True
        
        history = _build_history()
        assert history == mock_instance

    @patch("app.storage.mongo_history.MongoHistory")
    def test_falls_back_when_mongo_unreachable(self, MockMongo):
        MockMongo.side_effect = Exception("Connection refused")
        history = _build_history()
        assert isinstance(history, PredictionHistory)

    @patch("app.storage.mongo_history.MongoHistory")
    def test_falls_back_when_ping_fails(self, MockMongo):
        mock_instance = MockMongo.return_value
        mock_instance.ping.return_value = False
        
        history = _build_history()
        assert isinstance(history, PredictionHistory)

    @patch("app.storage.mongo_history.MongoHistory")
    def test_fallback_has_correct_interface(self, MockMongo):
        MockMongo.side_effect = Exception("down")
        history = _build_history()
        
        # Must expose the standard interface
        assert hasattr(history, "add")
        assert hasattr(history, "all")
        assert hasattr(history, "count")
        assert hasattr(history, "ping")