"""Unit tests for app/storage/history.py"""
import pytest
import sys
import os
from unittest.mock import Mock, patch
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'microservice'))


class TestPredictionHistory:
    """Tests for in-memory PredictionHistory class."""

    @pytest.mark.unit
    def test_prediction_history_init(self):
        """Test PredictionHistory initialization."""
        from app.storage.history import PredictionHistory
        
        history = PredictionHistory()
        assert history is not None
        assert history.count() == 0
        assert len(history.all()) == 0

    @pytest.mark.unit
    def test_add_prediction(self, sample_entity_result):
        """Test adding a prediction to history."""
        from app.storage.history import PredictionHistory
        
        history = PredictionHistory()
        timestamp = datetime.now(tz=timezone.utc)
        
        record = history.add(
            input_text="Test text",
            output=[sample_entity_result],
            model="en_core_web_sm",
            timestamp=timestamp,
        )
        
        assert record is not None
        assert record.id == 1
        assert record.input_text == "Test text"
        assert record.model == "en_core_web_sm"

    @pytest.mark.unit
    def test_add_multiple_predictions(self, sample_entity_result):
        """Test adding multiple predictions."""
        from app.storage.history import PredictionHistory
        
        history = PredictionHistory()
        timestamp = datetime.now(tz=timezone.utc)
        
        record1 = history.add(
            input_text="Text 1",
            output=[sample_entity_result],
            model="en_core_web_sm",
            timestamp=timestamp,
        )
        record2 = history.add(
            input_text="Text 2",
            output=[sample_entity_result],
            model="en_core_web_lg",
            timestamp=timestamp,
        )
        
        assert record1.id == 1
        assert record2.id == 2
        assert history.count() == 2

    @pytest.mark.unit
    def test_get_all_predictions(self, sample_entity_result):
        """Test retrieving all predictions."""
        from app.storage.history import PredictionHistory
        
        history = PredictionHistory()
        timestamp = datetime.now(tz=timezone.utc)
        
        # Add predictions
        for i in range(3):
            history.add(
                input_text=f"Text {i}",
                output=[sample_entity_result],
                model="en_core_web_sm",
                timestamp=timestamp,
            )
        
        all_records = history.all()
        assert len(all_records) == 3

    @pytest.mark.unit
    def test_count_predictions(self, sample_entity_result):
        """Test counting predictions."""
        from app.storage.history import PredictionHistory
        
        history = PredictionHistory()
        timestamp = datetime.now(tz=timezone.utc)
        
        assert history.count() == 0
        
        history.add(
            input_text="Test",
            output=[sample_entity_result],
            model="en_core_web_sm",
            timestamp=timestamp,
        )
        
        assert history.count() == 1

    @pytest.mark.unit
    def test_ping_returns_true(self):
        """Test that ping always returns True for in-memory history."""
        from app.storage.history import PredictionHistory
        
        history = PredictionHistory()
        assert history.ping() is True

    @pytest.mark.unit
    def test_thread_safety(self, sample_entity_result):
        """Test thread-safe operations."""
        from app.storage.history import PredictionHistory
        import threading
        
        history = PredictionHistory()
        timestamp = datetime.now(tz=timezone.utc)
        
        def add_record():
            for i in range(10):
                history.add(
                    input_text=f"Text {i}",
                    output=[sample_entity_result],
                    model="en_core_web_sm",
                    timestamp=timestamp,
                )
        
        threads = [threading.Thread(target=add_record) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert history.count() == 50


class TestHistoryFactory:
    """Tests for history factory function."""

    @pytest.mark.unit
    def test_build_history_with_redis_available(self, mock_redis_client):
        """Test factory returns RedisHistory when Redis is available."""
        with patch('app.storage.redis_history.RedisHistory') as mock_redis_class:
            mock_instance = Mock()
            mock_instance.ping.return_value = True
            mock_redis_class.return_value = mock_instance
            
            with patch('app.storage.history.logger'):
                # Need to reimport to get fresh factory
                import importlib
                import app.storage.history
                importlib.reload(app.storage.history)
                from app.storage.history import _build_history
                
                history = _build_history()
                assert history is not None

    @pytest.mark.unit
    def test_build_history_fallback_to_memory(self):
        """Test factory falls back to PredictionHistory when Redis unavailable."""
        with patch('app.storage.redis_history.RedisHistory', side_effect=Exception("Connection failed")):
            with patch('app.storage.history.logger'):
                import importlib
                import app.storage.history
                importlib.reload(app.storage.history)
                from app.storage.history import _build_history, PredictionHistory
                
                history = _build_history()
                assert isinstance(history, PredictionHistory)

    @pytest.mark.unit
    def test_build_history_redis_ping_fails(self):
        """Test factory falls back when Redis ping fails."""
        with patch('app.storage.redis_history.RedisHistory') as mock_redis_class:
            mock_instance = Mock()
            mock_instance.ping.return_value = False
            mock_redis_class.return_value = mock_instance
            
            with patch('app.storage.history.logger'):
                import importlib
                import app.storage.history
                importlib.reload(app.storage.history)
                from app.storage.history import _build_history, PredictionHistory
                
                history = _build_history()
                assert isinstance(history, PredictionHistory)
