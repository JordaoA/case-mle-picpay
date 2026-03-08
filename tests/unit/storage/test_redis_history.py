"""Unit tests for app/storage/redis_history.py"""
import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'microservice'))


class TestRedisHistory:
    """Tests for Redis-backed prediction history."""

    @pytest.fixture
    def redis_history(self, mock_redis_client, mock_settings):
        """Create RedisHistory instance with mocked Redis."""
        with patch('app.storage.redis_history.settings', mock_settings):
            from app.storage.redis_history import RedisHistory
            return RedisHistory(client=mock_redis_client)

    @pytest.mark.unit
    def test_redis_history_init(self, mock_redis_client, mock_settings):
        """Test RedisHistory initialization."""
        with patch('app.storage.redis_history.settings', mock_settings):
            with patch('app.storage.redis_history.logger'):
                from app.storage.redis_history import RedisHistory
                
                history = RedisHistory(client=mock_redis_client)
                assert history is not None

    @pytest.mark.unit
    def test_add_prediction(self, redis_history, mock_redis_client, sample_entity_result):
        """Test adding a prediction to Redis."""
        mock_redis_client.incr.return_value = 1
        mock_redis_client.pipeline.return_value.execute.return_value = [None, None, None]
        mock_pipeline = Mock()
        mock_redis_client.pipeline.return_value = mock_pipeline
        
        timestamp = datetime.now(tz=timezone.utc)
        
        record = redis_history.add(
            input_text="Test text",
            output=[sample_entity_result],
            model="en_core_web_sm",
            timestamp=timestamp,
        )
        
        assert record is not None
        assert record.input_text == "Test text"
        assert record.model == "en_core_web_sm"
        assert mock_redis_client.incr.called

    @pytest.mark.unit
    def test_add_calls_redis_operations(self, redis_history, mock_redis_client, sample_entity_result):
        """Test that add() calls Redis operations correctly."""
        mock_redis_client.incr.return_value = 42
        mock_pipeline = Mock()
        mock_redis_client.pipeline.return_value = mock_pipeline
        
        timestamp = datetime.now(tz=timezone.utc)
        
        redis_history.add(
            input_text="Test",
            output=[sample_entity_result],
            model="en_core_web_sm",
            timestamp=timestamp,
        )
        
        mock_redis_client.incr.assert_called()
        mock_redis_client.pipeline.assert_called()

    @pytest.mark.unit
    def test_all_returns_list(self, redis_history, mock_redis_client):
        """Test that all() returns a list of predictions."""
        mock_redis_client.lrange.return_value = []
        
        records = redis_history.all()
        
        assert isinstance(records, list)
        assert len(records) == 0

    @pytest.mark.unit
    def test_all_with_records(self, redis_history, mock_redis_client, sample_entity_result):
        """Test retrieving all predictions from Redis."""
        entity_dict = sample_entity_result.model_dump()
        raw_record = {
            'id': '1',
            'input_text': 'Test text',
            'output': json.dumps([entity_dict]),
            'model': 'en_core_web_sm',
            'timestamp': datetime.now(tz=timezone.utc).isoformat(),
        }
        
        mock_redis_client.lrange.return_value = ['1']
        mock_pipeline = Mock()
        mock_pipeline.execute.return_value = [raw_record]
        mock_redis_client.pipeline.return_value = mock_pipeline
        
        records = redis_history.all()
        
        assert isinstance(records, list)

    @pytest.mark.unit
    def test_count(self, redis_history, mock_redis_client):
        """Test getting prediction count."""
        mock_redis_client.get.return_value = "42"
        
        count = redis_history.count()
        
        assert count == 42

    @pytest.mark.unit
    def test_count_zero(self, redis_history, mock_redis_client):
        """Test count when no predictions exist."""
        mock_redis_client.get.return_value = None
        
        count = redis_history.count()
        
        assert count == 0

    @pytest.mark.unit
    def test_ping_success(self, redis_history, mock_redis_client):
        """Test successful ping."""
        mock_redis_client.ping.return_value = True
        
        result = redis_history.ping()
        
        assert result is True

    @pytest.mark.unit
    def test_ping_failure(self, redis_history, mock_redis_client):
        """Test ping failure handling."""
        import redis
        mock_redis_client.ping.side_effect = redis.RedisError("Connection failed")
        
        result = redis_history.ping()
        
        assert result is False

    @pytest.mark.unit
    def test_record_key_format(self):
        """Test _record_key produces correct format."""
        from app.storage.redis_history import _record_key
        
        key = _record_key(123)
        assert key == "prediction:123"

    @pytest.mark.unit
    def test_deserialise_record(self, redis_history, sample_entity_result):
        """Test deserialising a record from Redis."""
        entity_dict = sample_entity_result.model_dump()
        raw = {
            'id': '1',
            'input_text': 'Test text',
            'output': json.dumps([entity_dict]),
            'model': 'en_core_web_sm',
            'timestamp': datetime.now(tz=timezone.utc).isoformat(),
        }
        
        record = redis_history._deserialise(raw)
        
        assert record.id == 1
        assert record.input_text == 'Test text'
        assert record.model == 'en_core_web_sm'
        assert len(record.output) == 1

    @pytest.mark.unit
    def test_add_with_empty_output(self, redis_history, mock_redis_client):
        """Test adding prediction with no entities."""
        mock_redis_client.incr.return_value = 1
        mock_pipeline = Mock()
        mock_redis_client.pipeline.return_value = mock_pipeline
        
        timestamp = datetime.now(tz=timezone.utc)
        
        record = redis_history.add(
            input_text="No entities here",
            output=[],
            model="en_core_web_sm",
            timestamp=timestamp,
        )
        
        assert len(record.output) == 0

    @pytest.mark.unit
    def test_all_skip_expired_records(self, redis_history, mock_redis_client):
        """Test that expired records are skipped."""
        mock_redis_client.lrange.return_value = ['1', '2']
        mock_pipeline = Mock()
        mock_pipeline.execute.return_value = [{}, {}]
        mock_redis_client.pipeline.return_value = mock_pipeline
        
        records = redis_history.all()
        
        assert isinstance(records, list)

