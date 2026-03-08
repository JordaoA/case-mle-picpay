"""Unit tests for in-memory history implementation."""
import pytest
import sys
import os
from unittest.mock import patch
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'microservice'))


class TestInMemoryHistory:
    """Tests for in-memory prediction history."""

    @pytest.mark.unit
    def test_add_single_prediction(self, sample_entity_result):
        """Test adding a single prediction."""
        from app.storage.history import PredictionHistory
        
        history = PredictionHistory()
        timestamp = datetime.now(tz=timezone.utc)
        
        record = history.add(
            input_text="John eats pizza",
            output=[sample_entity_result],
            model="en_core_web_sm",
            timestamp=timestamp,
        )
        
        assert record.id > 0
        assert record.input_text == "John eats pizza"
        assert record.model == "en_core_web_sm"
        assert len(record.output) == 1
        assert record.timestamp is not None

    @pytest.mark.unit
    def test_add_prediction_increments_counter(self, sample_entity_result):
        """Test that counter increments correctly."""
        from app.storage.history import PredictionHistory
        
        history = PredictionHistory()
        timestamp = datetime.now(tz=timezone.utc)
        
        record1 = history.add(
            input_text="Text 1",
            output=[],
            model="en_core_web_sm",
            timestamp=timestamp,
        )
        record2 = history.add(
            input_text="Text 2",
            output=[],
            model="en_core_web_sm",
            timestamp=timestamp,
        )
        record3 = history.add(
            input_text="Text 3",
            output=[],
            model="en_core_web_sm",
            timestamp=timestamp,
        )
        
        assert record1.id == 1
        assert record2.id == 2
        assert record3.id == 3

    @pytest.mark.unit
    def test_all_returns_list(self, sample_entity_result):
        """Test that all() returns a list of predictions."""
        from app.storage.history import PredictionHistory
        
        history = PredictionHistory()
        timestamp = datetime.now(tz=timezone.utc)
        
        history.add(
            input_text="Test",
            output=[sample_entity_result],
            model="en_core_web_sm",
            timestamp=timestamp,
        )
        
        records = history.all()
        assert isinstance(records, list)
        assert len(records) == 1

    @pytest.mark.unit
    def test_all_returns_copy(self, sample_entity_result):
        """Test that all() returns a shallow copy (doesn't share reference)."""
        from app.storage.history import PredictionHistory
        
        history = PredictionHistory()
        timestamp = datetime.now(tz=timezone.utc)
        
        history.add(
            input_text="Test",
            output=[sample_entity_result],
            model="en_core_web_sm",
            timestamp=timestamp,
        )
        
        records1 = history.all()
        records2 = history.all()
        
        assert records1 == records2
        assert records1 is not records2

    @pytest.mark.unit
    def test_count_accuracy(self, sample_entity_result):
        """Test that count() returns accurate count."""
        from app.storage.history import PredictionHistory
        
        history = PredictionHistory()
        timestamp = datetime.now(tz=timezone.utc)
        
        for i in range(5):
            history.add(
                input_text=f"Text {i}",
                output=[sample_entity_result],
                model="en_core_web_sm",
                timestamp=timestamp,
            )
        
        assert history.count() == 5

    @pytest.mark.unit
    def test_with_empty_output(self):
        """Test adding prediction with no entities."""
        from app.storage.history import PredictionHistory
        
        history = PredictionHistory()
        timestamp = datetime.now(tz=timezone.utc)
        
        record = history.add(
            input_text="Text with no entities",
            output=[],
            model="en_core_web_sm",
            timestamp=timestamp,
        )
        
        assert len(record.output) == 0

    @pytest.mark.unit
    def test_with_multiple_entities(self):
        """Test adding prediction with multiple entities."""
        from app.storage.history import PredictionHistory
        from app.schemas.requests import EntityResult
        
        history = PredictionHistory()
        timestamp = datetime.now(tz=timezone.utc)
        
        entities = [
            EntityResult(label="PERSON", text="John", start=0, end=4),
            EntityResult(label="ORG", text="Google", start=13, end=19),
            EntityResult(label="DATE", text="Monday", start=24, end=30),
        ]
        
        record = history.add(
            input_text="John works at Google on Monday",
            output=entities,
            model="en_core_web_lg",
            timestamp=timestamp,
        )
        
        assert len(record.output) == 3

    @pytest.mark.unit
    def test_record_attributes(self, sample_entity_result):
        """Test that returned record has all expected attributes."""
        from app.storage.history import PredictionHistory
        
        history = PredictionHistory()
        timestamp = datetime.now(tz=timezone.utc)
        
        record = history.add(
            input_text="Test text",
            output=[sample_entity_result],
            model="en_core_web_sm",
            timestamp=timestamp,
        )
        
        assert hasattr(record, 'id')
        assert hasattr(record, 'input_text')
        assert hasattr(record, 'output')
        assert hasattr(record, 'model')
        assert hasattr(record, 'timestamp')

    @pytest.mark.unit
    def test_ping_health_check(self):
        """Test ping returns True for health check."""
        from app.storage.history import PredictionHistory
        
        history = PredictionHistory()
        assert history.ping() is True

