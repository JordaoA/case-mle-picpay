"""Unit tests for app/services/ner_service.py"""
import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'microservice'))


class TestNERService:
    """Tests for NER inference service."""

    @pytest.fixture
    def mock_model_manager(self):
        """Mock ModelManager."""
        manager = Mock()
        nlp = MagicMock()
        doc = MagicMock()
        
        # Create a mock entity
        ent = Mock()
        ent.label_ = "PERSON"
        ent.text = "John"
        ent.start_char = 0
        ent.end_char = 4
        doc.ents = [ent]
        
        nlp.return_value = doc
        manager.get.return_value = nlp
        return manager

    @pytest.fixture
    def mock_prediction_history(self):
        """Mock PredictionHistory."""
        history = Mock()
        history.add.return_value = Mock(id=1)
        return history

    @pytest.mark.unit
    def test_run_prediction_success(self, mock_model_manager, mock_prediction_history, sample_entity_result):
        """Test successful prediction."""
        with patch('app.services.ner_service.model_manager', mock_model_manager):
            with patch('app.services.ner_service.prediction_history', mock_prediction_history):
                run_prediction = __import__('app.services.ner_service', fromlist=['run_prediction']).run_prediction
                
                response = run_prediction(
                    text="John loves pizza",
                    model_name="en_core_web_sm"
                )
                
                assert response is not None
                assert response.model == "en_core_web_sm"
                assert response.text == "John loves pizza"
                assert len(response.entities) > 0
                assert response.timestamp is not None

    @pytest.mark.unit
    def test_run_prediction_empty_text_raises_error(self, mock_model_manager, mock_prediction_history):
        """Test that empty text raises ValueError."""
        with patch('app.services.ner_service.model_manager', mock_model_manager):
            with patch('app.services.ner_service.prediction_history', mock_prediction_history):
                run_prediction = __import__('app.services.ner_service', fromlist=['run_prediction']).run_prediction
                
                with pytest.raises(ValueError):
                    run_prediction(text="", model_name="en_core_web_sm")

    @pytest.mark.unit
    def test_run_prediction_whitespace_only_raises_error(self, mock_model_manager, mock_prediction_history):
        """Test that whitespace-only text raises ValueError."""
        with patch('app.services.ner_service.model_manager', mock_model_manager):
            with patch('app.services.ner_service.prediction_history', mock_prediction_history):
                run_prediction = __import__('app.services.ner_service', fromlist=['run_prediction']).run_prediction
                
                with pytest.raises(ValueError):
                    run_prediction(text="   ", model_name="en_core_web_sm")

    @pytest.mark.unit
    def test_run_prediction_history_logging(self, mock_model_manager, mock_prediction_history):
        """Test that prediction is logged to history."""
        with patch('app.services.ner_service.model_manager', mock_model_manager):
            with patch('app.services.ner_service.prediction_history', mock_prediction_history):
                run_prediction = __import__('app.services.ner_service', fromlist=['run_prediction']).run_prediction
                
                run_prediction(text="John loves pizza", model_name="en_core_web_sm")
                
                # Verify history.add was called
                mock_prediction_history.add.assert_called_once()
                call_args = mock_prediction_history.add.call_args
                assert call_args[1]['input_text'] == "John loves pizza"
                assert call_args[1]['model'] == "en_core_web_sm"

    @pytest.mark.unit
    def test_run_prediction_with_no_entities(self):
        """Test prediction when no entities are found."""
        mock_model_manager = Mock()
        nlp = MagicMock()
        doc = MagicMock()
        doc.ents = []  # No entities
        nlp.return_value = doc
        mock_model_manager.get.return_value = nlp
        
        mock_history = Mock()
        mock_history.add.return_value = Mock(id=1)
        
        with patch('app.services.ner_service.model_manager', mock_model_manager):
            with patch('app.services.ner_service.prediction_history', mock_history):
                run_prediction = __import__('app.services.ner_service', fromlist=['run_prediction']).run_prediction
                
                response = run_prediction(text="Hello world", model_name="en_core_web_sm")
                
                assert response is not None
                assert len(response.entities) == 0

    @pytest.mark.unit
    def test_run_prediction_response_structure(self, mock_model_manager, mock_prediction_history):
        """Test that response has correct structure."""
        with patch('app.services.ner_service.model_manager', mock_model_manager):
            with patch('app.services.ner_service.prediction_history', mock_prediction_history):
                run_prediction = __import__('app.services.ner_service', fromlist=['run_prediction']).run_prediction
                
                response = run_prediction(text="John works at Google", model_name="en_core_web_sm")
                
                # Check required fields exist
                assert hasattr(response, 'model')
                assert hasattr(response, 'text')
                assert hasattr(response, 'entities')
                assert hasattr(response, 'timestamp')
                
                # Check types
                assert isinstance(response.model, str)
                assert isinstance(response.text, str)
                assert isinstance(response.entities, list)
                assert isinstance(response.timestamp, datetime)

    @pytest.mark.unit
    def test_run_prediction_entity_structure(self, mock_model_manager, mock_prediction_history):
        """Test that entities have correct structure."""
        with patch('app.services.ner_service.model_manager', mock_model_manager):
            with patch('app.services.ner_service.prediction_history', mock_prediction_history):
                run_prediction = __import__('app.services.ner_service', fromlist=['run_prediction']).run_prediction
                
                response = run_prediction(text="John works at Google", model_name="en_core_web_sm")
                
                if len(response.entities) > 0:
                    entity = response.entities[0]
                    assert hasattr(entity, 'label')
                    assert hasattr(entity, 'text')
                    assert hasattr(entity, 'start')
                    assert hasattr(entity, 'end')
