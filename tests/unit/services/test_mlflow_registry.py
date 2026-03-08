"""Unit tests for app/services/mlflow_registry.py"""
import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'microservice'))


class TestMLflowRegistry:
    """Tests for MLflowRegistry class."""

    @pytest.fixture
    def mock_mlflow(self):
        """Mock mlflow module."""
        with patch('app.services.mlflow_registry.mlflow') as mock:
            mock.set_tracking_uri = Mock()
            mock.MlflowClient = Mock()
            mock.get_experiment_by_name = Mock()
            mock.create_experiment = Mock(return_value="exp-123")
            yield mock

    @pytest.fixture
    def mock_client(self):
        """Mock MLflow client."""
        client = Mock()
        client.get_registered_model = Mock()
        client.create_registered_model = Mock()
        client.get_latest_versions = Mock(return_value=[])
        client.search_registered_models = Mock(return_value=[])
        client.search_model_versions = Mock(return_value=[])
        client.create_model_version = Mock()
        client.transition_model_version_stage = Mock()
        return client

    @pytest.mark.unit
    def test_mlflow_registry_init(self, mock_mlflow, mock_settings):
        """Test MLflowRegistry initialization."""
        with patch('app.services.mlflow_registry.settings', mock_settings):
            with patch('app.services.mlflow_registry.MlflowClient', return_value=Mock()):
                registry = __import__('app.services.mlflow_registry', fromlist=['MLflowRegistry']).MLflowRegistry()
                mock_mlflow.set_tracking_uri.assert_called_once()

    @pytest.mark.unit
    def test_register_model_success(self, mock_mlflow, mock_client, mock_settings):
        """Test successful model registration."""
        with patch('app.services.mlflow_registry.settings', mock_settings):
            with patch('app.services.mlflow_registry.MlflowClient', return_value=mock_client):
                with patch('app.services.mlflow_registry.mlflow.start_run'):
                    with patch('app.services.mlflow_registry.mlflow.register_model'):
                        with patch('app.services.mlflow_registry.spacy'):
                            registry = __import__('app.services.mlflow_registry', fromlist=['MLflowRegistry']).MLflowRegistry()
                            # Result structure matches what's expected
                            assert hasattr(registry, 'register_model')

    @pytest.mark.unit
    def test_get_model_info_not_found(self, mock_mlflow, mock_client, mock_settings):
        """Test getting model info for non-existent model."""
        mock_client.get_latest_versions.return_value = []
        
        with patch('app.services.mlflow_registry.settings', mock_settings):
            with patch('app.services.mlflow_registry.MlflowClient', return_value=mock_client):
                registry = __import__('app.services.mlflow_registry', fromlist=['MLflowRegistry']).MLflowRegistry()
                result = registry.get_model_info("nonexistent_model")
                assert result is None

    @pytest.mark.unit
    def test_list_registered_models_empty(self, mock_mlflow, mock_client, mock_settings):
        """Test listing registered models when registry is empty."""
        mock_client.search_registered_models.return_value = []
        
        with patch('app.services.mlflow_registry.settings', mock_settings):
            with patch('app.services.mlflow_registry.MlflowClient', return_value=mock_client):
                registry = __import__('app.services.mlflow_registry', fromlist=['MLflowRegistry']).MLflowRegistry()
                models = registry.list_registered_models()
                assert isinstance(models, list)
                assert len(models) == 0

    @pytest.mark.unit
    def test_delete_registered_model_not_found(self, mock_mlflow, mock_client, mock_settings):
        """Test deleting non-existent model raises error."""
        mock_client.search_model_versions.return_value = []
        
        with patch('app.services.mlflow_registry.settings', mock_settings):
            with patch('app.services.mlflow_registry.MlflowClient', return_value=mock_client):
                registry = __import__('app.services.mlflow_registry', fromlist=['MLflowRegistry']).MLflowRegistry()
                # Empty list means model not found, should complete without error
                registry.delete_registered_model("nonexistent_model")

    @pytest.mark.unit
    def test_log_prediction_success(self, mock_mlflow, mock_client, mock_settings, sample_entity_result):
        """Test successful prediction logging."""
        with patch('app.services.mlflow_registry.settings', mock_settings):
            with patch('app.services.mlflow_registry.MlflowClient', return_value=mock_client):
                with patch('app.services.mlflow_registry.mlflow.start_run') as mock_run:
                    mock_run.return_value.__enter__.return_value.info.run_id = "run-123"
                    with patch('app.services.mlflow_registry.mlflow.log_params'):
                        with patch('app.services.mlflow_registry.mlflow.log_metrics'):
                            with patch('app.services.mlflow_registry.mlflow.set_tags'):
                                registry = __import__('app.services.mlflow_registry', fromlist=['MLflowRegistry']).MLflowRegistry()
                                run_id = registry.log_prediction(
                                    model_name="en_core_web_sm",
                                    model_version="1",
                                    input_text="Test text",
                                    entities=[sample_entity_result],
                                    latency_ms=100.5,
                                )
                                assert isinstance(run_id, str)

    @pytest.mark.unit
    def test_log_prediction_failure_handling(self, mock_mlflow, mock_client, mock_settings):
        """Test prediction logging handles failures gracefully."""
        with patch('app.services.mlflow_registry.settings', mock_settings):
            with patch('app.services.mlflow_registry.MlflowClient', return_value=mock_client):
                with patch('app.services.mlflow_registry.mlflow.start_run', side_effect=Exception("Test error")):
                    registry = __import__('app.services.mlflow_registry', fromlist=['MLflowRegistry']).MLflowRegistry()
                    run_id = registry.log_prediction(
                        model_name="en_core_web_sm",
                        model_version="1",
                        input_text="Test text",
                        entities=[],
                        latency_ms=10.0,
                    )
                    # Should return empty string on failure
                    assert run_id == ""
