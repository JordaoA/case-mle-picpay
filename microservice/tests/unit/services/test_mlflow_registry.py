"""
unit/services/test_mlflow_registry.py
---------------------------------------
MLflowRegistry — all methods tested with manually injected mocks
to guarantee scope isolation and avoid Pytest decorator clashing.
"""

import pytest
from unittest.mock import patch, MagicMock
import mlflow.exceptions

from app.services.mlflow_registry import MLflowRegistry


class DummyVersion:
    """A clean data class to mimic MLflow's ModelVersion object perfectly."""
    def __init__(self, name, version, current_stage, run_id="run-1", description="desc", creation_timestamp=1600000000000):
        self.name = name
        self.version = version
        self.current_stage = current_stage
        self.run_id = run_id
        self.description = description
        self.creation_timestamp = creation_timestamp


@pytest.fixture
def registry():
    """
    Patches the mlflow modules and keeps them open using `yield`.
    """
    with patch("app.services.mlflow_registry.mlflow") as mock_mlflow, \
         patch("app.services.mlflow_registry.MlflowClient") as MockClient:
            
        mock_mlflow.exceptions.MlflowException = mlflow.exceptions.MlflowException
            
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "test-exp-123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        
        reg = MLflowRegistry()
        
        reg._client = MockClient.return_value
        reg._mock_mlflow = mock_mlflow 
        
        yield reg


class TestMLflowRegistry:

    def test_register_model(self, registry):
        mock_client = registry._client
        mock_mlflow = registry._mock_mlflow
        
        mock_client.get_registered_model.side_effect = mlflow.exceptions.MlflowException("Not found")
        
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-123"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        mock_mlflow.register_model.side_effect = Exception("Force fallback")
        mock_client.create_model_version.return_value = DummyVersion("en_core_web_sm", "1", "None")
        
        mock_client.get_latest_versions.return_value = []

        result = registry.register_model("en_core_web_sm")

        mock_client.create_registered_model.assert_called_once()
        mock_client.transition_model_version_stage.assert_called_with(
            name="en_core_web_sm", version="1", stage="Production"
        )
        assert result["version"] == "1"

    def test_get_model_info(self, registry):
        mock_client = registry._client
        
        dummy = DummyVersion(
            name="en_core_web_sm",
            version="2",
            current_stage="Production",
            run_id="run-456",
            description="Test description",
            creation_timestamp=1600000000000
        )
        mock_client.get_latest_versions.return_value = [dummy]

        info = registry.get_model_info("en_core_web_sm")

        assert info["name"] == "en_core_web_sm"
        assert info["version"] == "2"
        assert info["stage"] == "Production"
        assert info["run_id"] == "run-456"

    def test_delete_registered_model(self, registry):
        mock_client = registry._client
        
        dummy = DummyVersion("en_core_web_sm", "1", "Production")
        mock_client.search_model_versions.return_value = [dummy]

        registry.delete_registered_model("en_core_web_sm")

        mock_client.transition_model_version_stage.assert_called_once_with(
            name="en_core_web_sm", version="1", stage="Archived"
        )