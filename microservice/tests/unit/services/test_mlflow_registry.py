"""
unit/services/test_mlflow_registry.py
---------------------------------------
MLflowRegistry — all methods tested with a mocked MlflowClient.
No real MLflow server is contacted.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.services.mlflow_registry import MLflowRegistry

@pytest.fixture
def registry():
    # Patch mlflow.set_tracking_uri if you do it in __init__
    with patch("app.services.mlflow_registry.mlflow.set_tracking_uri"):
        return MLflowRegistry()

class TestMLflowRegistry:

    @patch("app.services.mlflow_registry.MlflowClient")
    @patch("app.services.mlflow_registry.mlflow")
    def test_register_model(self, mock_mlflow, MockClient, registry):
        mock_client_instance = MockClient.return_value
        
        # Simulate active run
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-123"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        
        # Simulate created version
        mock_version = MagicMock()
        mock_version.version = "1"
        mock_client_instance.create_model_version.return_value = mock_version

        registry.register_model("en_core_web_sm")

        mock_mlflow.start_run.assert_called_once()
        mock_client_instance.create_registered_model.assert_called_once_with("en_core_web_sm")
        mock_client_instance.transition_model_version_stage.assert_called_once_with(
            name="en_core_web_sm", version="1", stage="Production", archive_existing_versions=True
        )

    @patch("app.services.mlflow_registry.MlflowClient")
    def test_get_model_info(self, MockClient, registry):
        mock_client_instance = MockClient.return_value
        mock_version = MagicMock()
        mock_version.version = "2"
        mock_version.current_stage = "Production"
        mock_version.run_id = "run-456"
        mock_client_instance.get_latest_versions.return_value = [mock_version]

        info = registry.get_model_info("en_core_web_sm")

        assert info == {
            "name": "en_core_web_sm",
            "version": "2",
            "stage": "Production",
            "run_id": "run-456"
        }

    @patch("app.services.mlflow_registry.MlflowClient")
    def test_delete_registered_model(self, MockClient, registry):
        mock_client_instance = MockClient.return_value
        mock_version = MagicMock()
        mock_version.version = "1"
        mock_version.current_stage = "Production"
        mock_client_instance.search_model_versions.return_value = [mock_version]

        registry.delete_registered_model("en_core_web_sm")

        mock_client_instance.transition_model_version_stage.assert_called_once_with(
            name="en_core_web_sm", version="1", stage="Archived"
        )