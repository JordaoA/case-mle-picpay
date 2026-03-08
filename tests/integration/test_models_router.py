"""
Integration tests for the Models API router.

Tests cover:
    - POST   /load/              — Download and register a spaCy model
    - GET    /models/            — List all registered models
    - DELETE /models/{model_name}— Archive and evict a model
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from datetime import datetime, timezone

from app.main import app
from app.schemas.requests import ModelInfo, LoadModelResponse


@pytest.fixture
def client():
    """FastAPI TestClient for the application."""
    return TestClient(app)


@pytest.fixture
def mock_model_manager():
    """Mock ModelManager service."""
    manager = Mock()
    manager.ensure_available.return_value = "downloaded"
    manager.list_models.return_value = [
        {"name": "en_core_web_sm", "status": "loaded"},
        {"name": "en_core_web_md", "status": "available"},
    ]
    manager.delete.return_value = None
    return manager


@pytest.fixture
def mock_mlflow_registry():
    """Mock MLflow registry service."""
    registry = Mock()
    registry.get_model_info.return_value = {
        "version": "1.0.0",
        "stage": "Production",
        "run_id": "abc123def456",
    }
    return registry


class TestLoadModelEndpoint:
    """Tests for POST /load/ endpoint."""

    @patch("app.routers.models.model_manager")
    @patch("app.routers.models.mlflow_registry")
    def test_load_model_success_downloaded(
        self, mock_mlflow, mock_manager, client, mock_model_manager, mock_mlflow_registry
    ):
        """Test successful model download and registration."""
        mock_manager.ensure_available = mock_model_manager.ensure_available
        mock_mlflow.get_model_info = mock_mlflow_registry.get_model_info

        response = client.post(
            "/load/",
            json={"model": "en_core_web_sm"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "en_core_web_sm"
        assert data["status"] == "downloaded"
        assert data["message"] == "Model 'en_core_web_sm' downloaded and registered in MLflow."

    @patch("app.routers.models.model_manager")
    @patch("app.routers.models.mlflow_registry")
    def test_load_model_success_already_available(
        self, mock_mlflow, mock_manager, client, mock_model_manager, mock_mlflow_registry
    ):
        """Test loading a model that's already available."""
        mock_manager.ensure_available = Mock(return_value="already_available")
        mock_mlflow.get_model_info = mock_mlflow_registry.get_model_info

        response = client.post(
            "/load/",
            json={"model": "en_core_web_md"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "en_core_web_md"
        assert data["status"] == "already_available"
        assert data["message"] == "Model 'en_core_web_md' already available — new version registered."

    @patch("app.routers.models.model_manager")
    def test_load_model_invalid_model_name(self, mock_manager, client):
        """Test loading an invalid model name raises bad request."""
        mock_manager.ensure_available.side_effect = ValueError("Invalid model 'unknown_model'")

        response = client.post(
            "/load/",
            json={"model": "unknown_model"}
        )

        assert response.status_code == 400
        assert "Invalid model" in response.json()["detail"]

    @patch("app.routers.models.model_manager")
    def test_load_model_download_error(self, mock_manager, client):
        """Test handling download errors."""
        mock_manager.ensure_available.side_effect = RuntimeError("Download failed: network error")

        response = client.post(
            "/load/",
            json={"model": "en_core_web_sm"}
        )

        assert response.status_code == 500
        assert "Download failed" in response.json()["detail"]

    def test_load_model_missing_request_body(self, client):
        """Test POST /load/ with missing required field."""
        response = client.post(
            "/load/",
            json={}
        )

        assert response.status_code == 422

    def test_load_model_extra_fields_ignored(self, client):
        """Test that extra fields in request are ignored."""
        with patch("app.routers.models.model_manager") as mock_manager, \
             patch("app.routers.models.mlflow_registry") as mock_mlflow:
            mock_manager.ensure_available.return_value = "downloaded"
            mock_mlflow.get_model_info.return_value = {
                "version": "1.0.0",
                "stage": "Production",
                "run_id": "abc123",
            }

            response = client.post(
                "/load/",
                json={
                    "model": "en_core_web_sm",
                    "extra_field": "should_be_ignored"
                }
            )

            assert response.status_code == 200


class TestListModelsEndpoint:
    """Tests for GET /models/ endpoint."""

    @patch("app.routers.models.model_manager")
    def test_list_models_success(self, mock_manager, client, mock_model_manager):
        """Test successful retrieval of all models."""
        mock_manager.list_models = mock_model_manager.list_models

        response = client.get("/models/")

        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) == 2
        assert data["models"][0]["name"] == "en_core_web_sm"
        assert data["models"][1]["name"] == "en_core_web_md"

    @patch("app.routers.models.model_manager")
    def test_list_models_empty(self, mock_manager, client):
        """Test list models when registry is empty."""
        mock_manager.list_models.return_value = []

        response = client.get("/models/")

        assert response.status_code == 200
        data = response.json()
        assert data["models"] == []

    @patch("app.routers.models.model_manager")
    def test_list_models_response_format(self, mock_manager, client):
        """Test that response follows expected schema."""
        mock_manager.list_models.return_value = [
            {
                "name": "en_core_web_sm",
                "status": "loaded",
            }
        ]

        response = client.get("/models/")

        assert response.status_code == 200
        data = response.json()
        model = data["models"][0]
        assert "name" in model
        assert "status" in model


class TestDeleteModelEndpoint:
    """Tests for DELETE /models/{model_name} endpoint."""

    @patch("app.routers.models.model_manager")
    def test_delete_model_success(self, mock_manager, client, mock_model_manager):
        """Test successful model deletion."""
        mock_manager.delete = mock_model_manager.delete

        response = client.delete("/models/en_core_web_sm")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "archived" in data["message"].lower()
        mock_manager.delete.assert_called_once_with("en_core_web_sm")

    @patch("app.routers.models.model_manager")
    def test_delete_nonexistent_model(self, mock_manager, client):
        """Test deleting a model that doesn't exist."""
        mock_manager.delete.side_effect = ValueError("Model 'nonexistent' not found")

        response = client.delete("/models/nonexistent")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @patch("app.routers.models.model_manager")
    def test_delete_model_with_special_characters(self, mock_manager, client):
        """Test deleting models with special characters in name."""
        mock_manager.delete.return_value = None

        response = client.delete("/models/en_core_web_sm-test")

        assert response.status_code == 200
        mock_manager.delete.assert_called_once_with("en_core_web_sm-test")

    @patch("app.routers.models.model_manager")
    def test_delete_model_case_sensitive(self, mock_manager, client):
        """Test that model names are preserved as sent."""
        mock_manager.delete.return_value = None

        response = client.delete("/models/EN_CORE_WEB_SM")

        assert response.status_code == 200
        # Verify the exact case was passed to delete
        mock_manager.delete.assert_called_once_with("EN_CORE_WEB_SM")

    @patch("app.routers.models.model_manager")
    def test_delete_model_service_error(self, mock_manager, client):
        """Test that unhandled errors during deletion raise exceptions."""
        mock_manager.delete.side_effect = RuntimeError("Database connection failed")

        # RuntimeError is not caught by the router, so it will bubble up
        # The TestClient will convert it to a 500 response in production,
        # but in tests it raises the exception
        with pytest.raises(RuntimeError):
            client.delete("/models/en_core_web_sm")


class TestModelsRouterIntegration:
    """Integration tests simulating full workflows."""

    @patch("app.routers.models.model_manager")
    @patch("app.routers.models.mlflow_registry")
    def test_load_list_delete_workflow(
        self, mock_mlflow, mock_manager, client, mock_model_manager, mock_mlflow_registry
    ):
        """Test complete workflow: load model -> list -> delete."""
        # Load a model
        mock_manager.ensure_available = mock_model_manager.ensure_available
        mock_mlflow.get_model_info = mock_mlflow_registry.get_model_info

        load_response = client.post(
            "/load/",
            json={"model": "en_core_web_sm"}
        )
        assert load_response.status_code == 200

        # List models
        mock_manager.list_models = Mock(return_value=[
            {"name": "en_core_web_sm", "status": "loaded"}
        ])

        list_response = client.get("/models/")
        assert list_response.status_code == 200
        assert len(list_response.json()["models"]) > 0

        # Delete the model
        mock_manager.delete = mock_model_manager.delete
        delete_response = client.delete("/models/en_core_web_sm")
        assert delete_response.status_code == 200

    @patch("app.routers.models.model_manager")
    def test_list_models_multiple_calls(self, mock_manager, client):
        """Test multiple consecutive list operations return consistent data."""
        models_data = [
            {"name": "model_1", "status": "loaded"},
            {"name": "model_2", "status": "available"},
        ]
        mock_manager.list_models.return_value = models_data

        response1 = client.get("/models/")
        response2 = client.get("/models/")

        assert response1.json() == response2.json()
        assert mock_manager.list_models.call_count == 2
