"""
Integration tests for the Predictions API router.

Tests cover:
    - POST /predict/  — Run NER inference on a text
    - GET  /list/     — Return the full prediction history
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from datetime import datetime, timezone

from app.main import app
from app.schemas.requests import EntityResult, PredictionRecord


@pytest.fixture
def client():
    """FastAPI TestClient for the application."""
    return TestClient(app)


@pytest.fixture
def sample_entities():
    """Sample entities for prediction responses."""
    return [
        {"label": "PERSON", "text": "Michael", "start": 32, "end": 39},
        {"label": "MONEY", "text": "$45", "start": 15, "end": 18},
        {"label": "DATE", "text": "June 3", "start": 43, "end": 49},
    ]


@pytest.fixture
def mock_prediction_history():
    """Mock prediction history storage."""
    history = Mock()
    history.all.return_value = [
        {
            "id": 1,
            "input_text": "Can you send $45 to Michael on June 3?",
            "output": [
                {"label": "MONEY", "text": "$45", "start": 15, "end": 18},
                {"label": "PERSON", "text": "Michael", "start": 32, "end": 39},
                {"label": "DATE", "text": "June 3", "start": 43, "end": 49},
            ],
            "model": "en_core_web_sm",
            "timestamp": datetime.now(tz=timezone.utc),
        }
    ]
    history.add.return_value = None
    return history


class TestPredictEndpoint:
    """Tests for POST /predict/ endpoint."""

    @patch("app.routers.predictions.run_prediction")
    def test_predict_success(self, mock_run_prediction, client, sample_entities):
        """Test successful NER prediction."""
        mock_run_prediction.return_value = {
            "model": "en_core_web_sm",
            "model_version": "3.5.0",
            "text": "Can you send $45 to Michael on June 3?",
            "entities": sample_entities,
            "timestamp": datetime.now(tz=timezone.utc),
        }

        response = client.post(
            "/predict/",
            json={
                "text": "Can you send $45 to Michael on June 3?",
                "model": "en_core_web_sm"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "en_core_web_sm"
        assert data["text"] == "Can you send $45 to Michael on June 3?"
        assert len(data["entities"]) == 3
        assert data["entities"][0]["label"] == "PERSON"

    @patch("app.routers.predictions.run_prediction")
    def test_predict_empty_entities(self, mock_run_prediction, client):
        """Test prediction on text with no recognizable entities."""
        mock_run_prediction.return_value = {
            "model": "en_core_web_sm",
            "model_version": "3.5.0",
            "text": "Hello world",
            "entities": [],
            "timestamp": datetime.now(tz=timezone.utc),
        }

        response = client.post(
            "/predict/",
            json={
                "text": "Hello world",
                "model": "en_core_web_sm"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["entities"] == []

    @patch("app.routers.predictions.run_prediction")
    def test_predict_model_not_found(self, mock_run_prediction, client):
        """Test prediction with unknown model."""
        mock_run_prediction.side_effect = ValueError("Model 'unknown_model' not found")

        response = client.post(
            "/predict/",
            json={
                "text": "Test text",
                "model": "unknown_model"
            }
        )

        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()

    @patch("app.routers.predictions.run_prediction")
    def test_predict_runtime_error(self, mock_run_prediction, client):
        """Test handling of runtime errors during prediction."""
        mock_run_prediction.side_effect = RuntimeError("Model loading failed")

        response = client.post(
            "/predict/",
            json={
                "text": "Test text",
                "model": "en_core_web_sm"
            }
        )

        assert response.status_code == 500
        assert "Inference failed" in response.json()["detail"]

    @patch("app.routers.predictions.run_prediction")
    def test_predict_generic_exception(self, mock_run_prediction, client):
        """Test handling of unexpected exceptions."""
        mock_run_prediction.side_effect = Exception("Unexpected error")

        response = client.post(
            "/predict/",
            json={
                "text": "Test text",
                "model": "en_core_web_sm"
            }
        )

        assert response.status_code == 500
        assert "Inference failed" in response.json()["detail"]

    def test_predict_missing_text(self, client):
        """Test POST /predict/ with missing text field."""
        response = client.post(
            "/predict/",
            json={"model": "en_core_web_sm"}
        )

        assert response.status_code == 422

    def test_predict_missing_model(self, client):
        """Test POST /predict/ with missing model field."""
        response = client.post(
            "/predict/",
            json={"text": "Test text"}
        )

        assert response.status_code == 422

    @patch("app.routers.predictions.run_prediction")
    def test_predict_long_text(self, mock_run_prediction, client):
        """Test prediction with very long text."""
        long_text = "This is a test. " * 500  # ~8000 characters
        mock_run_prediction.return_value = {
            "model": "en_core_web_sm",
            "model_version": "3.5.0",
            "text": long_text,
            "entities": [],
            "timestamp": datetime.now(tz=timezone.utc),
        }

        response = client.post(
            "/predict/",
            json={
                "text": long_text,
                "model": "en_core_web_sm"
            }
        )

        assert response.status_code == 200
        assert response.json()["text"] == long_text

    @patch("app.routers.predictions.run_prediction")
    def test_predict_special_characters(self, mock_run_prediction, client):
        """Test prediction on text with special characters."""
        special_text = "Email: john@example.com, Phone: +55 11 99999-9999 #test"
        mock_run_prediction.return_value = {
            "model": "en_core_web_sm",
            "model_version": "3.5.0",
            "text": special_text,
            "entities": [{"label": "EMAIL", "text": "john@example.com", "start": 7, "end": 23}],
            "timestamp": datetime.now(tz=timezone.utc),
        }

        response = client.post(
            "/predict/",
            json={
                "text": special_text,
                "model": "en_core_web_sm"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["text"] == special_text

    @patch("app.routers.predictions.run_prediction")
    def test_predict_unicode_text(self, mock_run_prediction, client):
        """Test prediction on unicode/non-ASCII text."""
        unicode_text = "João enviou R$50 para Maria em 3 de junho"
        mock_run_prediction.return_value = {
            "model": "en_core_web_sm",
            "model_version": "3.5.0",
            "text": unicode_text,
            "entities": [{"label": "PERSON", "text": "João", "start": 0, "end": 4}],
            "timestamp": datetime.now(tz=timezone.utc),
        }

        response = client.post(
            "/predict/",
            json={
                "text": unicode_text,
                "model": "en_core_web_sm"
            }
        )

        assert response.status_code == 200


class TestListPredictionsEndpoint:
    """Tests for GET /list/ endpoint."""

    @patch("app.routers.predictions.prediction_history")
    def test_list_predictions_success(self, mock_history, client, mock_prediction_history):
        """Test successful retrieval of prediction history."""
        mock_history.all = mock_prediction_history.all

        response = client.get("/list/")

        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "predictions" in data
        assert data["total"] == 1
        assert len(data["predictions"]) == 1

    @patch("app.routers.predictions.prediction_history")
    def test_list_predictions_empty_history(self, mock_history, client):
        """Test list predictions when history is empty."""
        mock_history.all.return_value = []

        response = client.get("/list/")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["predictions"] == []

    @patch("app.routers.predictions.prediction_history")
    def test_list_predictions_multiple_records(self, mock_history, client):
        """Test retrieving multiple prediction records."""
        predictions = [
            {
                "id": 1,
                "input_text": "John loves pizza",
                "output": [{"label": "PERSON", "text": "John", "start": 0, "end": 4}],
                "model": "en_core_web_sm",
                "timestamp": datetime.now(tz=timezone.utc),
            },
            {
                "id": 2,
                "input_text": "Send $50 to Mary",
                "output": [{"label": "MONEY", "text": "$50", "start": 5, "end": 8}],
                "model": "en_core_web_md",
                "timestamp": datetime.now(tz=timezone.utc),
            },
        ]
        mock_history.all.return_value = predictions

        response = client.get("/list/")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["predictions"]) == 2

    @patch("app.routers.predictions.prediction_history")
    def test_list_predictions_response_format(self, mock_history, client):
        """Test that response follows expected schema."""
        mock_history.all.return_value = [
            {
                "id": 1,
                "input_text": "Test text",
                "output": [],
                "model": "en_core_web_sm",
                "timestamp": datetime.now(tz=timezone.utc),
            }
        ]

        response = client.get("/list/")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["total"], int)
        assert isinstance(data["predictions"], list)
        record = data["predictions"][0]
        assert "id" in record
        assert "input_text" in record
        assert "output" in record
        assert "model" in record
        assert "timestamp" in record


class TestPredictionsRouterIntegration:
    """Integration tests simulating full workflows."""

    @patch("app.routers.predictions.run_prediction")
    @patch("app.routers.predictions.prediction_history")
    def test_predict_and_list_workflow(
        self, mock_history, mock_run_prediction, client, sample_entities, mock_prediction_history
    ):
        """Test workflow: make prediction -> retrieve history."""
        # Make a prediction
        mock_run_prediction.return_value = {
            "model": "en_core_web_sm",
            "model_version": "3.5.0",
            "text": "Can you send $45 to Michael on June 3?",
            "entities": sample_entities,
            "timestamp": datetime.now(tz=timezone.utc),
        }

        predict_response = client.post(
            "/predict/",
            json={
                "text": "Can you send $45 to Michael on June 3?",
                "model": "en_core_web_sm"
            }
        )
        assert predict_response.status_code == 200

        # List predictions
        mock_history.all = mock_prediction_history.all
        list_response = client.get("/list/")
        assert list_response.status_code == 200
        assert list_response.json()["total"] > 0

    @patch("app.routers.predictions.run_prediction")
    @patch("app.routers.predictions.prediction_history")
    def test_multiple_predictions_consistency(
        self, mock_history, mock_run_prediction, client
    ):
        """Test consistency across multiple prediction operations."""
        predictions_log = [
            {
                "id": 1,
                "input_text": "First prediction",
                "output": [],
                "model": "en_core_web_sm",
                "timestamp": datetime.now(tz=timezone.utc),
            },
            {
                "id": 2,
                "input_text": "Second prediction",
                "output": [],
                "model": "en_core_web_sm",
                "timestamp": datetime.now(tz=timezone.utc),
            },
        ]
        mock_history.all.return_value = predictions_log

        # First list call
        response1 = client.get("/list/")
        assert response1.json()["total"] == 2

        # Second list call
        response2 = client.get("/list/")
        assert response2.json()["total"] == 2

        # Both responses should match
        assert response1.json() == response2.json()

    @patch("app.routers.predictions.run_prediction")
    def test_predict_different_models(self, mock_run_prediction, client):
        """Test predictions with different spaCy models."""
        test_cases = [
            ("en_core_web_sm", "Test with small model"),
            ("en_core_web_md", "Test with medium model"),
            ("en_core_web_lg", "Test with large model"),
        ]

        for model_name, text in test_cases:
            mock_run_prediction.return_value = {
                "model": model_name,
                "model_version": "3.5.0",
                "text": text,
                "entities": [],
                "timestamp": datetime.now(tz=timezone.utc),
            }

            response = client.post(
                "/predict/",
                json={"text": text, "model": model_name}
            )

            assert response.status_code == 200
            assert response.json()["model"] == model_name

    @patch("app.routers.predictions.run_prediction")
    def test_predict_consecutive_calls(self, mock_run_prediction, client):
        """Test consecutive predictions work independently."""
        mock_run_prediction.return_value = {
            "model": "en_core_web_sm",
            "model_version": "3.5.0",
            "text": "Test",
            "entities": [],
            "timestamp": datetime.now(tz=timezone.utc),
        }

        # First prediction
        response1 = client.post(
            "/predict/",
            json={"text": "First", "model": "en_core_web_sm"}
        )
        assert response1.status_code == 200

        # Second prediction
        response2 = client.post(
            "/predict/",
            json={"text": "Second", "model": "en_core_web_sm"}
        )
        assert response2.status_code == 200

        # Both should succeed independently
        assert mock_run_prediction.call_count == 2


class TestPredictionsResponseFormat:
    """Tests for response format validation."""

    @patch("app.routers.predictions.run_prediction")
    def test_entity_result_format(self, mock_run_prediction, client):
        """Test EntityResult objects have correct structure."""
        mock_run_prediction.return_value = {
            "model": "en_core_web_sm",
            "model_version": "3.5.0",
            "text": "John works in NYC",
            "entities": [
                {
                    "label": "PERSON",
                    "text": "John",
                    "start": 0,
                    "end": 4,
                },
                {
                    "label": "GPE",
                    "text": "NYC",
                    "start": 14,
                    "end": 17,
                }
            ],
            "timestamp": datetime.now(tz=timezone.utc),
        }

        response = client.post(
            "/predict/",
            json={"text": "John works in NYC", "model": "en_core_web_sm"}
        )

        data = response.json()
        for entity in data["entities"]:
            assert "label" in entity
            assert "text" in entity
            assert "start" in entity
            assert "end" in entity
            assert isinstance(entity["start"], int)
            assert isinstance(entity["end"], int)

    @patch("app.routers.predictions.run_prediction")
    def test_predict_response_timestamp_format(self, mock_run_prediction, client):
        """Test that timestamp is in correct format."""
        now = datetime.now(tz=timezone.utc)
        mock_run_prediction.return_value = {
            "model": "en_core_web_sm",
            "model_version": "3.5.0",
            "text": "Test",
            "entities": [],
            "timestamp": now,
        }

        response = client.post(
            "/predict/",
            json={"text": "Test", "model": "en_core_web_sm"}
        )

        data = response.json()
        assert "timestamp" in data
        # Verify it's a valid ISO format timestamp string
        assert isinstance(data["timestamp"], str)
