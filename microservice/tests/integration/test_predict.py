"""
Integration tests for the POST /predict/ endpoint.
"""

import pytest
from fastapi.testclient import TestClient

from app.storage import get_history
from app.services import get_model_manager, get_mlflow_registry
from app.main import app
from tests.fakes.history_fake import FakePredictionHistory
from tests.fakes.mlflow_fake import FakeMLflowRegistry
from tests.fakes.spacy_fake import FakeModelManager

fake_manager = FakeModelManager()
fake_history = FakePredictionHistory()
fake_registry = FakeMLflowRegistry()


@pytest.fixture(autouse=True)
def override_dependencies():
    """Injects Fakes into the FastAPI application for all tests in this module."""
    app.dependency_overrides[get_model_manager] = lambda: fake_manager
    app.dependency_overrides[get_history] = lambda: fake_history
    app.dependency_overrides[get_mlflow_registry] = lambda: fake_registry
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestPredictEndpoint:
    def test_predict_returns_entities_and_saves_to_db(self, client: TestClient) -> None:
        """Verifies the complete HTTP request lifecycle for a valid prediction."""
        fake_registry.register_model("en_core_web_sm")
        payload = {
            "text": "Apple is opening a store in New York.",
            "model": "en_core_web_sm",
        }

        response = client.post("/predict/", json=payload)

        assert response.status_code == 200
        data = response.json()

        assert len(data["entities"]) == 2
        assert data["entities"][0]["text"] == "Apple"
        assert fake_history.count() == 1