"""
Integration tests for the GET /health/ check endpoint.
"""

import pytest
from fastapi.testclient import TestClient

from app.storage import get_history
from app.services import get_model_manager
from app.main import app
from tests.fakes.history_fake import FakePredictionHistory
from tests.fakes.spacy_fake import FakeModelManager

fake_history = FakePredictionHistory()
fake_manager = FakeModelManager()


@pytest.fixture(autouse=True)
def override_dependencies():
    """Injects Fakes into FastAPI."""
    app.dependency_overrides[get_history] = lambda: fake_history
    app.dependency_overrides[get_model_manager] = lambda: fake_manager
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_check_healthy_mongo(self, client: TestClient) -> None:
        """Verifies health check when MongoDB is up."""
        fake_history._is_healthy = True
        response = client.get("/health/")
        
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["mongodb_connected"] is True

    def test_health_check_mongo_down(self, client: TestClient) -> None:
        """Verifies health check flags when MongoDB is down."""
        fake_history._is_healthy = False
        response = client.get("/health/")
        
        assert response.status_code == 200
        assert response.json()["mongodb_connected"] is False