"""
Integration tests for the Models management router.
"""

import pytest
from fastapi.testclient import TestClient

from app.services import get_model_manager, get_mlflow_registry
from app.main import app
from tests.fakes.mlflow_fake import FakeMLflowRegistry
from tests.fakes.spacy_fake import FakeModelManager

fake_manager = FakeModelManager()
fake_registry = FakeMLflowRegistry()


@pytest.fixture(autouse=True)
def override_dependencies():
    """Injects Fakes into FastAPI."""
    app.dependency_overrides[get_model_manager] = lambda: fake_manager
    app.dependency_overrides[get_mlflow_registry] = lambda: fake_registry
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestModelsRouter:
    def test_load_model_success(self, client: TestClient) -> None:
        """Verifies POST /load/ successfully downloads and registers a model."""
        payload = {"model": "en_core_web_sm"}
        response = client.post("/load/", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "en_core_web_sm"
        assert data["status"] == "downloaded"

    def test_delete_model_success(self, client: TestClient) -> None:
        """Verifies DELETE /models/{name} archives the model properly."""
        fake_registry.register_model("en_core_web_sm")
        response = client.delete("/models/en_core_web_sm")

        assert response.status_code == 200
        assert fake_registry.get_model_info("en_core_web_sm")["stage"] == "Archived"