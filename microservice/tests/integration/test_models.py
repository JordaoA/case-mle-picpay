"""
integration/test_models.py
---------------------------
GET /models/  — list from registry
DELETE /models/{name}  — evict and archive
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import get_model_manager, get_mlflow_registry
from tests.fakes.spacy_fake import FakeModelManager
from tests.fakes.mlflow_fake import FakeMLflowRegistry

fake_manager = FakeModelManager()
fake_registry = FakeMLflowRegistry()

@pytest.fixture(autouse=True)
def override_dependencies():
    app.dependency_overrides[get_model_manager] = lambda: fake_manager
    app.dependency_overrides[get_mlflow_registry] = lambda: fake_registry
    yield
    app.dependency_overrides.clear()

@pytest.fixture
def client():
    return TestClient(app)

class TestModelsRouter:

    def test_load_model_success(self, client):
        payload = {"model": "en_core_web_sm"}
        response = client.post("/load/", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "en_core_web_sm"
        assert data["status"] == "downloaded"
        assert data["mlflow_version"] == "1"
        assert data["mlflow_stage"] == "Production"

    def test_load_model_validation_error(self, client):
        payload = {"model": "invalid_model"}
        response = client.post("/load/", json=payload)
        
        assert response.status_code in [400, 422]

    def test_list_models_merges_data(self, client):
        fake_registry.register_model("en_core_web_sm")
    
        response = client.get("/models/")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["models"]) >= 1
        
        model_info = next(m for m in data["models"] if m["name"] == "en_core_web_sm")
        assert model_info["stage"] == "Production"
        assert model_info["loaded"] is True

    def test_delete_model_success(self, client):
        fake_registry.register_model("en_core_web_sm")

        response = client.delete("/models/en_core_web_sm")

        assert response.status_code == 200
        assert response.json()["message"] == "Model 'en_core_web_sm' archived and evicted."
        assert fake_registry.get_model_info("en_core_web_sm")["stage"] == "Archived"