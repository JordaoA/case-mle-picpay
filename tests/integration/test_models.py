"""
integration/test_models.py
---------------------------
GET /models/  — list from registry
DELETE /models/{name}  — evict and archive
"""

import pytest


@pytest.mark.integration
class TestListModelsEndpoint:

    def test_returns_200(self, client):
        r = client.get("/models/")
        assert r.status_code == 200

    def test_empty_registry_returns_empty_list(self, client, mock_mm):
        mock_mm.list_models.return_value = []
        r = client.get("/models/")
        assert r.json()["models"] == []

    def test_returns_registered_models(self, client, mock_mm):
        mock_mm.list_models.return_value = [
            {"name": "en_core_web_sm", "version": "1", "stage": "Production",
             "loaded": True, "run_id": "r1"},
        ]
        r = client.get("/models/")
        models = r.json()["models"]
        assert len(models) == 1
        assert models[0]["name"] == "en_core_web_sm"

    def test_loaded_flag_true_when_in_cache(self, client, mock_mm):
        mock_mm.list_models.return_value = [
            {"name": "en_core_web_sm", "version": "1", "stage": "Production",
             "loaded": True, "run_id": None},
        ]
        r = client.get("/models/")
        assert r.json()["models"][0]["loaded"] is True

    def test_loaded_flag_false_when_not_cached(self, client, mock_mm):
        mock_mm.list_models.return_value = [
            {"name": "en_core_web_md", "version": "2", "stage": "Production",
             "loaded": False, "run_id": None},
        ]
        r = client.get("/models/")
        assert r.json()["models"][0]["loaded"] is False

    def test_multiple_models_returned(self, client, mock_mm):
        mock_mm.list_models.return_value = [
            {"name": "en_core_web_sm", "version": "1", "stage": "Production",
             "loaded": True,  "run_id": None},
            {"name": "en_core_web_md", "version": "1", "stage": "Production",
             "loaded": False, "run_id": None},
        ]
        r = client.get("/models/")
        assert len(r.json()["models"]) == 2

    def test_response_schema_has_models_key(self, client):
        r = client.get("/models/")
        assert "models" in r.json()


@pytest.mark.integration
class TestDeleteModelEndpoint:

    def test_returns_200_on_success(self, client, mock_mm):
        r = client.delete("/models/en_core_web_sm")
        assert r.status_code == 200

    def test_response_contains_confirmation_message(self, client, mock_mm):
        r = client.delete("/models/en_core_web_sm")
        assert "en_core_web_sm" in r.json()["message"]

    def test_delete_calls_model_manager(self, client, mock_mm):
        client.delete("/models/en_core_web_sm")
        mock_mm.delete.assert_called_once_with("en_core_web_sm")

    def test_not_found_returns_404(self, client, mock_mm):
        mock_mm.delete.side_effect = ValueError("Model 'en_core_web_sm' not found.")
        r = client.delete("/models/en_core_web_sm")
        assert r.status_code == 404

    def test_404_detail_contains_reason(self, client, mock_mm):
        mock_mm.delete.side_effect = ValueError("not found")
        r = client.delete("/models/en_core_web_sm")
        assert "not found" in r.json()["detail"]

    def test_invalid_model_name_returns_404(self, client, mock_mm):
        mock_mm.delete.side_effect = ValueError("not supported")
        r = client.delete("/models/invalid_model_name")
        assert r.status_code == 404