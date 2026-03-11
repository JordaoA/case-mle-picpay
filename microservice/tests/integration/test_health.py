"""
integration/test_health.py
---------------------------
GET /health/ — status, loaded_models, backends, redis flag.
"""

import pytest


@pytest.mark.integration
class TestHealthEndpoint:

    def test_returns_200(self, client):
        r = client.get("/health/")
        assert r.status_code == 200

    def test_status_is_ok(self, client):
        r = client.get("/health/")
        assert r.json()["status"] == "ok"

    def test_loaded_models_empty_by_default(self, client, mock_mm):
        mock_mm.loaded_model_names.return_value = []
        r = client.get("/health/")
        assert r.json()["loaded_models"] == []

    def test_loaded_models_reflects_cache(self, client, mock_mm):
        mock_mm.loaded_model_names.return_value = ["en_core_web_sm"]
        r = client.get("/health/")
        assert "en_core_web_sm" in r.json()["loaded_models"]

    def test_total_predictions_zero_initially(self, client):
        r = client.get("/health/")
        assert r.json()["total_predictions"] == 0

    def test_history_backend_in_memory(self, client):
        r = client.get("/health/")
        # The test fixture uses in-memory PredictionHistory
        assert r.json()["history_backend"] == "in-memory"

    def test_response_schema_complete(self, client):
        r = client.get("/health/")
        body = r.json()
        for key in ("status", "loaded_models", "total_predictions",
                    "history_backend", "redis_connected"):
            assert key in body