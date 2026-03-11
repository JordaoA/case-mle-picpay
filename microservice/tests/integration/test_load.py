"""
integration/test_load.py
-------------------------
POST /load/ — happy path, validation errors, download failures.
"""

import pytest


@pytest.mark.integration
class TestLoadEndpoint:

    def test_returns_200_on_success(self, client, mock_mm):
        mock_mm.ensure_available.return_value = "downloaded"
        r = client.post("/load/", json={"model": "en_core_web_sm"})
        assert r.status_code == 200

    def test_response_contains_model_name(self, client, mock_mm):
        mock_mm.ensure_available.return_value = "downloaded"
        r = client.post("/load/", json={"model": "en_core_web_sm"})
        assert r.json()["model"] == "en_core_web_sm"

    def test_response_status_downloaded(self, client, mock_mm):
        mock_mm.ensure_available.return_value = "downloaded"
        r = client.post("/load/", json={"model": "en_core_web_sm"})
        assert r.json()["status"] == "downloaded"

    def test_response_status_already_available(self, client, mock_mm):
        mock_mm.ensure_available.return_value = "already_available"
        r = client.post("/load/", json={"model": "en_core_web_sm"})
        assert r.json()["status"] == "already_available"

    def test_response_contains_mlflow_fields(self, client, mock_mm, mock_reg):
        mock_mm.ensure_available.return_value = "downloaded"
        mock_reg.get_model_info.return_value = {
            "version": "3",
            "stage": "Production",
            "run_id": "run-xyz",
        }
        r = client.post("/load/", json={"model": "en_core_web_sm"})
        body = r.json()
        assert body["mlflow_version"] == "3"
        assert body["mlflow_stage"] == "Production"
        assert body["mlflow_run_id"] == "run-xyz"

    def test_mlflow_fields_none_when_registry_empty(self, client, mock_mm, mock_reg):
        mock_mm.ensure_available.return_value = "downloaded"
        mock_reg.get_model_info.return_value = None
        r = client.post("/load/", json={"model": "en_core_web_sm"})
        body = r.json()
        assert body["mlflow_version"] is None
        assert body["mlflow_stage"] is None
        assert body["mlflow_run_id"] is None

    def test_ensure_available_called_with_model_name(self, client, mock_mm):
        mock_mm.ensure_available.return_value = "downloaded"
        client.post("/load/", json={"model": "en_core_web_sm"})
        mock_mm.ensure_available.assert_called_once_with("en_core_web_sm")

    def test_unsupported_model_returns_400(self, client, mock_mm):
        mock_mm.ensure_available.side_effect = ValueError("not supported")
        r = client.post("/load/", json={"model": "some_bad_model"})
        assert r.status_code == 400

    def test_400_detail_contains_reason(self, client, mock_mm):
        mock_mm.ensure_available.side_effect = ValueError("not supported. Allowed models")
        r = client.post("/load/", json={"model": "bad_model"})
        assert "not supported" in r.json()["detail"]

    def test_missing_model_field_returns_422(self, client):
        r = client.post("/load/", json={})
        assert r.status_code == 422

    def test_empty_body_returns_422(self, client):
        r = client.post("/load/", content=b"")
        assert r.status_code == 422

    def test_download_failure_returns_500(self, client, mock_mm):
        mock_mm.ensure_available.side_effect = RuntimeError("pip failed")
        r = client.post("/load/", json={"model": "en_core_web_sm"})
        assert r.status_code == 500

    def test_500_detail_contains_reason(self, client, mock_mm):
        mock_mm.ensure_available.side_effect = RuntimeError("pip subprocess returned 1")
        r = client.post("/load/", json={"model": "en_core_web_sm"})
        assert "pip" in r.json()["detail"]