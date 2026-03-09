"""
integration/test_list.py
-------------------------
GET /list/ — history retrieval: empty state, after predictions,
total count, ordering.
"""

from unittest.mock import MagicMock

import pytest


def _make_nlp():
    doc = MagicMock()
    doc.ents = []
    return MagicMock(return_value=doc)


@pytest.mark.integration
class TestListPredictionsEndpoint:

    def test_returns_200(self, client):
        r = client.get("/list/")
        assert r.status_code == 200

    def test_empty_initially(self, client):
        r = client.get("/list/")
        body = r.json()
        assert body["total"] == 0
        assert body["predictions"] == []

    def test_total_after_prediction(self, client, mock_mm, history):
        mock_mm.get.return_value = _make_nlp()
        client.post("/predict/", json={"text": "Hello", "model": "en_core_web_sm"})
        r = client.get("/list/")
        assert r.json()["total"] == 1

    def test_predictions_list_after_prediction(self, client, mock_mm):
        mock_mm.get.return_value = _make_nlp()
        client.post("/predict/", json={"text": "Hello", "model": "en_core_web_sm"})
        r = client.get("/list/")
        assert len(r.json()["predictions"]) == 1

    def test_prediction_record_has_correct_fields(self, client, mock_mm):
        mock_mm.get.return_value = _make_nlp()
        client.post("/predict/", json={"text": "Hello", "model": "en_core_web_sm"})
        record = client.get("/list/").json()["predictions"][0]
        for key in ("id", "input_text", "output", "model", "timestamp"):
            assert key in record

    def test_prediction_record_correct_model(self, client, mock_mm):
        mock_mm.get.return_value = _make_nlp()
        client.post("/predict/", json={"text": "Hello", "model": "en_core_web_sm"})
        record = client.get("/list/").json()["predictions"][0]
        assert record["model"] == "en_core_web_sm"

    def test_prediction_record_correct_text(self, client, mock_mm):
        mock_mm.get.return_value = _make_nlp()
        client.post("/predict/", json={"text": "Hello world", "model": "en_core_web_sm"})
        record = client.get("/list/").json()["predictions"][0]
        assert record["input_text"] == "Hello world"

    def test_total_matches_predictions_length(self, client, mock_mm):
        mock_mm.get.return_value = _make_nlp()
        for i in range(4):
            client.post("/predict/", json={"text": f"text {i}", "model": "en_core_web_sm"})
        body = client.get("/list/").json()
        assert body["total"] == len(body["predictions"])

    def test_all_predictions_accumulate(self, client, mock_mm):
        mock_mm.get.return_value = _make_nlp()
        for text in ["alpha", "beta", "gamma"]:
            client.post("/predict/", json={"text": text, "model": "en_core_web_sm"})
        body = client.get("/list/").json()
        assert body["total"] == 3

    def test_response_schema_complete(self, client):
        r = client.get("/list/")
        body = r.json()
        assert "total" in body
        assert "predictions" in body