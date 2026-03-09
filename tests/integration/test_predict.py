"""
integration/test_predict.py
-----------------------------
POST /predict/ — happy path, empty text, model not available,
entity extraction, history persistence.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from app.schemas.requests import EntityResult


def _make_nlp(*ent_specs):
    """
    Build a fake nlp callable that returns a doc with the given entities.
    Each spec is a (label, text, start, end) tuple.
    """
    def _ent(label, text, start, end):
        e = MagicMock()
        e.label_ = label
        e.text = text
        e.start_char = start
        e.end_char = end
        return e

    doc = MagicMock()
    doc.ents = [_ent(*s) for s in ent_specs]
    return MagicMock(return_value=doc)


@pytest.mark.integration
class TestPredictEndpoint:

    def test_returns_200(self, client, mock_mm):
        mock_mm.get.return_value = _make_nlp()
        r = client.post("/predict/", json={"text": "Hello world", "model": "en_core_web_sm"})
        assert r.status_code == 200

    def test_response_contains_model_name(self, client, mock_mm):
        mock_mm.get.return_value = _make_nlp()
        r = client.post("/predict/", json={"text": "Hello", "model": "en_core_web_sm"})
        assert r.json()["model"] == "en_core_web_sm"

    def test_response_contains_text(self, client, mock_mm):
        mock_mm.get.return_value = _make_nlp()
        r = client.post("/predict/", json={"text": "Hello world", "model": "en_core_web_sm"})
        assert r.json()["text"] == "Hello world"

    def test_text_is_stripped_in_response(self, client, mock_mm):
        mock_mm.get.return_value = _make_nlp()
        r = client.post("/predict/", json={"text": "  Hello  ", "model": "en_core_web_sm"})
        assert r.json()["text"] == "Hello"

    def test_response_has_timestamp(self, client, mock_mm):
        mock_mm.get.return_value = _make_nlp()
        r = client.post("/predict/", json={"text": "Hello", "model": "en_core_web_sm"})
        assert "timestamp" in r.json()

    def test_response_schema_complete(self, client, mock_mm):
        mock_mm.get.return_value = _make_nlp()
        r = client.post("/predict/", json={"text": "Hello", "model": "en_core_web_sm"})
        body = r.json()
        for key in ("model", "model_version", "text", "entities", "timestamp"):
            assert key in body

    def test_single_entity_extracted(self, client, mock_mm):
        mock_mm.get.return_value = _make_nlp(("MONEY", "$45", 13, 16))
        r = client.post("/predict/", json={
            "text": "Can you send $45?",
            "model": "en_core_web_sm",
        })
        entities = r.json()["entities"]
        assert len(entities) == 1
        assert entities[0]["label"] == "MONEY"
        assert entities[0]["text"] == "$45"
        assert entities[0]["start"] == 13
        assert entities[0]["end"] == 16

    def test_multiple_entities_extracted(self, client, mock_mm):
        mock_mm.get.return_value = _make_nlp(
            ("MONEY",  "$45",     13, 16),
            ("PERSON", "Michael", 20, 27),
            ("DATE",   "June 3",  31, 37),
        )
        r = client.post("/predict/", json={
            "text": "Can you send $45 to Michael on June 3?",
            "model": "en_core_web_sm",
        })
        assert len(r.json()["entities"]) == 3

    def test_no_entities_returns_empty_list(self, client, mock_mm):
        mock_mm.get.return_value = _make_nlp()  # no ents
        r = client.post("/predict/", json={"text": "plain text", "model": "en_core_web_sm"})
        assert r.json()["entities"] == []

    def test_model_version_in_response(self, client, mock_mm, mock_reg):
        mock_mm.get.return_value = _make_nlp()
        mock_reg.get_model_info.return_value = {"version": "5", "stage": "Production"}
        r = client.post("/predict/", json={"text": "text", "model": "en_core_web_sm"})
        assert r.json()["model_version"] == "5"

    def test_prediction_is_stored_in_history(self, client, mock_mm, history):
        mock_mm.get.return_value = _make_nlp(("MONEY", "$45", 0, 3))
        client.post("/predict/", json={"text": "Send $45", "model": "en_core_web_sm"})
        assert history.count() == 1

    def test_history_record_has_correct_model(self, client, mock_mm, history):
        mock_mm.get.return_value = _make_nlp()
        client.post("/predict/", json={"text": "Hello", "model": "en_core_web_sm"})
        assert history.all()[0].model == "en_core_web_sm"

    def test_history_record_has_correct_text(self, client, mock_mm, history):
        mock_mm.get.return_value = _make_nlp()
        client.post("/predict/", json={"text": "Hello", "model": "en_core_web_sm"})
        assert history.all()[0].input_text == "Hello"

    def test_multiple_predictions_accumulated(self, client, mock_mm, history):
        mock_mm.get.return_value = _make_nlp()
        for text in ["first", "second", "third"]:
            client.post("/predict/", json={"text": text, "model": "en_core_web_sm"})
        assert history.count() == 3

    def test_empty_text_returns_400(self, client, mock_mm):
        mock_mm.get.side_effect = None  # not even called
        r = client.post("/predict/", json={"text": "", "model": "en_core_web_sm"})
        assert r.status_code == 400

    def test_whitespace_text_returns_400(self, client, mock_mm):
        r = client.post("/predict/", json={"text": "   ", "model": "en_core_web_sm"})
        assert r.status_code == 400

    def test_model_not_loaded_returns_400(self, client, mock_mm):
        mock_mm.get.side_effect = ValueError("not available. Call POST /load/ first.")
        r = client.post("/predict/", json={"text": "Hello", "model": "en_core_web_sm"})
        assert r.status_code == 400
        assert "load" in r.json()["detail"].lower()

    def test_deleted_model_returns_400(self, client, mock_mm):
        mock_mm.get.side_effect = ValueError("was deleted. Call POST /load/ to register it again.")
        r = client.post("/predict/", json={"text": "text", "model": "en_core_web_sm"})
        assert r.status_code == 400

    def test_missing_text_field_returns_422(self, client):
        r = client.post("/predict/", json={"model": "en_core_web_sm"})
        assert r.status_code == 422

    def test_missing_model_field_returns_422(self, client):
        r = client.post("/predict/", json={"text": "Hello"})
        assert r.status_code == 422