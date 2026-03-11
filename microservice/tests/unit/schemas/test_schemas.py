"""
unit/schemas/test_schemas.py
----------------------------
Tests for Pydantic schema validation.
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.requests import (
    EntityResult,
    HealthResponse,
    PredictionRecord,
    PredictRequest,
    PredictResponse,
)


class TestPredictRequest:
    def test_valid(self) -> None:
        req = PredictRequest(text="Hello", model="en_core_web_sm")
        assert req.text == "Hello"
        assert req.model == "en_core_web_sm"

    def test_text_required(self) -> None:
        with pytest.raises(ValidationError):
            PredictRequest(model="en_core_web_sm")


class TestEntityResult:
    def test_valid(self) -> None:
        ent = EntityResult(label="ORG", text="Apple", start=0, end=5)
        assert ent.label == "ORG"
        assert ent.text == "Apple"
        assert ent.start == 0
        assert ent.end == 5


class TestPredictResponse:
    def test_valid(self) -> None:
        ent = EntityResult(label="ORG", text="Apple", start=0, end=5)
        resp = PredictResponse(entities=[ent], record_id="65a2b1c3d4e5f6g7h8i9j0k0")
        assert len(resp.entities) == 1
        assert resp.entities[0].text == "Apple"
        assert resp.record_id == "65a2b1c3d4e5f6g7h8i9j0k0"

    def test_empty_entities_allowed(self) -> None:
        resp = PredictResponse(entities=[], record_id="12345")
        assert resp.entities == []
        assert resp.record_id == "12345"
        
    def test_missing_record_id_fails(self) -> None:
        with pytest.raises(ValidationError):
            PredictResponse(entities=[]) # type: ignore


class TestHealthResponse:
    def test_valid(self) -> None:
        resp = HealthResponse(
            status="ok",
            loaded_models=["en_core_web_sm"],
            total_predictions=10,
            history_backend="mongodb",
            mongodb_connected=True,
        )
        assert resp.status == "ok"
        assert resp.mongodb_connected is True
        assert resp.history_backend == "mongodb"

    def test_in_memory_backend(self) -> None:
        resp = HealthResponse(
            status="ok",
            loaded_models=[],
            total_predictions=0,
            history_backend="in-memory",
            mongodb_connected=False,
        )
        assert resp.history_backend == "in-memory"
        assert resp.mongodb_connected is False


class TestPredictionRecord:
    def test_valid_string_id(self) -> None:
        rec = PredictionRecord(
            id="mongo-id-123",
            input_text="Test",
            output=[],
            model="en_core_web_sm",
            timestamp=datetime.now(timezone.utc),
        )
        assert rec.id == "mongo-id-123"
        assert rec.input_text == "Test"