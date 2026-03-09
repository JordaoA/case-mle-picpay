"""
unit/schemas/test_schemas.py
------------------------------
Pydantic model validation — required fields, defaults, and type coercions.
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.requests import (
    EntityResult,
    ErrorResponse,
    HealthResponse,
    ListModelsResponse,
    ListPredictionsResponse,
    LoadModelRequest,
    LoadModelResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
    PredictionRecord,
)


@pytest.mark.unit
class TestLoadModelRequest:

    def test_valid(self):
        r = LoadModelRequest(model="en_core_web_sm")
        assert r.model == "en_core_web_sm"

    def test_missing_model_raises(self):
        with pytest.raises(ValidationError):
            LoadModelRequest()


@pytest.mark.unit
class TestLoadModelResponse:

    def test_required_fields(self):
        r = LoadModelResponse(model="en_core_web_sm", status="downloaded", message="ok")
        assert r.model == "en_core_web_sm"
        assert r.status == "downloaded"

    def test_optional_fields_default_to_none(self):
        r = LoadModelResponse(model="en_core_web_sm", status="downloaded", message="ok")
        assert r.mlflow_version is None
        assert r.mlflow_stage is None
        assert r.mlflow_run_id is None

    def test_optional_fields_populated(self):
        r = LoadModelResponse(
            model="en_core_web_sm",
            status="downloaded",
            message="ok",
            mlflow_version="3",
            mlflow_stage="Production",
            mlflow_run_id="abc123",
        )
        assert r.mlflow_version == "3"
        assert r.mlflow_stage == "Production"
        assert r.mlflow_run_id == "abc123"


@pytest.mark.unit
class TestPredictRequest:

    def test_valid(self):
        r = PredictRequest(text="Hello world", model="en_core_web_sm")
        assert r.text == "Hello world"
        assert r.model == "en_core_web_sm"

    def test_missing_text_raises(self):
        with pytest.raises(ValidationError):
            PredictRequest(model="en_core_web_sm")

    def test_missing_model_raises(self):
        with pytest.raises(ValidationError):
            PredictRequest(text="Hello world")

    def test_both_missing_raises(self):
        with pytest.raises(ValidationError):
            PredictRequest()


@pytest.mark.unit
class TestEntityResult:

    def test_valid(self):
        e = EntityResult(label="MONEY", text="$45", start=13, end=16)
        assert e.label == "MONEY"
        assert e.text == "$45"
        assert e.start == 13
        assert e.end == 16

    def test_missing_label_raises(self):
        with pytest.raises(ValidationError):
            EntityResult(text="$45", start=0, end=3)

    def test_missing_text_raises(self):
        with pytest.raises(ValidationError):
            EntityResult(label="MONEY", start=0, end=3)

    def test_missing_start_raises(self):
        with pytest.raises(ValidationError):
            EntityResult(label="MONEY", text="$45", end=3)

    def test_missing_end_raises(self):
        with pytest.raises(ValidationError):
            EntityResult(label="MONEY", text="$45", start=0)


@pytest.mark.unit
class TestPredictResponse:

    def test_model_version_default(self):
        r = PredictResponse(
            model="en_core_web_sm",
            text="Hello",
            entities=[],
            timestamp=datetime.now(tz=timezone.utc),
        )
        assert r.model_version == "unknown"

    def test_model_version_explicit(self):
        r = PredictResponse(
            model="en_core_web_sm",
            model_version="5",
            text="Hello",
            entities=[],
            timestamp=datetime.now(tz=timezone.utc),
        )
        assert r.model_version == "5"

    def test_entities_attached(self):
        entity = EntityResult(label="PERSON", text="Alice", start=0, end=5)
        r = PredictResponse(
            model="en_core_web_sm",
            text="Alice",
            entities=[entity],
            timestamp=datetime.now(tz=timezone.utc),
        )
        assert len(r.entities) == 1
        assert r.entities[0].label == "PERSON"

    def test_empty_entities_allowed(self):
        r = PredictResponse(
            model="en_core_web_sm",
            text="blah",
            entities=[],
            timestamp=datetime.now(tz=timezone.utc),
        )
        assert r.entities == []


@pytest.mark.unit
class TestPredictionRecord:

    def test_valid(self):
        r = PredictionRecord(
            id=7,
            input_text="Send $45",
            output=[EntityResult(label="MONEY", text="$45", start=5, end=8)],
            model="en_core_web_sm",
            timestamp=datetime.now(tz=timezone.utc),
        )
        assert r.id == 7
        assert r.model == "en_core_web_sm"
        assert len(r.output) == 1

    def test_empty_output_allowed(self):
        r = PredictionRecord(
            id=1,
            input_text="plain text",
            output=[],
            model="en_core_web_sm",
            timestamp=datetime.now(tz=timezone.utc),
        )
        assert r.output == []


@pytest.mark.unit
class TestListPredictionsResponse:

    def test_empty(self):
        r = ListPredictionsResponse(total=0, predictions=[])
        assert r.total == 0
        assert r.predictions == []

    def test_total_independent_of_list(self):
        # total is explicitly set — it can differ from len(predictions)
        r = ListPredictionsResponse(total=100, predictions=[])
        assert r.total == 100


@pytest.mark.unit
class TestModelInfo:

    def test_defaults(self):
        m = ModelInfo(name="en_core_web_sm")
        assert m.version == "—"
        assert m.stage == "—"
        assert m.loaded is False
        assert m.run_id is None

    def test_explicit(self):
        m = ModelInfo(
            name="en_core_web_sm",
            version="2",
            stage="Production",
            loaded=True,
            run_id="run-xyz",
        )
        assert m.loaded is True
        assert m.stage == "Production"
        assert m.run_id == "run-xyz"


@pytest.mark.unit
class TestListModelsResponse:

    def test_empty(self):
        r = ListModelsResponse(models=[])
        assert r.models == []

    def test_with_entries(self):
        r = ListModelsResponse(
            models=[ModelInfo(name="en_core_web_sm"), ModelInfo(name="en_core_web_md")]
        )
        assert len(r.models) == 2


@pytest.mark.unit
class TestHealthResponse:

    def test_valid(self):
        r = HealthResponse(
            status="ok",
            loaded_models=["en_core_web_sm"],
            total_predictions=42,
            history_backend="redis",
            redis_connected=True,
        )
        assert r.status == "ok"
        assert r.total_predictions == 42
        assert r.redis_connected is True

    def test_in_memory_backend(self):
        r = HealthResponse(
            status="ok",
            loaded_models=[],
            total_predictions=0,
            history_backend="in-memory",
            redis_connected=False,
        )
        assert r.history_backend == "in-memory"
        assert r.redis_connected is False


@pytest.mark.unit
class TestErrorResponse:

    def test_required_error_field(self):
        r = ErrorResponse(error="something went wrong")
        assert r.error == "something went wrong"
        assert r.detail is None

    def test_with_detail(self):
        r = ErrorResponse(error="not found", detail="Model not in registry")
        assert r.detail == "Model not in registry"