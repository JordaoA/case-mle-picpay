"""
integration/conftest.py
------------------------
Patches every singleton before the FastAPI app is imported, then
exposes a TestClient and controllable mock objects as fixtures.

Patching strategy
-----------------
All singletons (model_manager, mlflow_registry, prediction_history) are
module-level objects instantiated at import time.  Because Python caches
modules, we patch the singletons IN the modules that CONSUME them (the
routers and ner_service), not in the modules that define them.  This
guarantees every route handler sees the mock regardless of import order.

The integration conftest is function-scoped: each test gets a fresh set
of mocks and a fresh in-memory history, so tests are fully isolated.
"""

import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import fakeredis
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True, scope="session")
def _block_real_services(session_mocker=None):
    """
    Inject safe env vars once for the entire integration session.
    Uses os.environ directly so they survive module-level singleton init.
    """
    import os
    overrides = {
        "MLFLOW_TRACKING_URI":    "http://fake-mlflow:5000",
        "MLFLOW_EXPERIMENT_NAME": "test-integration",
        "REDIS_HOST":             "fake-redis",
        "REDIS_PORT":             "6379",
        "REDIS_DB":               "0",
        "REDIS_PASSWORD":         "",
        "REDIS_TTL_SECONDS":      "3600",
        "SERVICE_NAME":           "test-ner",
        "SERVICE_VERSION":        "0.0.1",
    }
    original = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    yield
    for k, v in original.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _make_model_manager_mock(loaded: list[str] | None = None):
    mm = MagicMock()
    mm.loaded_model_names.return_value = loaded or []
    mm.list_models.return_value = []
    return mm


def _make_mlflow_mock():
    reg = MagicMock()
    reg.get_model_info.return_value = {
        "version": "1",
        "stage": "Production",
        "run_id": "run-test-001",
    }
    reg.register_model.return_value = {
        "name": "en_core_web_sm",
        "version": "1",
        "stage": "Production",
        "run_id": "run-test-001",
    }
    reg.list_registered_models.return_value = []
    reg.log_prediction.return_value = "run-test-001"
    return reg


def _make_history():
    """Real in-memory history — lets us assert actual records are stored."""
    from app.storage.history import PredictionHistory
    return PredictionHistory()


@pytest.fixture()
def mock_mm():
    return _make_model_manager_mock()


@pytest.fixture()
def mock_reg():
    return _make_mlflow_mock()


@pytest.fixture()
def history():
    return _make_history()


@pytest.fixture()
def client(mock_mm, mock_reg, history):
    """
    Returns a TestClient with all singletons replaced by mocks.
    The app is imported fresh inside the patch context so the mocks
    are in place before any route handler code runs.
    """
    with patch("app.routers.models.model_manager",          mock_mm), \
         patch("app.routers.models.mlflow_registry",        mock_reg), \
         patch("app.routers.predictions.run_prediction",    _make_run_prediction(mock_mm, mock_reg, history)), \
         patch("app.routers.predictions.prediction_history", history), \
         patch("app.main.model_manager",                    mock_mm), \
         patch("app.main.prediction_history",               history):

        from app.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


def _make_run_prediction(mm, reg, hist):
    """
    Returns a drop-in replacement for run_prediction() that uses
    real logic but pulls from mocked model_manager/mlflow_registry/history.
    """
    from datetime import datetime, timezone
    from app.schemas.requests import EntityResult, PredictResponse

    def _run(text: str, model_name: str) -> PredictResponse:
        text = text.strip()
        if not text:
            raise ValueError("Input text must not be empty.")

        nlp = mm.get(model_name)           # raises ValueError if not loaded
        info = reg.get_model_info(model_name)
        version = info["version"] if info else "unknown"

        doc = nlp(text)
        entities = [
            EntityResult(
                label=e.label_,
                text=e.text,
                start=e.start_char,
                end=e.end_char,
            )
            for e in doc.ents
        ]
        ts = datetime.now(tz=timezone.utc)
        reg.log_prediction(
            model_name=model_name,
            model_version=version,
            input_text=text,
            entities=[{"label": e.label, "text": e.text} for e in entities],
            latency_ms=0.0,
        )
        hist.add(input_text=text, output=entities, model=model_name, timestamp=ts)
        return PredictResponse(
            model=model_name,
            model_version=version,
            text=text,
            entities=entities,
            timestamp=ts,
        )

    return _run