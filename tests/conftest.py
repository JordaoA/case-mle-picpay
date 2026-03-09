"""
conftest.py  (root)
-------------------
Two things must happen at MODULE LEVEL (before pytest collects any test):

1. Environment variables must be set — so Settings() doesn't crash when
   it calls int(os.getenv("REDIS_DB")) at class definition time.

2. MLflow network calls must be patched — mlflow_registry.py previously
   had `mlflow_registry = MLflowRegistry()` at module level. Even with the
   lazy singleton fix, any test module that triggers an import chain reaching
   MLflowRegistry.__init__ before a fixture runs will attempt a real HTTP
   connection to 'fake-mlflow:5000' and fail with a DNS error.

   We start long-lived patches here at conftest import time and stop them
   at session end via a session-scoped autouse fixture. This guarantees
   MLflow is never contacted during unit tests regardless of import order.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "microservice_v3"))

_TEST_ENV = {
    "MLFLOW_HOST":             "fake-mlflow",
    "MLFLOW_PORT":             "5000",
    "MLFLOW_EXPERIMENT_NAME":  "test-experiment",
    "REDIS_HOST":              "fake-redis",
    "REDIS_PORT":              "6379",
    "REDIS_DB":                "0",
    "REDIS_PASSWORD":          "",
    "REDIS_TTL_SECONDS":       "3600",
    "SERVICE_NAME":            "test-ner-service",
    "SERVICE_VERSION":         "0.0.1",
}
for _key, _val in _TEST_ENV.items():
    os.environ.setdefault(_key, _val)

_mock_mlflow_client = MagicMock()
_mock_mlflow_client.get_experiment_by_name.return_value = None
_mock_mlflow_client.create_experiment.return_value = "exp-test-001"

_patches = [
    patch("mlflow.set_tracking_uri"),
    patch("mlflow.get_experiment_by_name", return_value=None),
    patch("mlflow.create_experiment",      return_value="exp-test-001"),
    patch("mlflow.MlflowClient",           return_value=_mock_mlflow_client),
]

for _p in _patches:
    _p.start()

import pytest

@pytest.fixture(autouse=True, scope="session")
def _stop_global_patches():
    yield
    for _p in _patches:
        try:
            _p.stop()
        except RuntimeError:
            pass