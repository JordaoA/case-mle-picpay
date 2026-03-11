"""
unit/conftest.py
-----------------
Per-test env isolation and shared mock fixtures.

The root conftest.py sets all env vars at module level so collection
succeeds. This conftest re-applies them via monkeypatch so that changes
made inside one test (e.g. monkeypatch.setenv in test_config.py) are
automatically rolled back before the next test runs.
"""

from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_HOST",             "fake-mlflow")
    monkeypatch.setenv("MLFLOW_PORT",             "5000")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME",  "test-experiment")
    monkeypatch.setenv("REDIS_HOST",              "fake-redis")
    monkeypatch.setenv("REDIS_PORT",              "6379")
    monkeypatch.setenv("REDIS_DB",                "0")
    monkeypatch.setenv("REDIS_PASSWORD",          "")
    monkeypatch.setenv("REDIS_TTL_SECONDS",       "3600")
    monkeypatch.setenv("SERVICE_NAME",            "test-ner-service")
    monkeypatch.setenv("SERVICE_VERSION",         "0.0.1")


@pytest.fixture()
def mock_entity():
    e = MagicMock()
    e.label_     = "MONEY"
    e.text       = "$45"
    e.start_char = 13
    e.end_char   = 16
    return e


@pytest.fixture()
def mock_doc(mock_entity):
    doc = MagicMock()
    doc.ents = [mock_entity]
    return doc


@pytest.fixture()
def mock_nlp(mock_doc):
    return MagicMock(return_value=mock_doc)