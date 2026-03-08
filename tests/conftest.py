"""Pytest configuration and shared fixtures for all tests."""
import os
from unittest.mock import Mock, MagicMock, patch
import pytest
from datetime import datetime, timezone

# Add microservice directory to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'microservice'))


@pytest.fixture
def mock_settings():
    """Mock Settings object with sensible defaults."""
    settings = Mock()
    settings.mlflow_host = "localhost"
    settings.mlflow_port = "5000"
    settings.mlflow_experiment_name = "test-experiment"
    settings.service_name = "test-service"
    settings.service_version = "1.0.0"
    settings.redis_host = "localhost"
    settings.redis_port = 6379
    settings.redis_db = 0
    settings.redis_password = ""
    settings.redis_ttl_seconds = 86400
    settings.mlflow_tracking_uri = "http://localhost:5000"
    return settings


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing."""
    client = Mock()
    client.ping.return_value = True
    client.incr.return_value = 1
    client.lpush.return_value = 1
    client.lrange.return_value = []
    client.get.return_value = None
    client.hgetall.return_value = {}
    client.pipeline.return_value = client
    return client


@pytest.fixture
def sample_entity_result():
    """Sample EntityResult for testing."""
    from app.schemas.requests import EntityResult
    return EntityResult(
        label="PERSON",
        text="John",
        start=0,
        end=4,
    )


@pytest.fixture
def sample_prediction_record(sample_entity_result):
    """Sample PredictionRecord for testing."""
    from app.schemas.requests import PredictionRecord
    return PredictionRecord(
        id=1,
        input_text="John loves pizza",
        output=[sample_entity_result],
        model="en_core_web_sm",
        timestamp=datetime.now(tz=timezone.utc),
    )


@pytest.fixture
def mock_spacy_nlp():
    """Mock spaCy NLP model."""
    nlp = Mock()
    doc = Mock()
    ent = Mock()
    ent.label_ = "PERSON"
    ent.text = "John"
    ent.start_char = 0
    ent.end_char = 4
    doc.ents = [ent]
    nlp.return_value = doc
    nlp.__call__ = Mock(return_value=doc)
    return nlp


@pytest.fixture
def env_vars(monkeypatch):
    """Set up test environment variables."""
    test_env = {
        "MLFLOW_HOST": "localhost",
        "MLFLOW_PORT": "5000",
        "MLFLOW_EXPERIMENT_NAME": "test-experiment",
        "SERVICE_NAME": "test-service",
        "SERVICE_VERSION": "1.0.0",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "REDIS_DB": "0",
        "REDIS_PASSWORD": "",
    }
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    return test_env
