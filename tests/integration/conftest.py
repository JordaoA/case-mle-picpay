"""
Pytest configuration and shared fixtures for integration tests.

This module provides fixtures and setup specific to API integration tests,
building on the base fixtures defined in tests/conftest.py.
"""

import os
import sys
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone

# Ensure the microservice module is importable
microservice_path = os.path.join(
    os.path.dirname(__file__), 
    "..", "..", 
    "microservice"
)
if microservice_path not in sys.path:
    sys.path.insert(0, microservice_path)


@pytest.fixture
def sample_predict_response():
    """Sample response from run_prediction service."""
    from app.schemas.requests import EntityResult
    
    return {
        "model": "en_core_web_sm",
        "model_version": "3.5.0",
        "text": "Can you send $45 to Michael on June 3?",
        "entities": [
            EntityResult(
                label="MONEY",
                text="$45",
                start=15,
                end=18,
            ),
            EntityResult(
                label="PERSON",
                text="Michael",
                start=32,
                end=39,
            ),
            EntityResult(
                label="DATE",
                text="June 3",
                start=43,
                end=49,
            ),
        ],
        "timestamp": datetime.now(tz=timezone.utc),
    }


@pytest.fixture
def sample_model_info():
    """Sample model info from MLflow registry."""
    return {
        "name": "en_core_web_sm",
        "version": "1.0.0",
        "stage": "Production",
        "loaded": True,
        "run_id": "abc123def456",
    }


@pytest.fixture
def api_response_headers():
    """Standard headers expected in API responses."""
    return {
        "content-type": "application/json",
    }
