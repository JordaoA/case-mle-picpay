"""
Integration tests for the GET /list/ endpoint.
"""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.storage import get_history
from app.main import app
from tests.fakes.history_fake import FakePredictionHistory

fake_history = FakePredictionHistory()


@pytest.fixture(autouse=True)
def override_dependencies():
    """Injects Fake MongoDB into FastAPI."""
    app.dependency_overrides[get_history] = lambda: fake_history
    yield
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def clean_state():
    """Ensures a clean database before each test."""
    fake_history.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestListPredictionsEndpoint:
    @pytest.mark.parametrize("num_records", [1, 3, 5])
    def test_total_matches_predictions_length(
        self, client: TestClient, num_records: int
    ) -> None:
        """Directly seeds the database and verifies HTTP retrieval."""
        for i in range(num_records):
            fake_history.add(
                input_text=f"text {i}",
                model="en_core_web_sm",
                output=[],
                timestamp=datetime.now(timezone.utc),
            )

        response = client.get("/list/")
        body = response.json()

        assert response.status_code == 200
        assert body["total"] == num_records
        assert len(body["predictions"]) == num_records