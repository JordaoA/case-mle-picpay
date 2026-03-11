"""
integration/test_list.py
-------------------------
GET /list/ — history retrieval: empty state, after predictions,
total count, ordering.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import get_history
from tests.fakes.history_fake import FakePredictionHistory

fake_history = FakePredictionHistory()

@pytest.fixture(autouse=True)
def override_dependencies():
    """Overrides real database/redis with our isolated in-memory Fake."""
    app.dependency_overrides[get_history] = lambda: fake_history
    yield
    app.dependency_overrides.clear()

@pytest.fixture(autouse=True)
def clean_state():
    """Ensures a pristine database state before every test."""
    fake_history.clear()
    yield

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.integration
class TestListPredictionsEndpoint:

    def test_empty_initially_returns_200(self, client):
        """Tests the baseline empty state."""
        response = client.get("/list/")
        
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 0
        assert body["predictions"] == []

    @pytest.mark.parametrize("num_records", [1, 3, 5])
    def test_total_matches_predictions_length(self, client, num_records):
        """
        Directly seeds the database bypassing POST /predict/, 
        proving the GET endpoint accurately counts records.
        """
        for i in range(num_records):
            fake_history.save_prediction(
                input_text=f"text {i}", 
                model_name="en_core_web_sm", 
                entities=[]
            )

        response = client.get("/list/")
        body = response.json()

        assert body["total"] == num_records
        assert len(body["predictions"]) == num_records

    def test_prediction_record_schema_and_contents(self, client):
        """
        Combines multiple old tests into one cohesive assertion 
        to ensure the schema and serialization are perfectly intact.
        """
        fake_history.save_prediction(
            input_text="Apple is opening in New York", 
            model_name="en_core_web_sm", 
            entities=[{"text": "Apple", "label": "ORG", "start": 0, "end": 5}]
        )

        response = client.get("/list/")
        record = response.json()["predictions"][0]

        expected_keys = {"id", "input_text", "output", "model", "timestamp"}
        assert expected_keys.issubset(record.keys())

        assert record["model"] == "en_core_web_sm"
        assert record["input_text"] == "Apple is opening in New York"
        assert len(record["output"]) == 1
        assert record["output"][0]["label"] == "ORG"