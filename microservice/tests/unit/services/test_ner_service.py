"""
Unit tests for the NER inference pure function.
Requires no patching because all dependencies are injected.
"""

import pytest

from app.services.ner_service import run_prediction
from tests.fakes.history_fake import FakePredictionHistory
from tests.fakes.mlflow_fake import FakeMLflowRegistry
from tests.fakes.spacy_fake import FakeModelManager


def test_run_prediction_success() -> None:
    """Ensures inference successfully coordinates all injected dependencies."""
    manager = FakeModelManager()
    history = FakePredictionHistory()
    registry = FakeMLflowRegistry()
    registry.register_model("en_core_web_sm")

    result = run_prediction(
        text="Apple is in New York",
        model_name="en_core_web_sm",
        model_manager=manager,
        history_repo=history,
        model_registry=registry,
    )

    assert len(result["entities"]) == 2
    assert "record_id" in result
    assert history.count() == 1
    assert len(registry._logs) == 1
    assert registry._logs[0]["model"] == "en_core_web_sm"


def test_run_prediction_empty_text() -> None:
    """Ensures empty text is rejected before loading models."""
    with pytest.raises(ValueError, match="Input text must not be empty"):
        run_prediction(
            text="   ",
            model_name="en_core_web_sm",
            model_manager=FakeModelManager(),
            history_repo=FakePredictionHistory(),
            model_registry=FakeMLflowRegistry(),
        )