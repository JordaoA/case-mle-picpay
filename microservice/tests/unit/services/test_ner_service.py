"""
unit/services/test_ner_service.py
-----------------------------------
run_prediction() — output, validation, side-effect calls.
model_manager, mlflow_registry, and prediction_history are all patched.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.schemas.requests import EntityResult, PredictResponse
from app.services.ner_service import run_prediction


def _make_ent(label, text, start, end):
    e = MagicMock()
    e.label_ = label
    e.text = text
    e.start_char = start
    e.end_char = end
    return e


def _make_doc(*ents):
    doc = MagicMock()
    doc.ents = list(ents)
    return doc


def _patches(nlp, model_info=None, history=None):
    """Context manager stack: patches the three singletons."""
    import contextlib
    if model_info is None:
        model_info = {"version": "1", "stage": "Production"}
    if history is None:
        history = MagicMock()
        history.add.return_value = MagicMock()

    return contextlib.ExitStack()


@pytest.mark.unit
class TestRunPredictionOutput:

    def test_returns_predict_response(self):
        nlp = MagicMock(return_value=_make_doc())

        with patch("app.services.ner_service.model_manager") as mm, \
             patch("app.services.ner_service.mlflow_registry") as reg, \
             patch("app.services.ner_service.prediction_history"):
            mm.get.return_value = nlp
            reg.get_model_info.return_value = {"version": "1"}
            reg.log_prediction.return_value = "run-1"

            result = run_prediction("Hello world", "en_core_web_sm")

        assert isinstance(result, PredictResponse)

    def test_text_is_stripped(self):
        nlp = MagicMock(return_value=_make_doc())

        with patch("app.services.ner_service.model_manager") as mm, \
             patch("app.services.ner_service.mlflow_registry") as reg, \
             patch("app.services.ner_service.prediction_history"):
            mm.get.return_value = nlp
            reg.get_model_info.return_value = {"version": "1"}
            reg.log_prediction.return_value = ""

            result = run_prediction("  Hello world  ", "en_core_web_sm")

        assert result.text == "Hello world"

    def test_model_name_in_response(self):
        nlp = MagicMock(return_value=_make_doc())

        with patch("app.services.ner_service.model_manager") as mm, \
             patch("app.services.ner_service.mlflow_registry") as reg, \
             patch("app.services.ner_service.prediction_history"):
            mm.get.return_value = nlp
            reg.get_model_info.return_value = {"version": "2"}
            reg.log_prediction.return_value = ""

            result = run_prediction("text", "en_core_web_sm")

        assert result.model == "en_core_web_sm"

    def test_model_version_from_mlflow(self):
        nlp = MagicMock(return_value=_make_doc())

        with patch("app.services.ner_service.model_manager") as mm, \
             patch("app.services.ner_service.mlflow_registry") as reg, \
             patch("app.services.ner_service.prediction_history"):
            mm.get.return_value = nlp
            reg.get_model_info.return_value = {"version": "7"}
            reg.log_prediction.return_value = ""

            result = run_prediction("text", "en_core_web_sm")

        assert result.model_version == "7"

    def test_model_version_unknown_when_no_registry_info(self):
        nlp = MagicMock(return_value=_make_doc())

        with patch("app.services.ner_service.model_manager") as mm, \
             patch("app.services.ner_service.mlflow_registry") as reg, \
             patch("app.services.ner_service.prediction_history"):
            mm.get.return_value = nlp
            reg.get_model_info.return_value = None
            reg.log_prediction.return_value = ""

            result = run_prediction("text", "en_core_web_sm")

        assert result.model_version == "unknown"

    def test_entities_extracted(self):
        ent = _make_ent("MONEY", "$45", 13, 16)
        nlp = MagicMock(return_value=_make_doc(ent))

        with patch("app.services.ner_service.model_manager") as mm, \
             patch("app.services.ner_service.mlflow_registry") as reg, \
             patch("app.services.ner_service.prediction_history"):
            mm.get.return_value = nlp
            reg.get_model_info.return_value = {"version": "1"}
            reg.log_prediction.return_value = ""

            result = run_prediction("Send $45", "en_core_web_sm")

        assert len(result.entities) == 1
        e = result.entities[0]
        assert e.label == "MONEY"
        assert e.text == "$45"
        assert e.start == 13
        assert e.end == 16

    def test_multiple_entities(self):
        ents = [
            _make_ent("MONEY",  "$45",     13, 16),
            _make_ent("PERSON", "Michael", 20, 27),
            _make_ent("DATE",   "June 3",  31, 37),
        ]
        nlp = MagicMock(return_value=_make_doc(*ents))

        with patch("app.services.ner_service.model_manager") as mm, \
             patch("app.services.ner_service.mlflow_registry") as reg, \
             patch("app.services.ner_service.prediction_history"):
            mm.get.return_value = nlp
            reg.get_model_info.return_value = {"version": "1"}
            reg.log_prediction.return_value = ""

            result = run_prediction("Can you send $45 to Michael on June 3?", "en_core_web_sm")

        assert len(result.entities) == 3

    def test_empty_entities_when_none_found(self):
        nlp = MagicMock(return_value=_make_doc())  # no ents

        with patch("app.services.ner_service.model_manager") as mm, \
             patch("app.services.ner_service.mlflow_registry") as reg, \
             patch("app.services.ner_service.prediction_history"):
            mm.get.return_value = nlp
            reg.get_model_info.return_value = {"version": "1"}
            reg.log_prediction.return_value = ""

            result = run_prediction("no entities here", "en_core_web_sm")

        assert result.entities == []

    def test_timestamp_is_utc(self):
        nlp = MagicMock(return_value=_make_doc())

        with patch("app.services.ner_service.model_manager") as mm, \
             patch("app.services.ner_service.mlflow_registry") as reg, \
             patch("app.services.ner_service.prediction_history"):
            mm.get.return_value = nlp
            reg.get_model_info.return_value = {"version": "1"}
            reg.log_prediction.return_value = ""

            result = run_prediction("text", "en_core_web_sm")

        assert result.timestamp.tzinfo is not None


@pytest.mark.unit
class TestRunPredictionValidation:

    def test_empty_text_raises(self):
        with pytest.raises(ValueError, match="empty"):
            run_prediction("", "en_core_web_sm")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="empty"):
            run_prediction("   ", "en_core_web_sm")

    def test_model_not_available_propagates(self):
        with patch("app.services.ner_service.model_manager") as mm:
            mm.get.side_effect = ValueError("Model 'en_core_web_sm' is not available.")

            with pytest.raises(ValueError, match="not available"):
                run_prediction("some text", "en_core_web_sm")


@pytest.mark.unit
class TestRunPredictionSideEffects:

    def test_model_manager_get_called(self):
        nlp = MagicMock(return_value=_make_doc())

        with patch("app.services.ner_service.model_manager") as mm, \
             patch("app.services.ner_service.mlflow_registry") as reg, \
             patch("app.services.ner_service.prediction_history"):
            mm.get.return_value = nlp
            reg.get_model_info.return_value = {"version": "1"}
            reg.log_prediction.return_value = ""

            run_prediction("text", "en_core_web_sm")

        mm.get.assert_called_once_with("en_core_web_sm")

    def test_mlflow_log_prediction_called(self):
        ent = _make_ent("MONEY", "$45", 0, 3)
        nlp = MagicMock(return_value=_make_doc(ent))

        with patch("app.services.ner_service.model_manager") as mm, \
             patch("app.services.ner_service.mlflow_registry") as reg, \
             patch("app.services.ner_service.prediction_history"):
            mm.get.return_value = nlp
            reg.get_model_info.return_value = {"version": "1"}
            reg.log_prediction.return_value = ""

            run_prediction("text", "en_core_web_sm")

        reg.log_prediction.assert_called_once()
        kwargs = reg.log_prediction.call_args.kwargs
        assert kwargs["model_name"] == "en_core_web_sm"
        assert kwargs["model_version"] == "1"
        assert len(kwargs["entities"]) == 1

    def test_prediction_history_add_called(self):
        nlp = MagicMock(return_value=_make_doc())

        with patch("app.services.ner_service.model_manager") as mm, \
             patch("app.services.ner_service.mlflow_registry") as reg, \
             patch("app.services.ner_service.prediction_history") as hist:
            mm.get.return_value = nlp
            reg.get_model_info.return_value = {"version": "1"}
            reg.log_prediction.return_value = ""
            hist.add.return_value = MagicMock()

            run_prediction("text", "en_core_web_sm")

        hist.add.assert_called_once()
        kwargs = hist.add.call_args.kwargs
        assert kwargs["model"] == "en_core_web_sm"
        assert kwargs["input_text"] == "text"

    def test_mlflow_failure_does_not_break_response(self):
        """log_prediction() failures must never propagate — inference must succeed."""
        nlp = MagicMock(return_value=_make_doc())

        with patch("app.services.ner_service.model_manager") as mm, \
             patch("app.services.ner_service.mlflow_registry") as reg, \
             patch("app.services.ner_service.prediction_history") as hist:
            mm.get.return_value = nlp
            reg.get_model_info.return_value = {"version": "1"}
            reg.log_prediction.return_value = ""   # error swallowed inside registry
            hist.add.return_value = MagicMock()

            result = run_prediction("text", "en_core_web_sm")

        assert isinstance(result, PredictResponse)