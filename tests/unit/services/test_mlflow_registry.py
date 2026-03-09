"""
unit/services/test_mlflow_registry.py
---------------------------------------
MLflowRegistry — all methods tested with a mocked MlflowClient.
No real MLflow server is contacted.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pytest

from app.services.mlflow_registry import MLflowRegistry


def _make_registry() -> MLflowRegistry:
    """
    Build an MLflowRegistry with a fake client injected directly.
    __init__ is bypassed to avoid real MLflow calls.

    Also resets the module-level lazy singleton so this fresh instance
    is returned by _get_registry() for the duration of the test.
    The autouse fixture below restores the original value afterwards.
    """
    reg = MLflowRegistry.__new__(MLflowRegistry)
    reg._client = MagicMock()
    reg._experiment_id = "exp-001"
    return reg


@pytest.fixture(autouse=True)
def _reset_singleton():
    """
    Each test gets a clean lazy singleton slot.
    Prevents state from one test leaking into the next.
    """
    import app.services.mlflow_registry as mod
    original = mod._registry_instance
    mod._registry_instance = None
    yield
    mod._registry_instance = original


@pytest.mark.unit
class TestRegisterModel:

    def _make_mv(self, version="1", run_id="run-abc"):
        mv = MagicMock()
        mv.version = version
        mv.run_id = run_id
        return mv

    def test_returns_dict_with_expected_keys(self):
        reg = _make_registry()
        reg._client.get_latest_versions.return_value = []

        mv = self._make_mv()
        with patch("app.services.mlflow_registry.mlflow") as mock_mlflow, \
             patch.object(reg, "_ensure_registered_model"), \
             patch.object(reg, "_promote_to_production"):

            mock_run = MagicMock()
            mock_run.info.run_id = "run-abc"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            mock_mlflow.register_model.return_value = mv

            result = reg.register_model("en_core_web_sm")

        assert "name" in result
        assert "version" in result
        assert "stage" in result
        assert "run_id" in result

    def test_ensure_registered_model_called(self):
        reg = _make_registry()

        mv = self._make_mv()
        with patch("app.services.mlflow_registry.mlflow") as mock_mlflow, \
             patch.object(reg, "_ensure_registered_model") as mock_ensure, \
             patch.object(reg, "_promote_to_production"):

            mock_run = MagicMock()
            mock_run.info.run_id = "run-abc"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            mock_mlflow.register_model.return_value = mv

            reg.register_model("en_core_web_sm")

        mock_ensure.assert_called_once_with("en_core_web_sm")

    def test_falls_back_to_create_model_version_on_register_error(self):
        reg = _make_registry()

        fallback_mv = self._make_mv(version="2")
        with patch("app.services.mlflow_registry.mlflow") as mock_mlflow, \
             patch.object(reg, "_ensure_registered_model"), \
             patch.object(reg, "_promote_to_production"):

            mock_run = MagicMock()
            mock_run.info.run_id = "run-xyz"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            mock_mlflow.register_model.side_effect = Exception("artifact not found")
            reg._client.create_model_version.return_value = fallback_mv

            result = reg.register_model("en_core_web_sm")

        reg._client.create_model_version.assert_called_once()
        assert result["version"] == "2"


@pytest.mark.unit
class TestGetModelInfo:

    def _make_version(self, version="1", stage="Production", run_id="run-001"):
        v = MagicMock()
        v.name = "en_core_web_sm"
        v.version = version
        v.current_stage = stage
        v.run_id = run_id
        v.description = "test"
        v.creation_timestamp = 1_700_000_000_000
        return v

    def test_returns_production_version(self):
        reg = _make_registry()
        reg._client.get_latest_versions.return_value = [self._make_version()]

        result = reg.get_model_info("en_core_web_sm")

        assert result is not None
        assert result["version"] == "1"
        assert result["stage"] == "Production"

    def test_returns_none_when_no_versions(self):
        reg = _make_registry()
        reg._client.get_latest_versions.return_value = []

        result = reg.get_model_info("en_core_web_sm")

        assert result is None

    def test_returns_none_on_mlflow_exception(self):
        import mlflow
        reg = _make_registry()
        reg._client.get_latest_versions.side_effect = mlflow.exceptions.MlflowException("not found")

        result = reg.get_model_info("en_core_web_sm")

        assert result is None

    def test_falls_back_to_any_stage_when_no_production(self):
        reg = _make_registry()
        staging_v = self._make_version(stage="Staging")

        def side_effect(name, stages=None):
            if stages == ["Production"]:
                return []
            return [staging_v]

        reg._client.get_latest_versions.side_effect = side_effect

        result = reg.get_model_info("en_core_web_sm")
        assert result["stage"] == "Staging"

    def test_result_contains_all_expected_keys(self):
        reg = _make_registry()
        reg._client.get_latest_versions.return_value = [self._make_version()]

        result = reg.get_model_info("en_core_web_sm")
        for key in ("name", "version", "stage", "run_id", "description", "created_at"):
            assert key in result


@pytest.mark.unit
class TestListRegisteredModels:

    def _make_rm(self, name, stage="Production"):
        v = MagicMock()
        v.version = "1"
        v.current_stage = stage
        v.run_id = "run-001"

        rm = MagicMock()
        rm.name = name
        rm.latest_versions = [v]
        return rm

    def test_returns_list_of_dicts(self):
        reg = _make_registry()
        reg._client.search_registered_models.return_value = [
            self._make_rm("en_core_web_sm")
        ]
        result = reg.list_registered_models()
        assert isinstance(result, list)
        assert result[0]["name"] == "en_core_web_sm"

    def test_returns_empty_on_exception(self):
        reg = _make_registry()
        reg._client.search_registered_models.side_effect = Exception("unavailable")
        result = reg.list_registered_models()
        assert result == []

    def test_prefers_production_version(self):
        reg = _make_registry()

        prod_v = MagicMock()
        prod_v.version = "3"
        prod_v.current_stage = "Production"
        prod_v.run_id = "run-prod"

        staging_v = MagicMock()
        staging_v.version = "2"
        staging_v.current_stage = "Staging"
        staging_v.run_id = "run-staging"

        rm = MagicMock()
        rm.name = "en_core_web_sm"
        rm.latest_versions = [staging_v, prod_v]

        reg._client.search_registered_models.return_value = [rm]
        result = reg.list_registered_models()
        assert result[0]["version"] == "3"

    def test_handles_model_with_no_versions(self):
        reg = _make_registry()

        rm = MagicMock()
        rm.name = "en_core_web_sm"
        rm.latest_versions = []

        reg._client.search_registered_models.return_value = [rm]
        result = reg.list_registered_models()
        assert result[0]["version"] == "—"


@pytest.mark.unit
class TestDeleteRegisteredModel:

    def test_archives_non_archived_versions(self):
        reg = _make_registry()

        v1 = MagicMock()
        v1.version = "1"
        v1.current_stage = "Production"

        v2 = MagicMock()
        v2.version = "2"
        v2.current_stage = "Archived"

        reg._client.search_model_versions.return_value = [v1, v2]

        reg.delete_registered_model("en_core_web_sm")

        # Only v1 should be transitioned; v2 is already Archived
        reg._client.transition_model_version_stage.assert_called_once_with(
            name="en_core_web_sm",
            version="1",
            stage="Archived",
        )

    def test_raises_value_error_on_mlflow_exception(self):
        import mlflow
        reg = _make_registry()
        reg._client.search_model_versions.side_effect = mlflow.exceptions.MlflowException("no")

        with pytest.raises(ValueError, match="not found in registry"):
            reg.delete_registered_model("en_core_web_sm")

    def test_no_transition_calls_when_all_already_archived(self):
        reg = _make_registry()

        v = MagicMock()
        v.version = "1"
        v.current_stage = "Archived"
        reg._client.search_model_versions.return_value = [v]

        reg.delete_registered_model("en_core_web_sm")

        reg._client.transition_model_version_stage.assert_not_called()


@pytest.mark.unit
class TestLogPrediction:

    def test_returns_run_id(self):
        reg = _make_registry()

        with patch("app.services.mlflow_registry.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "pred-run-001"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            run_id = reg.log_prediction(
                model_name="en_core_web_sm",
                model_version="1",
                input_text="Send $45",
                entities=[{"label": "MONEY", "text": "$45"}],
                latency_ms=12.5,
            )

        assert run_id == "pred-run-001"

    def test_logs_correct_params(self):
        reg = _make_registry()

        with patch("app.services.mlflow_registry.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "r1"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            reg.log_prediction(
                model_name="en_core_web_sm",
                model_version="2",
                input_text="Hello",
                entities=[{"label": "PERSON", "text": "Alice"}],
                latency_ms=5.0,
            )

        logged_params = mock_mlflow.log_params.call_args[0][0]
        assert logged_params["model_name"] == "en_core_web_sm"
        assert logged_params["model_version"] == "2"
        assert "PERSON" in logged_params["entity_labels"]

    def test_logs_correct_metrics(self):
        reg = _make_registry()

        with patch("app.services.mlflow_registry.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "r1"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            reg.log_prediction(
                model_name="en_core_web_sm",
                model_version="1",
                input_text="Send $45 to Michael",
                entities=[{"label": "MONEY", "text": "$45"}],
                latency_ms=30.0,
            )

        logged_metrics = mock_mlflow.log_metrics.call_args[0][0]
        assert logged_metrics["num_entities"] == 1
        assert logged_metrics["latency_ms"] == 30.0
        assert logged_metrics["input_length"] == len("Send $45 to Michael")

    def test_returns_empty_string_on_exception(self):
        reg = _make_registry()

        with patch("app.services.mlflow_registry.mlflow") as mock_mlflow:
            mock_mlflow.start_run.side_effect = Exception("mlflow down")

            run_id = reg.log_prediction(
                model_name="en_core_web_sm",
                model_version="1",
                input_text="text",
                entities=[],
                latency_ms=0.0,
            )

        assert run_id == ""

    def test_no_entities_logs_none_label(self):
        reg = _make_registry()

        with patch("app.services.mlflow_registry.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "r1"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            reg.log_prediction(
                model_name="en_core_web_sm",
                model_version="1",
                input_text="text",
                entities=[],
                latency_ms=0.0,
            )

        logged_params = mock_mlflow.log_params.call_args[0][0]
        assert logged_params["entity_labels"] == "none"