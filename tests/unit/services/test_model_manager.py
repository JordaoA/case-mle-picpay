"""
unit/services/test_model_manager.py
-------------------------------------
ModelManager — validate, cache operations, download guard, _deleted set.
All spaCy and MLflow calls are patched; no real models are downloaded.
"""

import threading
from unittest.mock import MagicMock, call, patch

import pytest

from app.services.model_manager import ALLOWED_MODELS, ModelManager


def _make(installed=False, loaded_names=None):
    """
    Returns a ModelManager whose _refresh_cache() is a no-op.
    Optionally pre-populate _loaded with mock nlp objects.
    """
    with patch("app.services.model_manager.spacy.util.is_package", return_value=installed), \
         patch("app.services.model_manager.spacy.load", return_value=MagicMock()):
        mm = ModelManager()

    if loaded_names:
        for name in loaded_names:
            mm._loaded[name] = MagicMock()

    return mm


@pytest.mark.unit
class TestValidate:

    def test_valid_model_passes(self):
        mm = _make()
        mm._validate("en_core_web_sm")  # no exception

    def test_invalid_model_raises(self):
        mm = _make()
        with pytest.raises(ValueError, match="not supported"):
            mm._validate("some_invalid_model")

    def test_all_allowed_models_pass(self):
        mm = _make()
        for name in ALLOWED_MODELS:
            mm._validate(name)


@pytest.mark.unit
class TestEnsureAvailable:

    def test_returns_already_available_if_installed(self):
        mm = _make()
        mock_nlp = MagicMock()

        with patch.object(mm, "_is_installed", return_value=True), \
             patch.object(mm, "_load_into_cache"), \
             patch("app.services.mlflow_registry.mlflow_registry") as mock_reg:
            mock_reg.register_model.return_value = {}
            result = mm.ensure_available("en_core_web_sm")

        assert result == "already_available"

    def test_returns_downloaded_if_not_installed(self):
        mm = _make()

        with patch.object(mm, "_is_installed", return_value=False), \
             patch.object(mm, "_download"), \
             patch.object(mm, "_refresh_sys_path"), \
             patch.object(mm, "_load_into_cache"), \
             patch("app.services.mlflow_registry.mlflow_registry") as mock_reg:
            mock_reg.register_model.return_value = {}
            result = mm.ensure_available("en_core_web_sm")

        assert result == "downloaded"

    def test_download_called_when_not_installed(self):
        mm = _make()

        with patch.object(mm, "_is_installed", return_value=False), \
             patch.object(mm, "_download") as mock_dl, \
             patch.object(mm, "_refresh_sys_path"), \
             patch.object(mm, "_load_into_cache"), \
             patch("app.services.mlflow_registry.mlflow_registry") as mock_reg:
            mock_reg.register_model.return_value = {}
            mm.ensure_available("en_core_web_sm")

        mock_dl.assert_called_once_with("en_core_web_sm")

    def test_download_not_called_when_installed(self):
        mm = _make()

        with patch.object(mm, "_is_installed", return_value=True), \
             patch.object(mm, "_download") as mock_dl, \
             patch.object(mm, "_load_into_cache"), \
             patch("app.services.mlflow_registry.mlflow_registry") as mock_reg:
            mock_reg.register_model.return_value = {}
            mm.ensure_available("en_core_web_sm")

        mock_dl.assert_not_called()

    def test_mlflow_register_always_called(self):
        mm = _make()

        with patch.object(mm, "_is_installed", return_value=True), \
             patch.object(mm, "_load_into_cache"), \
             patch("app.services.mlflow_registry.mlflow_registry") as mock_reg:
            mock_reg.register_model.return_value = {}
            mm.ensure_available("en_core_web_sm")

        mock_reg.register_model.assert_called_once_with("en_core_web_sm")

    def test_clears_deleted_set(self):
        mm = _make()
        mm._deleted.add("en_core_web_sm")

        with patch.object(mm, "_is_installed", return_value=True), \
             patch.object(mm, "_load_into_cache"), \
             patch("app.services.mlflow_registry.mlflow_registry") as mock_reg:
            mock_reg.register_model.return_value = {}
            mm.ensure_available("en_core_web_sm")

        assert "en_core_web_sm" not in mm._deleted

    def test_invalid_model_raises_before_any_io(self):
        mm = _make()
        with pytest.raises(ValueError, match="not supported"):
            mm.ensure_available("bad_model_xyz")


@pytest.mark.unit
class TestGet:

    def test_returns_from_cache_immediately(self):
        mm = _make()
        mock_nlp = MagicMock()
        mm._loaded["en_core_web_sm"] = mock_nlp

        with patch.object(mm, "_is_installed") as mock_installed:
            result = mm.get("en_core_web_sm")

        # _is_installed must never be called when model is in cache
        mock_installed.assert_not_called()
        assert result is mock_nlp

    def test_raises_when_not_installed(self):
        mm = _make()
        with patch.object(mm, "_is_installed", return_value=False):
            with pytest.raises(ValueError, match="not available"):
                mm.get("en_core_web_sm")

    def test_raises_with_load_hint_in_message(self):
        mm = _make()
        with patch.object(mm, "_is_installed", return_value=False):
            with pytest.raises(ValueError, match="POST /load/"):
                mm.get("en_core_web_sm")

    def test_loads_from_disk_and_caches(self):
        mm = _make()
        mock_nlp = MagicMock()

        with patch.object(mm, "_is_installed", return_value=True), \
             patch("app.services.model_manager.spacy.load", return_value=mock_nlp):
            result = mm.get("en_core_web_sm")

        assert result is mock_nlp
        assert mm._loaded["en_core_web_sm"] is mock_nlp

    def test_raises_for_deleted_model(self):
        mm = _make()
        mm._deleted.add("en_core_web_sm")

        with pytest.raises(ValueError, match="deleted"):
            mm.get("en_core_web_sm")

    def test_deleted_check_prevents_disk_fallback(self):
        mm = _make()
        mm._deleted.add("en_core_web_sm")

        with patch.object(mm, "_is_installed") as mock_inst, \
             pytest.raises(ValueError):
            mm.get("en_core_web_sm")

        mock_inst.assert_not_called()

    def test_invalid_model_raises(self):
        mm = _make()
        with pytest.raises(ValueError, match="not supported"):
            mm.get("bad_model")


@pytest.mark.unit
class TestDelete:

    def test_evicts_from_cache(self):
        mm = _make(loaded_names=["en_core_web_sm"])
        with patch.object(mm, "_is_installed", return_value=True), \
             patch("app.services.mlflow_registry.mlflow_registry"):
            mm.delete("en_core_web_sm")
        assert "en_core_web_sm" not in mm._loaded

    def test_adds_to_deleted_set(self):
        mm = _make(loaded_names=["en_core_web_sm"])
        with patch.object(mm, "_is_installed", return_value=True), \
             patch("app.services.mlflow_registry.mlflow_registry"):
            mm.delete("en_core_web_sm")
        assert "en_core_web_sm" in mm._deleted

    def test_raises_if_not_loaded_and_not_installed(self):
        mm = _make()
        with patch.object(mm, "_is_installed", return_value=False):
            with pytest.raises(ValueError, match="not found"):
                mm.delete("en_core_web_sm")

    def test_archives_in_mlflow(self):
        mm = _make(loaded_names=["en_core_web_sm"])
        with patch.object(mm, "_is_installed", return_value=True), \
             patch("app.services.mlflow_registry.mlflow_registry") as mock_reg:
            mm.delete("en_core_web_sm")
        mock_reg.delete_registered_model.assert_called_once_with("en_core_web_sm")

    def test_invalid_model_raises_before_io(self):
        mm = _make()
        with pytest.raises(ValueError, match="not supported"):
            mm.delete("bad_model")

    def test_delete_then_get_raises(self):
        """Core regression: deleting must prevent get() from reloading from disk."""
        mm = _make(loaded_names=["en_core_web_sm"])
        with patch.object(mm, "_is_installed", return_value=True), \
             patch("app.services.mlflow_registry.mlflow_registry"):
            mm.delete("en_core_web_sm")

        with pytest.raises(ValueError, match="deleted"):
            mm.get("en_core_web_sm")

    def test_delete_then_reload_clears_deleted(self):
        """ensure_available() after delete() must make the model available again."""
        mm = _make(loaded_names=["en_core_web_sm"])

        # Delete
        with patch.object(mm, "_is_installed", return_value=True), \
             patch("app.services.mlflow_registry.mlflow_registry"):
            mm.delete("en_core_web_sm")

        assert "en_core_web_sm" in mm._deleted

        # Re-load
        with patch.object(mm, "_is_installed", return_value=True), \
             patch.object(mm, "_load_into_cache"), \
             patch("app.services.mlflow_registry.mlflow_registry") as mock_reg:
            mock_reg.register_model.return_value = {}
            mm.ensure_available("en_core_web_sm")

        assert "en_core_web_sm" not in mm._deleted


@pytest.mark.unit
class TestListModels:

    def test_enriches_loaded_flag_true(self):
        mm = _make(loaded_names=["en_core_web_sm"])
        with patch("app.services.mlflow_registry.mlflow_registry") as mock_reg:
            mock_reg.list_registered_models.return_value = [
                {"name": "en_core_web_sm", "version": "1", "stage": "Production", "run_id": None}
            ]
            result = mm.list_models()
        assert result[0]["loaded"] is True

    def test_enriches_loaded_flag_false(self):
        mm = _make()  # nothing in cache
        with patch("app.services.mlflow_registry.mlflow_registry") as mock_reg:
            mock_reg.list_registered_models.return_value = [
                {"name": "en_core_web_sm", "version": "1", "stage": "Production", "run_id": None}
            ]
            result = mm.list_models()
        assert result[0]["loaded"] is False

    def test_returns_empty_when_registry_empty(self):
        mm = _make()
        with patch("app.services.mlflow_registry.mlflow_registry") as mock_reg:
            mock_reg.list_registered_models.return_value = []
            assert mm.list_models() == []


@pytest.mark.unit
class TestLoadedModelNames:

    def test_returns_names_in_cache(self):
        mm = _make(loaded_names=["en_core_web_sm", "en_core_web_md"])
        names = mm.loaded_model_names()
        assert set(names) == {"en_core_web_sm", "en_core_web_md"}

    def test_empty_when_nothing_loaded(self):
        mm = _make()
        assert mm.loaded_model_names() == []


@pytest.mark.unit
class TestLoadIntoCache:

    def test_skips_if_already_in_cache(self):
        mm = _make()
        mock_nlp = MagicMock()
        mm._loaded["en_core_web_sm"] = mock_nlp

        with patch("app.services.model_manager.spacy.load") as mock_load:
            mm._load_into_cache("en_core_web_sm")

        mock_load.assert_not_called()
        assert mm._loaded["en_core_web_sm"] is mock_nlp

    def test_raises_if_spacy_load_fails(self):
        mm = _make()
        with patch("app.services.model_manager.spacy.load", side_effect=OSError("not found")):
            with pytest.raises(ValueError, match="could not be loaded"):
                mm._load_into_cache("en_core_web_sm")