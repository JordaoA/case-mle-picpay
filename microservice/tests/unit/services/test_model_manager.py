"""
unit/services/test_model_manager.py
-------------------------------------
ModelManager — validate, cache operations, download guard, _deleted set.
All spaCy and MLflow calls are patched; no real models are downloaded.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.services.model_manager import ModelManager

@pytest.fixture
def manager():
    # Patch _refresh_cache to prevent loading real models during instantiation
    with patch.object(ModelManager, '_refresh_cache'):
        return ModelManager()

class TestModelManager:

    @patch("app.services.model_manager.spacy.util.is_package")
    @patch("app.services.model_manager.spacy.load")
    def test_ensure_available_already_installed(self, mock_load, mock_is_package, manager):
        mock_is_package.return_value = True
        mock_nlp = MagicMock()
        mock_load.return_value = mock_nlp

        status = manager.ensure_available("en_core_web_sm")

        assert status == "already_available"
        assert manager._loaded["en_core_web_sm"] == mock_nlp
        mock_load.assert_called_once_with("en_core_web_sm")

    @patch("app.services.model_manager.spacy.util.is_package")
    @patch("app.services.model_manager.subprocess.run")
    @patch("app.services.model_manager.spacy.load")
    @patch("app.services.model_manager.ModelManager._refresh_sys_path")
    def test_ensure_available_triggers_download(
        self, mock_refresh_sys, mock_load, mock_subprocess, mock_is_package, manager
    ):
        mock_is_package.return_value = False
        mock_subprocess.return_value = MagicMock(returncode=0)

        status = manager.ensure_available("en_core_web_sm")

        assert status == "downloaded"
        mock_subprocess.assert_called_once()
        mock_refresh_sys.assert_called_once()
        mock_load.assert_called_once()

    def test_ensure_available_unsupported_model(self, manager):
        with pytest.raises(ValueError, match="is not supported"):
            manager.ensure_available("invalid_model_name")

    def test_get_from_cache(self, manager):
        mock_nlp = MagicMock()
        manager._loaded["en_core_web_sm"] = mock_nlp

        assert manager.get("en_core_web_sm") == mock_nlp

    @patch("app.services.model_manager.spacy.util.is_package")
    def test_get_not_installed(self, mock_is_package, manager):
        mock_is_package.return_value = False
        with pytest.raises(ValueError, match="is not available"):
            manager.get("en_core_web_sm")

    def test_delete_removes_from_cache(self, manager):
        manager._loaded["en_core_web_sm"] = MagicMock()
        
        manager.delete("en_core_web_sm")
        
        assert "en_core_web_sm" not in manager._loaded
        assert "en_core_web_sm" in manager._deleted

    def test_list_models_returns_loaded_keys(self, manager):
        manager._loaded["en_core_web_sm"] = MagicMock()
        manager._loaded["en_core_web_md"] = MagicMock()
        
        loaded = manager.list_models()
        assert set(loaded) == {"en_core_web_sm", "en_core_web_md"}