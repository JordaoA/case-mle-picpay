"""
Unit tests for the ModelManager.
Patches external subprocess and spacy calls to prevent actual downloads.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.services.model_manager import ModelManager


@pytest.fixture
def manager() -> ModelManager:
    """Provides a fresh ModelManager, skipping the initial cache refresh."""
    with patch.object(ModelManager, "_refresh_cache"):
        return ModelManager()


class TestModelManager:
    @patch("app.services.model_manager.spacy.util.is_package")
    @patch("app.services.model_manager.subprocess.run")
    @patch("app.services.model_manager.spacy.load")
    @patch("app.services.model_manager.ModelManager._refresh_sys_path")
    def test_ensure_available_triggers_download(
        self, mock_refresh_sys, mock_load, mock_subprocess, mock_is_package, manager
    ) -> None:
        """Verifies missing models trigger a subprocess download."""
        mock_is_package.return_value = False
        mock_subprocess.return_value = MagicMock(returncode=0)

        status = manager.ensure_available("en_core_web_sm")

        assert status == "downloaded"
        mock_subprocess.assert_called_once()
        mock_refresh_sys.assert_called_once()
        mock_load.assert_called_once()

    def test_get_from_cache(self, manager) -> None:
        """Verifies models are served from memory if already loaded."""
        mock_nlp = MagicMock()
        manager._loaded["en_core_web_sm"] = mock_nlp
        assert manager.get("en_core_web_sm") == mock_nlp