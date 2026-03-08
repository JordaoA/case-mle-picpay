"""Unit tests for app/services/model_manager.py"""
import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'microservice'))


class TestModelManager:
    """Tests for ModelManager class."""

    @pytest.fixture
    def mock_spacy(self):
        """Mock spacy module."""
        with patch('app.services.model_manager.spacy') as mock:
            mock.util.is_package = Mock(return_value=False)
            mock.load = Mock(return_value=MagicMock())
            yield mock

    @pytest.mark.unit
    def test_model_manager_init(self, mock_spacy):
        """Test ModelManager initialization."""
        with patch('app.services.model_manager.subprocess'):
            manager = __import__('app.services.model_manager', fromlist=['ModelManager']).ModelManager()
            assert manager is not None
            assert hasattr(manager, '_loaded')
            assert hasattr(manager, '_available')

    @pytest.mark.unit
    def test_ensure_available_model_not_installed(self, mock_spacy):
        """Test ensuring availability when model is not installed."""
        mock_spacy.util.is_package.return_value = False
        
        with patch('app.services.model_manager.subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)
            with patch('app.services.model_manager.logger'):
                manager = __import__('app.services.model_manager', fromlist=['ModelManager']).ModelManager()
                result = manager.ensure_available("en_core_web_sm")
                assert result == "downloaded"

    @pytest.mark.unit
    def test_ensure_available_model_already_installed(self, mock_spacy):
        """Test ensuring availability when model is already installed."""
        mock_spacy.util.is_package.return_value = True
        
        with patch('app.services.model_manager.logger'):
            manager = __import__('app.services.model_manager', fromlist=['ModelManager']).ModelManager()
            result = manager.ensure_available("en_core_web_sm")
            assert result == "already_available"

    @pytest.mark.unit
    def test_ensure_available_invalid_model(self, mock_spacy):
        """Test ensuring availability with invalid model name."""
        with patch('app.services.model_manager.logger'):
            manager = __import__('app.services.model_manager', fromlist=['ModelManager']).ModelManager()
            with pytest.raises(ValueError):
                manager.ensure_available("invalid_model")

    @pytest.mark.unit
    def test_get_loaded_model(self, mock_spacy):
        """Test getting a loaded model."""
        mock_spacy.util.is_package.return_value = True
        mock_nlp = MagicMock()
        mock_spacy.load.return_value = mock_nlp
        
        with patch('app.services.model_manager.logger'):
            manager = __import__('app.services.model_manager', fromlist=['ModelManager']).ModelManager()
            nlp = manager.get("en_core_web_sm")
            assert nlp is not None

    @pytest.mark.unit
    def test_get_model_not_available(self, mock_spacy):
        """Test getting a model that's not available."""
        mock_spacy.util.is_package.return_value = False
        
        with patch('app.services.model_manager.logger'):
            manager = __import__('app.services.model_manager', fromlist=['ModelManager']).ModelManager()
            with pytest.raises(ValueError):
                manager.get("en_core_web_sm")

    @pytest.mark.unit
    def test_get_invalid_model(self, mock_spacy):
        """Test getting an invalid model."""
        with patch('app.services.model_manager.logger'):
            manager = __import__('app.services.model_manager', fromlist=['ModelManager']).ModelManager()
            with pytest.raises(ValueError):
                manager.get("invalid_model")

    @pytest.mark.unit
    def test_delete_model(self, mock_spacy):
        """Test deleting a model."""
        mock_spacy.util.is_package.return_value = True
        mock_nlp = MagicMock()
        mock_spacy.load.return_value = mock_nlp
        
        with patch('app.services.model_manager.logger'):
            manager = __import__('app.services.model_manager', fromlist=['ModelManager']).ModelManager()
            # First load a model
            manager.get("en_core_web_sm")
            # Then delete it
            manager.delete("en_core_web_sm")
            # After deletion, model should be removed from available
            assert "en_core_web_sm" not in manager.loaded_model_names()

    @pytest.mark.unit
    def test_delete_nonexistent_model(self, mock_spacy):
        """Test deleting a model that doesn't exist."""
        with patch('app.services.model_manager.logger'):
            manager = __import__('app.services.model_manager', fromlist=['ModelManager']).ModelManager()
            with pytest.raises(ValueError):
                manager.delete("nonexistent_model")

    @pytest.mark.unit
    def test_list_models_empty(self, mock_spacy):
        """Test listing models when none are available."""
        mock_spacy.util.is_package.return_value = False
        
        with patch('app.services.model_manager.logger'):
            manager = __import__('app.services.model_manager', fromlist=['ModelManager']).ModelManager()
            models = manager.list_models()
            assert isinstance(models, list)
            assert len(models) == 0

    @pytest.mark.unit
    def test_list_models_with_available(self, mock_spacy):
        """Test listing available models."""
        def is_package_side_effect(name):
            return name == "en_core_web_sm"
        
        mock_spacy.util.is_package.side_effect = is_package_side_effect
        mock_nlp = MagicMock()
        mock_spacy.load.return_value = mock_nlp
        
        with patch('app.services.model_manager.logger'):
            manager = __import__('app.services.model_manager', fromlist=['ModelManager']).ModelManager()
            manager.get("en_core_web_sm")
            models = manager.list_models()
            assert len(models) > 0
            assert any(m['name'] == 'en_core_web_sm' for m in models)

    @pytest.mark.unit
    def test_loaded_model_names(self, mock_spacy):
        """Test getting list of loaded model names."""
        mock_spacy.util.is_package.return_value = True
        mock_nlp = MagicMock()
        mock_spacy.load.return_value = mock_nlp
        
        with patch('app.services.model_manager.logger'):
            manager = __import__('app.services.model_manager', fromlist=['ModelManager']).ModelManager()
            manager.get("en_core_web_sm")
            names = manager.loaded_model_names()
            assert "en_core_web_sm" in names

    @pytest.mark.unit
    def test_download_failure(self, mock_spacy):
        """Test handling download failure."""
        mock_spacy.util.is_package.return_value = False
        
        with patch('app.services.model_manager.subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=1, stderr="Download failed")
            with patch('app.services.model_manager.logger'):
                manager = __import__('app.services.model_manager', fromlist=['ModelManager']).ModelManager()
                with pytest.raises(RuntimeError):
                    manager.ensure_available("en_core_web_sm")
