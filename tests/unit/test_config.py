"""Unit tests for app/config.py"""
import os
import pytest
from unittest.mock import patch, Mock
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'microservice'))


class TestSettings:
    """Tests for Settings configuration class."""

    @pytest.mark.unit
    def test_settings_mlflow_defaults(self, env_vars):
        """Test MLflow settings with environment variables."""
        from app.config import Settings
        
        with patch.dict(os.environ, env_vars):
            settings = Settings()
            assert settings.mlflow_host == "localhost"
            assert settings.mlflow_port == "5000"
            assert settings.mlflow_experiment_name == "test-experiment"

    @pytest.mark.unit
    def test_settings_service_defaults(self, env_vars):
        """Test service settings with environment variables."""
        from app.config import Settings
        
        with patch.dict(os.environ, env_vars):
            settings = Settings()
            assert settings.service_name == "test-service"
            assert settings.service_version == "1.0.0"

    @pytest.mark.unit
    def test_settings_redis_defaults(self, env_vars):
        """Test Redis settings with environment variables."""
        from app.config import Settings
        
        with patch.dict(os.environ, env_vars):
            settings = Settings()
            assert settings.redis_host == "localhost"
            assert settings.redis_port == 6379
            assert settings.redis_db == 0
            assert settings.redis_password == ""

    @pytest.mark.unit
    def test_mlflow_tracking_uri_property(self, env_vars):
        """Test mlflow_tracking_uri property construction."""
        from app.config import Settings
        
        with patch.dict(os.environ, env_vars):
            settings = Settings()
            expected_uri = "http://localhost:5000"
            assert settings.mlflow_tracking_uri == expected_uri

    @pytest.mark.unit
    def test_settings_with_none_values(self):
        """Test Settings handles None values gracefully."""
        from app.config import Settings
        
        with patch.dict(os.environ, clear=True):
            with patch('app.config.os.getenv', return_value=None):
                # Should still create settings with defaults
                settings = Settings()
                # Values should be None or defaults
                assert isinstance(settings.redis_port, int)

    @pytest.mark.unit
    def test_settings_redis_port_conversion_to_int(self, env_vars):
        """Test that Redis port is correctly converted to int."""
        from app.config import Settings
        
        with patch.dict(os.environ, {**env_vars, "REDIS_PORT": "6380"}):
            settings = Settings()
            assert isinstance(settings.redis_port, int)
            assert settings.redis_port == 6380

    @pytest.mark.unit
    def test_settings_redis_db_conversion_to_int(self, env_vars):
        """Test that Redis DB is correctly converted to int."""
        from app.config import Settings
        
        with patch.dict(os.environ, {**env_vars, "REDIS_DB": "2"}):
            settings = Settings()
            assert isinstance(settings.redis_db, int)
            assert settings.redis_db == 2
