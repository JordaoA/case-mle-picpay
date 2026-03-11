"""
unit/test_config.py
--------------------
Settings — reading from env and property composition.
"""

import pytest


@pytest.mark.unit
class TestSettingsDefaults:
    """
    Each test sets the env var to its expected production default value,
    then verifies Settings() reads it correctly.
    """

    def test_mlflow_host_default(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_HOST", "mlflow")
        from app.config import Settings
        assert Settings().mlflow_host == "mlflow"

    def test_mlflow_port_default(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_PORT", "5000")
        from app.config import Settings
        assert Settings().mlflow_port == "5000"

    def test_mlflow_tracking_uri_property(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_HOST", "myhost")
        monkeypatch.setenv("MLFLOW_PORT", "9999")
        from app.config import Settings
        assert Settings().mlflow_tracking_uri == "http://myhost:9999"

    def test_mongo_uri_default(self, monkeypatch):
        monkeypatch.setenv("MONGO_URI", "mongodb://mongo:27017")
        from app.config import Settings
        assert Settings().mongo_uri == "mongodb://mongo:27017"

    def test_mongo_db_name_default(self, monkeypatch):
        monkeypatch.setenv("MONGO_DB_NAME", "picpay_ner")
        from app.config import Settings
        assert Settings().mongo_db_name == "picpay_ner"


@pytest.mark.unit
class TestSettingsEnvOverrides:
    """
    Each test sets a non-default value and verifies Settings() picks it up.
    """

    def test_mlflow_host_reads_env(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_HOST", "prod-mlflow")
        from app.config import Settings
        assert Settings().mlflow_host == "prod-mlflow"

    def test_mongo_uri_reads_env(self, monkeypatch):
        monkeypatch.setenv("MONGO_URI", "mongodb://prod-db:27017")
        from app.config import Settings
        assert Settings().mongo_uri == "mongodb://prod-db:27017"

    def test_mongo_db_name_reads_env(self, monkeypatch):
        monkeypatch.setenv("MONGO_DB_NAME", "custom_db")
        from app.config import Settings
        assert Settings().mongo_db_name == "custom_db"