"""
unit/test_config.py
--------------------
Settings — reading from env and property composition.

Why we use monkeypatch.setenv (not delenv) for defaults tests:
    config.py uses `default=os.getenv("KEY", "fallback")` inside Field().
    os.getenv() is evaluated ONCE at class definition time — the moment
    the module is first imported. If the root conftest had already set
    MLFLOW_HOST="fake-mlflow" in os.environ when the module was imported,
    the Field default is frozen as "fake-mlflow". Calling monkeypatch.delenv
    afterward cannot change a default that was already baked in at parse time.

    The correct approach is monkeypatch.setenv with the expected value.
    This tests what actually matters: pydantic-settings reads env vars at
    instantiation time and returns the correct value.
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

    def test_mlflow_experiment_default(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "ner-inference-service")
        from app.config import Settings
        assert Settings().mlflow_experiment_name == "ner-inference-service"

    def test_redis_host_default(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "redis")
        from app.config import Settings
        assert Settings().redis_host == "redis"

    def test_redis_port_default(self, monkeypatch):
        monkeypatch.setenv("REDIS_PORT", "6379")
        from app.config import Settings
        assert Settings().redis_port == 6379

    def test_redis_db_default(self, monkeypatch):
        monkeypatch.setenv("REDIS_DB", "0")
        from app.config import Settings
        assert Settings().redis_db == 0

    def test_redis_password_empty(self, monkeypatch):
        monkeypatch.setenv("REDIS_PASSWORD", "")
        from app.config import Settings
        assert Settings().redis_password == ""

    def test_redis_ttl_is_seven_days(self, monkeypatch):
        monkeypatch.setenv("REDIS_TTL_SECONDS", str(60 * 60 * 24 * 7))
        from app.config import Settings
        assert Settings().redis_ttl_seconds == 60 * 60 * 24 * 7

    def test_service_name_default(self, monkeypatch):
        monkeypatch.setenv("SERVICE_NAME", "picpay-ner-service")
        from app.config import Settings
        assert Settings().service_name == "picpay-ner-service"

    def test_service_version_default(self, monkeypatch):
        monkeypatch.setenv("SERVICE_VERSION", "1.0.0")
        from app.config import Settings
        assert Settings().service_version == "1.0.0"


@pytest.mark.unit
class TestSettingsEnvOverrides:
    """
    Each test sets a non-default value and verifies Settings() picks it up.
    This confirms pydantic-settings reads from os.environ at instantiation.
    """

    def test_mlflow_host_reads_env(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_HOST", "prod-mlflow")
        from app.config import Settings
        assert Settings().mlflow_host == "prod-mlflow"

    def test_mlflow_port_reads_env(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_PORT", "9000")
        from app.config import Settings
        assert Settings().mlflow_port == "9000"

    def test_mlflow_tracking_uri_composed_from_host_and_port(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_HOST", "tracker")
        monkeypatch.setenv("MLFLOW_PORT", "1234")
        from app.config import Settings
        assert Settings().mlflow_tracking_uri == "http://tracker:1234"

    def test_redis_host_reads_env(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "prod-redis")
        from app.config import Settings
        assert Settings().redis_host == "prod-redis"

    def test_redis_port_coerced_to_int(self, monkeypatch):
        monkeypatch.setenv("REDIS_PORT", "6380")
        from app.config import Settings
        s = Settings()
        assert s.redis_port == 6380
        assert isinstance(s.redis_port, int)

    def test_redis_db_coerced_to_int(self, monkeypatch):
        monkeypatch.setenv("REDIS_DB", "2")
        from app.config import Settings
        s = Settings()
        assert s.redis_db == 2
        assert isinstance(s.redis_db, int)

    def test_redis_ttl_reads_env(self, monkeypatch):
        monkeypatch.setenv("REDIS_TTL_SECONDS", "9999")
        from app.config import Settings
        assert Settings().redis_ttl_seconds == 9999

    def test_redis_ttl_zero_allowed(self, monkeypatch):
        monkeypatch.setenv("REDIS_TTL_SECONDS", "0")
        from app.config import Settings
        assert Settings().redis_ttl_seconds == 0

    def test_redis_password_reads_env(self, monkeypatch):
        monkeypatch.setenv("REDIS_PASSWORD", "s3cr3t")
        from app.config import Settings
        assert Settings().redis_password == "s3cr3t"