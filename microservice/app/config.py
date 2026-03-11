"""
config.py
---------
Centralized configuration via environment variables.
Pydantic Settings reads from .env or the environment automatically.
"""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings


dotenv_path = find_dotenv(filename=".env")

if not Path(dotenv_path).exists():
    os.system("touch .env")

load_dotenv(dotenv_path)


class Settings(BaseSettings):
    
    mlflow_host: str = Field(
        default=os.getenv("MLFLOW_HOST", "mlflow"),
        alias="MLFLOW_HOST",
    )
    mlflow_port: str = Field(
        default=os.getenv("MLFLOW_PORT", "5000"),
        alias="MLFLOW_PORT",
    )
    mlflow_experiment_name: str = Field(
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "ner-inference-service"),
        alias="MLFLOW_EXPERIMENT_NAME",
    )

    service_name: str = Field(
        default=os.getenv("SERVICE_NAME", "picpay-ner-service"),
        alias="SERVICE_NAME",
    )
    service_version: str = Field(
        default=os.getenv("SERVICE_VERSION", "1.0.0"),
        alias="SERVICE_VERSION",
    )
    mongo_uri: str = Field(
        default=os.getenv("MONGO_URI", "mongodb://mongo:27017"),
        alias="MONGO_URI",
    )
    mongo_db_name: str = Field(
        default=os.getenv("MONGO_DB_NAME", "picpay_ner"),
        alias="MONGO_DB_NAME",
    )

    @property
    def mlflow_tracking_uri(self) -> str:
        return f"http://{self.mlflow_host}:{self.mlflow_port}"

    class Config:
        env_file_encoding = "utf-8"

settings = Settings()