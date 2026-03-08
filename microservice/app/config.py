"""
config.py
---------
Centralized configuration via environment variables.
Pydantic Settings reads from .env or the environment automatically.
"""

import os

from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


dotenv_path = find_dotenv(filename='.env')

if not Path(dotenv_path).exists():
    os.system("touch .env")

load_dotenv(dotenv_path)



class Settings(BaseSettings):
    # MLflow
    mlflow_host: str = Field(
        default=os.getenv("MLFLOW_HOST"),
        alias="MLFLOW_HOST",
    )
    mlflow_port: str = Field(
        default=os.getenv("MLFLOW_PORT"),
        alias="MLFLOW_PORT",
    )
    mlflow_experiment_name: str = Field(
        default=os.getenv("MLFLOW_EXPERIMENT_NAME"),
        alias="MLFLOW_EXPERIMENT_NAME",
    )

    # Service
    service_name: str = Field(
        default=os.getenv("SERVICE_NAME"),
        alias="SERVICE_NAME",
    )
    service_version: str = Field(
        default=os.getenv("SERVICE_VERSION"),
        alias="SERVICE_VERSION",
    )
    
    @property
    def mlflow_tracking_uri(self) -> str:
        print(self.mlflow_host, self.mlflow_port, self.mlflow_experiment_name, self.service_name, self.service_version)
        return f"http://{self.mlflow_host}:{self.mlflow_port}"

    class Config:
        env_file_encoding = "utf-8"


# Singleton — imported everywhere
settings = Settings()