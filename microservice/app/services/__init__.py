from app.services.model_manager import ModelManager
from app.services.mlflow_registry import MLflowRegistry

_model_manager = ModelManager()
_mlflow_registry = MLflowRegistry()

def get_model_manager() -> ModelManager:
    return _model_manager

def get_mlflow_registry() -> MLflowRegistry:
    return _mlflow_registry