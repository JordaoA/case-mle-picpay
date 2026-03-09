"""
routers/models.py
------------------
Endpoints related to model lifecycle management.

Routes:
    POST   /load/                — Download and register a spaCy model in MLflow
    GET    /models/              — List all models from MLflow registry
    DELETE /models/{model_name}  — Archive a model in MLflow and evict from cache
"""

import logging

from fastapi import APIRouter, HTTPException, status

from app.schemas.requests import (
    ListModelsResponse,
    LoadModelRequest,
    LoadModelResponse,
    ModelInfo,
)
from app.services.model_manager import model_manager
from app.services.mlflow_registry import mlflow_registry

logger = logging.getLogger("ner_service.router.models")
router = APIRouter(tags=["Models"])


@router.post(
    "/load/",
    response_model=LoadModelResponse,
    status_code=status.HTTP_200_OK,
    summary="Download, register and promote a spaCy model to Production in MLflow",
)
def load_model(payload: LoadModelRequest) -> LoadModelResponse:
    """
    Downloads the spaCy model if not already installed, then:
    - Creates a RegisteredModel entry in MLflow (if first time)
    - Logs a new ModelVersion with metadata (spaCy version, timestamp)
    - Transitions the new version to **Production**
    - Archives any previous Production version automatically
    """
    logger.info(f"Load request received for model: '{payload.model}'")

    try:
        download_status = model_manager.ensure_available(payload.model)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    model_info = mlflow_registry.get_model_info(payload.model)

    messages = {
        "downloaded":       f"Model '{payload.model}' downloaded and registered in MLflow.",
        "already_available": f"Model '{payload.model}' already available — new version registered.",
    }

    return LoadModelResponse(
        model=payload.model,
        status=download_status,
        message=messages[download_status],
        mlflow_version=model_info["version"] if model_info else None,
        mlflow_stage=model_info["stage"] if model_info else None,
        mlflow_run_id=model_info["run_id"] if model_info else None,
    )


@router.get(
    "/models/",
    response_model=ListModelsResponse,
    summary="List all models from the MLflow registry",
)
def list_models() -> ListModelsResponse:
    """
    Returns all registered models from MLflow, enriched with
    whether they are currently loaded in the in-memory cache.
    """
    models = model_manager.list_models()
    return ListModelsResponse(
        models=[ModelInfo(**m) for m in models]
    )


@router.delete(
    "/models/{model_name}",
    status_code=status.HTTP_200_OK,
    summary="Archive a model in MLflow and evict from memory cache",
)
def delete_model(model_name: str) -> dict:
    """
    Archives all versions of the model in the MLflow registry
    and evicts it from the in-memory cache.

    Note: This does NOT uninstall the spaCy package or hard-delete
    the MLflow entry — the audit history is preserved.
    """
    logger.info(f"Delete request for model: '{model_name}'")

    try:
        model_manager.delete(model_name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    return {"message": f"Model '{model_name}' archived in MLflow and evicted from cache."}