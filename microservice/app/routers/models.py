"""
routers/models.py
------------------
Endpoints related to model lifecycle management.

Routes:
    POST   /load/              — Download and register a spaCy model
    GET    /models/            — List all available models
    DELETE /models/{model_name} — Remove a model from the registry
"""

import logging

from fastapi import APIRouter, HTTPException, status

from app.schemas.requests import (
    ListModelsResponse,
    LoadModelRequest,
    LoadModelResponse,
    ModelInfo,
)
from app.services import model_manager

logger = logging.getLogger("ner_service.router.models")
router = APIRouter(tags=["Models"])


@router.post(
    "/load/",
    response_model=LoadModelResponse,
    status_code=status.HTTP_200_OK,
    summary="Download and register a spaCy model",
)
def load_model(payload: LoadModelRequest) -> LoadModelResponse:
    """
    Checks if the requested spaCy model is already installed.
    If not, downloads it and makes it available for inference.
    """
    logger.info(f"Load request received for model: '{payload.model}'")

    try:
        result_status = model_manager.ensure_available(payload.model)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    messages = {
        "downloaded": f"Model '{payload.model}' downloaded and registered successfully.",
        "already_available": f"Model '{payload.model}' is already available.",
    }

    return LoadModelResponse(
        model=payload.model,
        status=result_status,
        message=messages[result_status],
    )


@router.get(
    "/models/",
    response_model=ListModelsResponse,
    summary="List all available models",
)
def list_models() -> ListModelsResponse:
    """Returns all models that have been downloaded, with their load status."""
    models = model_manager.list_models()
    return ListModelsResponse(
        models=[ModelInfo(**m) for m in models]
    )


@router.delete(
    "/models/{model_name}",
    status_code=status.HTTP_200_OK,
    summary="Remove a model from the registry",
)
def delete_model(model_name: str) -> dict:
    """
    Unloads a model from memory and removes it from the available registry.
    Does NOT uninstall the spaCy package — just de-registers it from this service.
    """
    logger.info(f"Delete request for model: '{model_name}'")

    try:
        model_manager.delete(model_name)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )

    return {"message": f"Model '{model_name}' removed from registry."}