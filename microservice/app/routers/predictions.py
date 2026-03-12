"""
routers/predictions.py
-----------------------
Endpoints related to NER inference and prediction history.

Routes:
    POST /predict/  — Run NER inference on a text
    GET  /list/     — Return the full prediction history
"""

import logging

from fastapi import Depends, APIRouter, HTTPException, status

from app.schemas.requests import (
    ListPredictionsResponse,
    PredictRequest,
    PredictResponse,
)
from app.services.model_manager import ModelManager
from app.services.mlflow_registry import MLflowRegistry
from app.storage.mongo_history import MongoHistory

from app.services.ner_service import run_prediction
from app.services import get_model_manager, get_mlflow_registry
from app.storage import get_history

logger = logging.getLogger("ner_service.router.predictions")
router = APIRouter(tags=["Predictions"])

@router.post(
    "/predict/",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Run NER inference on a text",
)
def predict(
        payload: PredictRequest,
        history: MongoHistory =Depends(get_history),
        manager: ModelManager = Depends(get_model_manager),
        registry: MLflowRegistry = Depends(get_mlflow_registry)
    ) -> PredictResponse:
    """
    Loads the requested spaCy model (from cache if available) and runs
    Named Entity Recognition on the provided text.

    Returns extracted entities with their label, text span, and character offsets.

    Example input:
        { "text": "Can you send $45 to Michael on June 3?", "model": "en_core_web_sm" }

    Example output entities:
        MONEY → $45 | PERSON → Michael | DATE → June 3
    """
    logger.info(
        f"Predict request — model: '{payload.model}' | "
        f"text length: {len(payload.text)} chars"
    )

    try:
        response = run_prediction(
            text=payload.text, 
            model_name=payload.model,
            model_manager=manager,
            model_registry=registry,
            history_repo=history
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(exc)}",
        )

    return response


@router.get(
    "/list/",
    response_model=ListPredictionsResponse,
    summary="List all predictions made so far",
)
def list_predictions(
        history=Depends(get_history)
    ) -> ListPredictionsResponse:
    """
    Returns the full history of predictions made since the service started.
    Each record includes the input text, extracted entities, model used, and timestamp.
    """
    records = history.all()
    return ListPredictionsResponse(
        total=len(records),
        predictions=records,
    )