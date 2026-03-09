"""
routers/predictions.py
-----------------------
Endpoints related to NER inference and prediction history.

Routes:
    POST /predict/  — Run NER inference on a text
    GET  /list/     — Return the full prediction history
"""

import logging

from fastapi import APIRouter, HTTPException, status

from app.schemas.requests import (
    ListPredictionsResponse,
    PredictRequest,
    PredictResponse,
)
from app.services.ner_service import run_prediction
from app.storage.history import prediction_history

logger = logging.getLogger("ner_service.router.predictions")
router = APIRouter(tags=["Predictions"])


@router.post(
    "/predict/",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Run NER inference on a text",
)
def predict(payload: PredictRequest) -> PredictResponse:
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
        response = run_prediction(text=payload.text, model_name=payload.model)
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
def list_predictions() -> ListPredictionsResponse:
    """
    Returns the full history of predictions made since the service started.
    Each record includes the input text, extracted entities, model used, and timestamp.
    """
    records = prediction_history.all()
    return ListPredictionsResponse(
        total=len(records),
        predictions=records,
    )