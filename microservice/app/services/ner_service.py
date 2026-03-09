"""
services/ner_service.py
------------------------
Encapsulates the NER inference logic.
Delegates model loading to ModelManager, history to PredictionHistory,
and prediction telemetry to MLflowRegistry.
"""

import logging
import time
from datetime import datetime, timezone

from app.schemas.requests import EntityResult, PredictResponse
from app.services.model_manager import model_manager
from app.services.mlflow_registry import mlflow_registry
from app.storage.history import prediction_history

logger = logging.getLogger("ner_service.inference")


def run_prediction(text: str, model_name: str) -> PredictResponse:
    """
    Loads the requested spaCy model, runs NER on the input text,
    logs the prediction to MLflow, persists to history, and returns
    a structured response.

    Args:
        text: Raw input text (English).
        model_name: spaCy model identifier (e.g. "en_core_web_sm").

    Returns:
        PredictResponse with extracted entities and metadata.

    Raises:
        ValueError: If the model is not available or the text is empty.
    """
    text = text.strip()
    if not text:
        raise ValueError("Input text must not be empty.")

    nlp = model_manager.get(model_name)

    model_info = mlflow_registry.get_model_info(model_name)
    model_version = model_info["version"] if model_info else "unknown"

    logger.info(
        f"Running NER — model: '{model_name}' v{model_version} | "
        f"input length: {len(text)} chars"
    )

    t0 = time.perf_counter()
    doc = nlp(text)
    latency_ms = (time.perf_counter() - t0) * 1000

    entities = [
        EntityResult(
            label=ent.label_,
            text=ent.text,
            start=ent.start_char,
            end=ent.end_char,
        )
        for ent in doc.ents
    ]

    timestamp = datetime.now(tz=timezone.utc)

    mlflow_registry.log_prediction(
        model_name=model_name,
        model_version=model_version,
        input_text=text,
        entities=[{"label": e.label, "text": e.text} for e in entities],
        latency_ms=latency_ms,
    )

    prediction_history.add(
        input_text=text,
        output=entities,
        model=model_name,
        timestamp=timestamp,
    )

    logger.info(
        f"Prediction complete — {len(entities)} entities | "
        f"{latency_ms:.1f}ms | "
        f"[{', '.join(e.label for e in entities)}]"
    )

    return PredictResponse(
        model=model_name,
        model_version=model_version,
        text=text,
        entities=entities,
        timestamp=timestamp,
    )