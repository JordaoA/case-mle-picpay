"""
services/ner_service.py
------------------------
Encapsulates the NER inference logic.
Delegates model loading to ModelManager and history logging to PredictionHistory.
"""

import logging
from datetime import datetime, timezone

from app.schemas.requests import EntityResult, PredictResponse
from app.services.model_manager import model_manager
from app.storage.history import prediction_history

logger = logging.getLogger("ner_service.inference")


def run_prediction(text: str, model_name: str) -> PredictResponse:
    """
    Loads the requested spaCy model, runs NER on the input text,
    persists the result to history, and returns a structured response.

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

    logger.info(f"Running NER on text ({len(text)} chars) with model '{model_name}'")
    doc = nlp(text)

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

    prediction_history.add(
        input_text=text,
        output=entities,
        model=model_name,
        timestamp=timestamp,
    )

    logger.info(
        f"Prediction complete — {len(entities)} entities found "
        f"[{', '.join(e.label for e in entities)}]"
    )

    return PredictResponse(
        model=model_name,
        text=text,
        entities=entities,
        timestamp=timestamp,
    )