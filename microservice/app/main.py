"""
main.py
-------
FastAPI application entrypoint.
Configures logging, registers routers, and exposes the health check endpoint.
"""
import logging
import logging.config

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.routers import models, predictions
from app.schemas.requests import HealthResponse
from app.services.model_manager import model_manager
from app.storage.history import PredictionHistory, prediction_history

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json_like": {
            "format": (
                '{"time": "%(asctime)s", "level": "%(levelname)s", '
                '"logger": "%(name)s", "message": "%(message)s"}'
            ),
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json_like",
            "stream": "ext://sys.stdout",
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("ner_service.main")

app = FastAPI(
    title="PicPay NER Inference Service",
    description=(
        "Named Entity Recognition microservice for extracting structured "
        "information from financial transaction messages. "
        "Built for the PicPay MLE technical case."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(models.router)
app.include_router(predictions.router)


@app.get(
    "/health/",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Service health check",
)
def health() -> HealthResponse:
    """Returns service status, loaded models, and total predictions count."""
    is_redis = not isinstance(prediction_history, PredictionHistory)
    redis_ok = prediction_history.ping() if hasattr(prediction_history, "ping") else True

    return HealthResponse(
        status="ok",
        loaded_models=model_manager.loaded_model_names(),
        total_predictions=prediction_history.count(),
        history_backend="redis" if is_redis else "in-memory",
        redis_connected=redis_ok,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> JSONResponse:
    logger.exception(f"Unhandled exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("NER Inference Service starting up...")
    available = model_manager.list_models()
    if available:
        logger.info(f"Pre-registered models: {[m['name'] for m in available]}")
    else:
        logger.info("No pre-installed models found. Use POST /load/ to register one.")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    logger.info("NER Inference Service shutting down.")