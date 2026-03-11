"""
main.py
-------
FastAPI application entrypoint.
Configures logging, registers routers, and exposes the health check endpoint.
"""
import logging
import logging.config

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from app.services import get_model_manager
from app.storage import get_history 
from app.routers import models, predictions
from app.schemas.requests import HealthResponse
from app.storage.history import PredictionHistory

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
)
def health(
    history=Depends(get_history),
    manager=Depends(get_model_manager),
) -> dict:
    is_mongo = not isinstance(history, PredictionHistory)
    mongo_ok = history.ping() if hasattr(history, "ping") else True

    return {
        "status": "ok",
        "loaded_models": manager.list_models(),
        "total_predictions": history.count(),
        "history_backend": "mongodb" if is_mongo else "in-memory",
        "mongodb_connected": mongo_ok,
    }


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
    manager = get_model_manager()
    available = manager.list_models()
    
    if available:
        logger.info(f"Pre-loaded models: {available}")
    else:
        logger.info("No pre-installed models found. Use POST /load/ to register one.")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    logger.info("NER Inference Service shutting down.")