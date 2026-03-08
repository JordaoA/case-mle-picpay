"""
main.py
-------
FastAPI application entrypoint.
Configures logging, registers routers, and exposes the health check endpoint.
"""

import logging
import logging.config
from fastapi import FastAPI

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