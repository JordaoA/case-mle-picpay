"""
service/model_manager.py
------------------------
Handles downloading, caching, and in-memory loading of spaCy models.

Responsibilities:
- Check if a model is already installed (spacy.util.is_package)
- Download missing models via subprocess (mirrors `python -m spacy download`)
- Load models into memory and cache them to avoid reloading on every request
- Provide a list of all available (downloaded) models
"""

import logging
import subprocess
import sys
import threading

import spacy
from spacy.language import Language

logger = logging.getLogger("ner_service.model_manager")

# Valid spaCy English web models accepted by this service
ALLOWED_MODELS = {
    "en_core_web_sm",
    "en_core_web_md",
    "en_core_web_lg",
    "en_core_web_trf",
}


class ModelManager:
    """Thread-safe spaCy model registry with lazy loading and caching."""

    def __init__(self) -> None:
        self._loaded: dict[str, Language] = {}  # name → spacy model
        self._available: set[str] = set()        # downloaded but maybe not loaded
        self._lock = threading.Lock()
        self._refresh_available()

    def ensure_available(self, model_name: str) -> str:
        """
        Ensures the model is downloaded and registered.
        Returns "downloaded" if it had to be fetched, or "already_available".
        """
        self._validate(model_name)

        if self._is_installed(model_name):
            with self._lock:
                self._available.add(model_name)
            logger.info(f"Model '{model_name}' already installed.")
            return "already_available"

        logger.info(f"Downloading model '{model_name}'...")
        self._download(model_name)
        with self._lock:
            self._available.add(model_name)
        logger.info(f"Model '{model_name}' downloaded successfully.")
        return "downloaded"

    def get(self, model_name: str) -> Language:
        """
        Returns a loaded spaCy model, loading it into memory if needed.
        Raises ValueError if the model is not available (not downloaded).
        """
        self._validate(model_name)

        with self._lock:
            if model_name in self._loaded:
                return self._loaded[model_name]

        # Load outside the lock to avoid blocking other threads
        if not self._is_installed(model_name):
            raise ValueError(
                f"Model '{model_name}' is not available. "
                f"Call POST /load/ first."
            )

        logger.info(f"Loading model '{model_name}' into memory...")
        nlp = spacy.load(model_name)

        with self._lock:
            self._loaded[model_name] = nlp
            self._available.add(model_name)

        logger.info(f"Model '{model_name}' loaded and cached.")
        return nlp

    def delete(self, model_name: str) -> None:
        """Unloads a model from memory and removes it from the registry."""
        with self._lock:
            if model_name not in self._available:
                raise ValueError(f"Model '{model_name}' not found.")
            self._loaded.pop(model_name, None)
            self._available.discard(model_name)
        logger.info(f"Model '{model_name}' removed from registry.")

    def list_models(self) -> list[dict]:
        """Returns all available models with their load status."""
        with self._lock:
            return [
                {
                    "name": name,
                    "status": "loaded" if name in self._loaded else "available",
                }
                for name in sorted(self._available)
            ]

    def loaded_model_names(self) -> list[str]:
        with self._lock:
            return list(self._loaded.keys())

    def _validate(self, model_name: str) -> None:
        if model_name not in ALLOWED_MODELS:
            raise ValueError(
                f"Model '{model_name}' is not supported. "
                f"Allowed models: {sorted(ALLOWED_MODELS)}"
            )

    def _is_installed(self, model_name: str) -> bool:
        return spacy.util.is_package(model_name)

    def _download(self, model_name: str) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to download model '{model_name}':\n{result.stderr}"
            )

    def _refresh_available(self) -> None:
        """Scan for already-installed models on startup."""
        for name in ALLOWED_MODELS:
            if self._is_installed(name):
                self._available.add(name)
        if self._available:
            logger.info(f"Pre-existing models found: {sorted(self._available)}")


# Singleton — shared across the entire app lifetime
model_manager = ModelManager()