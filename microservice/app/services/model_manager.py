"""
services/model_manager.py
--------------------------
Handles downloading, caching, and in-memory loading of spaCy models.

Responsibilities:
- Validate model names against ALLOWED_MODELS
- Download missing spaCy models via subprocess
- Load models into memory and cache them (avoids reloading on every request)
- Delegate all registry concerns (versioning, stages, metadata) to MLflowRegistry

Separation of concerns:
    ModelManager     → spaCy lifecycle (download, load, cache)
    MLflowRegistry   → tracking, versioning, stage transitions
"""

import importlib
import importlib.metadata
import logging
import os
import site
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
    """
    Thread-safe spaCy model loader with in-memory cache.
    Registry metadata is stored in MLflow — not here.
    """

    def __init__(self) -> None:
        self._loaded: dict[str, Language] = {}
        self._deleted: set[str] = set()
        self._lock = threading.Lock()
        self._refresh_cache()

    def ensure_available(self, model_name: str) -> str:
        """
        Ensures the spaCy model package is installed on disk and loaded
        into the in-memory cache.

        Why we load into cache immediately after download:
            pip install --user writes to $PYTHONUSERBASE, a directory that
            didn't exist at interpreter startup. Even after _refresh_sys_path()
            fixes sys.path, loading the model right here means get() will
            always find it in self._loaded and never has to call
            _is_installed() — which avoids any remaining stale-cache edge cases.

        Returns:
            "downloaded"        — model was fetched from the internet
            "already_available" — model was already installed
        """
        self._validate(model_name)

        if self._is_installed(model_name):
            logger.info(f"Model '{model_name}' already installed on disk.")
            status = "already_available"
        else:
            logger.info(f"Downloading model '{model_name}'...")
            self._download(model_name)

            self._refresh_sys_path()
            status = "downloaded"

        with self._lock:
            self._deleted.discard(model_name)
        self._load_into_cache(model_name)

        from app.services.mlflow_registry import mlflow_registry
        mlflow_registry.register_model(model_name)

        return status

    def get(self, model_name: str) -> Language:
        """
        Returns a loaded spaCy Language object from the in-memory cache.
        Falls back to loading from disk if not cached yet.

        Raises ValueError if the model was never downloaded.
        """
        self._validate(model_name)

        with self._lock:
            if model_name in self._loaded:
                return self._loaded[model_name]
            if model_name in self._deleted:
                raise ValueError(
                    f"Model '{model_name}' was deleted. "
                    f"Call POST /load/ to register it again."
                )

        if not self._is_installed(model_name):
            raise ValueError(
                f"Model '{model_name}' is not available. "
                f"Call POST /load/ first."
            )

        logger.info(f"Loading model '{model_name}' into memory cache...")
        nlp = spacy.load(model_name)

        with self._lock:
            self._loaded[model_name] = nlp

        logger.info(f"Model '{model_name}' loaded and cached.")
        return nlp

    def delete(self, model_name: str) -> None:
        """
        Evicts the model from the in-memory cache and archives
        all its versions in the MLflow registry.

        Adds model_name to self._deleted so get() won't silently reload
        it from disk on the next /predict/ call. The package stays on disk
        (pip uninstall would be too destructive) but is treated as absent
        until ensure_available() is called again.
        """
        self._validate(model_name)

        with self._lock:
            if model_name not in self._loaded and not self._is_installed(model_name):
                raise ValueError(f"Model '{model_name}' not found.")
            self._loaded.pop(model_name, None)
            self._deleted.add(model_name)

        from app.services.mlflow_registry import mlflow_registry
        mlflow_registry.delete_registered_model(model_name)

        logger.info(f"Model '{model_name}' evicted from cache and archived in MLflow.")

    def list_models(self) -> list[dict]:
        """
        Returns model info from MLflow registry enriched with
        the in-memory load status from the local cache.
        """
        from app.services.mlflow_registry import mlflow_registry
        registry_models = mlflow_registry.list_registered_models()

        with self._lock:
            loaded_names = set(self._loaded.keys())

        for m in registry_models:
            m["loaded"] = m["name"] in loaded_names

        return registry_models

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
        """
        Downloads a spaCy model via pip at runtime.

        --user         → installs into $PYTHONUSERBASE (writable by appuser)
        --no-cache-dir → never write to ~/.cache/pip
        The `--` separator tells `spacy download` to forward everything
        after it directly to the underlying pip call.
        """
        env = os.environ.copy()
        env["PIP_NO_CACHE_DIR"] = "1"
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

        result = subprocess.run(
            [
                sys.executable, "-m", "spacy", "download", model_name,
                "--",
                "--user",
                "--no-cache-dir",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to download model '{model_name}':\n{result.stderr}"
            )

    def _refresh_sys_path(self) -> None:
        """
        After `pip install --user` the model package lives in
        $PYTHONUSERBASE/lib/pythonX.Y/site-packages — a directory that
        didn't exist when this interpreter started.

        This method:
          1. Adds the user site-packages dir to sys.path so importlib
             can find the newly installed package.
          2. Invalidates pkg_resources' working set (used internally by
             spacy.util.is_package via importlib.metadata).
          3. Calls importlib.invalidate_caches() to flush the finder cache.
        """
        user_site = site.getusersitepackages()
        if user_site not in sys.path:
            sys.path.insert(0, user_site)
            logger.info(f"Added user site-packages to sys.path: {user_site}")

        try:
            import pkg_resources
            pkg_resources._initialize_master_working_set()
        except Exception as exc:
            logger.debug(f"pkg_resources refresh skipped: {exc}")

        importlib.invalidate_caches()
        logger.debug("importlib caches invalidated.")

    def _load_into_cache(self, model_name: str) -> None:
        """
        Loads a spaCy model into self._loaded.
        Safe to call multiple times — skips if already cached.

        Raises ValueError if spacy.load() fails, so the caller
        (ensure_available) surfaces the error cleanly as an HTTP 500.
        """
        with self._lock:
            if model_name in self._loaded:
                logger.info(f"Model '{model_name}' already in cache — skipping load.")
                return

        logger.info(f"Loading model '{model_name}' into memory cache...")
        try:
            nlp = spacy.load(model_name)
        except Exception as exc:
            raise ValueError(
                f"Model '{model_name}' was downloaded but could not be loaded: {exc}"
            ) from exc

        with self._lock:
            self._loaded[model_name] = nlp

        logger.info(f"Model '{model_name}' cached successfully.")

    def _refresh_cache(self) -> None:
        """Pre-loads already-installed models into memory cache on startup."""
        for name in ALLOWED_MODELS:
            if self._is_installed(name):
                try:
                    self._loaded[name] = spacy.load(name)
                    logger.info(f"Pre-loaded model from disk: '{name}'")
                except Exception as exc:
                    logger.warning(f"Could not pre-load '{name}': {exc}")

model_manager = ModelManager()