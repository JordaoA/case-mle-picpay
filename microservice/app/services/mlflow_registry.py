"""
services/mlflow_registry.py
-----------------------------
Wraps all MLflow interactions for the NER service.

Responsibilities:
- Create and manage Registered Models in the MLflow Model Registry
- Log model registration events as MLflow Runs (with metadata)
- Transition model versions through stages (Staging → Production)
- Log prediction runs with entity metrics
- Provide a clean interface so model_manager.py stays free of MLflow details

MLflow concepts used here:
    RegisteredModel  → a named model (e.g. "en_core_web_sm") in the registry
    ModelVersion     → a specific version of that model (v1, v2, ...)
    Stage            → lifecycle stage: None | Staging | Production | Archived
    Run              → a tracked execution (used to log registration + prediction events)
    Experiment       → a named group of Runs
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import mlflow
import spacy
from mlflow import MlflowClient
from mlflow.entities.model_registry.model_version_stages import (
    STAGE_PRODUCTION,
    STAGE_STAGING,
)

from app.config import settings

logger = logging.getLogger("ner_service.mlflow_registry")


class MLflowRegistry:
    """
    Thin wrapper around MLflow tracking + model registry APIs.
    All public methods are safe to call from FastAPI route handlers.
    """

    def __init__(self) -> None:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self._client = MlflowClient()
        self._experiment_id = self._ensure_experiment()
        logger.info(
            f"MLflow registry initialized — "
            f"tracking URI: {settings.mlflow_tracking_uri} | "
            f"experiment: '{settings.mlflow_experiment_name}'"
        )

    # ------------------------------------------------------------------
    # Model registration
    # ------------------------------------------------------------------

    def register_model(self, model_name: str) -> dict:
        """
        Registers a spaCy model in the MLflow Model Registry.

        Flow:
            1. Start a new MLflow Run to log metadata
            2. Log model params (name, spaCy version, timestamp)
            3. Register the run as a new ModelVersion
            4. Transition that version to Production
               (demoting any previous Production version to Archived)

        Returns a dict with: name, version, stage, run_id
        """
        self._ensure_registered_model(model_name)

        spacy_version = spacy.__version__
        registered_at = datetime.now(tz=timezone.utc).isoformat()

        with mlflow.start_run(
            experiment_id=self._experiment_id,
            run_name=f"register-{model_name}",
        ) as run:
            # Log metadata about this registration event
            mlflow.log_params({
                "model_name":    model_name,
                "spacy_version": spacy_version,
                "registered_at": registered_at,
            })
            mlflow.set_tags({
                "event":        "model_registration",
                "service":      settings.service_name,
            })

            # Register this run as a versioned model artifact
            # We use a dummy artifact URI since spaCy models live in site-packages,
            # not in the MLflow artifact store. The registry entry tracks metadata only.
            model_uri = f"runs:/{run.info.run_id}/model"

            try:
                mv = mlflow.register_model(
                    model_uri=model_uri,
                    name=model_name,
                )
            except Exception as exc:
                logger.warning(
                    f"Could not register model artifact (expected for spaCy models "
                    f"without a logged artifact): {exc}. Registering metadata only."
                )
                # Create a version directly via the client as a fallback
                mv = self._client.create_model_version(
                    name=model_name,
                    source=f"spacy://{model_name}",
                    run_id=run.info.run_id,
                    description=f"spaCy {model_name} v{spacy_version}",
                )

        # Demote existing Production → Archived, then promote new version
        self._promote_to_production(model_name, mv.version)

        logger.info(
            f"Registered '{model_name}' as version {mv.version} → Production"
        )
        return {
            "name":    model_name,
            "version": mv.version,
            "stage":   STAGE_PRODUCTION,
            "run_id":  run.info.run_id,
        }

    # ------------------------------------------------------------------
    # Model queries
    # ------------------------------------------------------------------

    def get_model_info(self, model_name: str) -> Optional[dict]:
        """
        Returns the latest Production version info for a model.
        Returns None if the model is not registered.
        """
        try:
            versions = self._client.get_latest_versions(
                model_name, stages=[STAGE_PRODUCTION]
            )
            if not versions:
                # Fall back to any stage
                versions = self._client.get_latest_versions(model_name)
            if not versions:
                return None

            v = versions[0]
            return {
                "name":        v.name,
                "version":     v.version,
                "stage":       v.current_stage,
                "run_id":      v.run_id,
                "description": v.description,
                "created_at":  datetime.fromtimestamp(
                    v.creation_timestamp / 1000, tz=timezone.utc
                ).isoformat(),
            }
        except mlflow.exceptions.MlflowException:
            return None

    def list_registered_models(self) -> list[dict]:
        """Lists all models registered in the MLflow Model Registry."""
        try:
            registered = self._client.search_registered_models()
        except Exception as exc:
            logger.warning(f"Could not list MLflow models: {exc}")
            return []

        result = []
        for rm in registered:
            latest = rm.latest_versions
            production = next(
                (v for v in latest if v.current_stage == STAGE_PRODUCTION), None
            )
            active = production or (latest[0] if latest else None)

            result.append({
                "name":    rm.name,
                "version": active.version if active else "—",
                "stage":   active.current_stage if active else "—",
                "run_id":  active.run_id if active else None,
            })
        return result

    def delete_registered_model(self, model_name: str) -> None:
        """
        Archives all versions of a model.
        We archive rather than hard-delete to preserve audit history.
        """
        try:
            versions = self._client.search_model_versions(f"name='{model_name}'")
            for v in versions:
                if v.current_stage != "Archived":
                    self._client.transition_model_version_stage(
                        name=model_name,
                        version=v.version,
                        stage="Archived",
                    )
            logger.info(f"All versions of '{model_name}' archived in MLflow.")
        except mlflow.exceptions.MlflowException as exc:
            raise ValueError(f"Model '{model_name}' not found in registry: {exc}")

    # ------------------------------------------------------------------
    # Prediction logging
    # ------------------------------------------------------------------

    def log_prediction(
        self,
        model_name: str,
        model_version: str,
        input_text: str,
        entities: list[dict],
        latency_ms: float,
    ) -> str:
        """
        Logs a single prediction as an MLflow Run.
        Returns the run_id for traceability.

        Logged metrics:
            - num_entities: how many entities were extracted
            - latency_ms: inference time in milliseconds
            - input_length: character count of the input text

        Logged params:
            - model_name, model_version, entity_labels (comma-separated)
        """
        try:
            with mlflow.start_run(
                experiment_id=self._experiment_id,
                run_name=f"predict-{model_name}",
            ) as run:
                mlflow.log_params({
                    "model_name":    model_name,
                    "model_version": model_version,
                    "entity_labels": ",".join(e["label"] for e in entities) or "none",
                })
                mlflow.log_metrics({
                    "num_entities": len(entities),
                    "latency_ms":   latency_ms,
                    "input_length": len(input_text),
                })
                mlflow.set_tags({
                    "event":   "prediction",
                    "service": settings.service_name,
                })
                return run.info.run_id
        except Exception as exc:
            # Prediction logging should never break the inference response
            logger.warning(f"Failed to log prediction to MLflow: {exc}")
            return ""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_experiment(self) -> str:
        """Creates the MLflow experiment if it doesn't exist yet."""
        experiment = mlflow.get_experiment_by_name(
            settings.mlflow_experiment_name
        )
        if experiment:
            return experiment.experiment_id
        return mlflow.create_experiment(settings.mlflow_experiment_name)

    def _ensure_registered_model(self, model_name: str) -> None:
        """Creates the RegisteredModel entry if it doesn't exist yet."""
        try:
            self._client.get_registered_model(model_name)
        except mlflow.exceptions.MlflowException:
            self._client.create_registered_model(
                name=model_name,
                description=f"spaCy NER model: {model_name}",
                tags={"framework": "spacy", "task": "ner"},
            )
            logger.info(f"Created registered model entry: '{model_name}'")

    def _promote_to_production(self, model_name: str, new_version: str) -> None:
        """
        Demotes any existing Production version to Archived,
        then transitions the new version to Production.
        """
        existing = self._client.get_latest_versions(
            model_name, stages=[STAGE_PRODUCTION]
        )
        for v in existing:
            if v.version != new_version:
                self._client.transition_model_version_stage(
                    name=model_name,
                    version=v.version,
                    stage="Archived",
                )
                logger.info(
                    f"Archived previous Production version "
                    f"'{model_name}' v{v.version}"
                )

        self._client.transition_model_version_stage(
            name=model_name,
            version=new_version,
            stage=STAGE_PRODUCTION,
        )


# Singleton
mlflow_registry = MLflowRegistry()