"""
Fake implementation of the MLflow Tracking and Model Registry service.
"""

class FakeMLflowRegistry:
    """Mimics app.services.mlflow_registry.MLflowRegistry."""

    def __init__(self) -> None:
        self._registry: dict[str, dict] = {}
        self._logs: list[dict] = []

    def register_model(self, model_name: str) -> None:
        self._registry[model_name] = {
            "name": model_name,
            "version": "1",
            "stage": "Production",
            "run_id": "fake-run-id",
        }

    def get_model_info(self, model_name: str) -> dict | None:
        return self._registry.get(model_name)

    def delete_registered_model(self, model_name: str) -> None:
        if model_name in self._registry:
            self._registry[model_name]["stage"] = "Archived"

    def list_registered_models(self) -> list[dict]:
        return list(self._registry.values())

    def log_prediction(
        self,
        model_name: str,
        model_version: str,
        input_text: str,
        entities: list,
        latency_ms: float,
    ) -> None:
        """Records telemetry locally instead of making HTTP calls to MLflow."""
        self._logs.append(
            {
                "model": model_name,
                "version": model_version,
                "latency": latency_ms,
            }
        )