"""Experiment tracking adapters."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from qts.core.errors import ConfigError


class NullTracker:
    @contextmanager
    def start_run(self, *, run_name: str, nested: bool = False, tags=None):
        yield run_name

    def log_params(self, params: dict[str, Any]) -> None:
        return None

    def log_metrics(self, metrics: dict[str, float]) -> None:
        return None

    def log_artifact(self, path: str) -> None:
        return None


class MLflowTracker:
    def __init__(
        self,
        *,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
    ) -> None:
        try:
            import mlflow
        except ImportError as exc:
            raise ConfigError("MLflow tracking requires the tracking optional dependency") from exc
        self._mlflow = mlflow
        if tracking_uri:
            self._mlflow.set_tracking_uri(tracking_uri)
        if experiment_name:
            self._mlflow.set_experiment(experiment_name)

    def start_run(self, *, run_name: str, nested: bool = False, tags=None):
        return self._mlflow.start_run(run_name=run_name, nested=nested, tags=tags)

    def log_params(self, params: dict[str, Any]) -> None:
        safe_params = {key: _stringify_param(value) for key, value in params.items()}
        if safe_params:
            self._mlflow.log_params(safe_params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        safe_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, int | float)
        }
        if safe_metrics:
            self._mlflow.log_metrics(safe_metrics)

    def log_artifact(self, path: str) -> None:
        self._mlflow.log_artifact(path)


def create_tracker(config: dict[str, Any] | None):
    if not config or not bool(config.get("enabled", False)):
        return NullTracker()
    tracker_type = str(config.get("type", "mlflow")).lower()
    if tracker_type != "mlflow":
        raise ConfigError(f"Unknown tracking type: {tracker_type}")
    return MLflowTracker(
        tracking_uri=config.get("tracking_uri") or config.get("uri"),
        experiment_name=config.get("experiment_name"),
    )


def _stringify_param(value: Any) -> str | int | float | bool:
    if isinstance(value, str | int | float | bool):
        return value
    return repr(value)
