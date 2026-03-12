from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator

import mlflow

from src.config.settings import get_settings


@dataclass(frozen=True, slots=True)
class MLflowConfig:
    tracking_uri: str
    experiment_name: str
    enabled: bool = True
    strict: bool = False

    @staticmethod
    def from_params(params: dict[str, Any], stage_params: dict[str, Any] | None = None) -> "MLflowConfig":
        settings = get_settings()
        tracking_cfg = params.get("tracking", {}).get("mlflow", {})
        stage_cfg = stage_params or {}
        return MLflowConfig(
            tracking_uri=str(
                stage_cfg.get(
                    "tracking_uri",
                    tracking_cfg.get("tracking_uri", settings.mlflow_tracking_uri),
                )
            ),
            experiment_name=str(
                stage_cfg.get(
                    "experiment_name",
                    tracking_cfg.get("experiment_name", settings.mlflow_experiment),
                )
            ),
            enabled=bool(stage_cfg.get("enabled", True)),
            strict=bool(stage_cfg.get("strict", False)),
        )


class MLflowTracker:
    """Thin wrapper to keep tracking calls isolated from training code."""

    def __init__(self, config: MLflowConfig) -> None:
        self.config = config
        self._configured = False
        self.last_error: str | None = None

    def _on_error(self, exc: Exception) -> None:
        self.last_error = str(exc)
        if self.config.strict:
            raise exc

    def _configure(self) -> None:
        if not self.config.enabled:
            return
        if self._configured:
            return
        try:
            mlflow.set_tracking_uri(self.config.tracking_uri)
            mlflow.set_experiment(self.config.experiment_name)
        except Exception as exc:
            self._on_error(exc)
            return
        self._configured = True

    @contextmanager
    def run(self, run_name: str, tags: dict[str, str] | None = None, nested: bool = True) -> Iterator[str | None]:
        if not self.config.enabled:
            yield None
            return
        self._configure()
        if not self._configured:
            yield None
            return
        try:
            run_ctx = mlflow.start_run(run_name=run_name, tags=tags or {}, nested=nested)
        except Exception as exc:
            self._on_error(exc)
            yield None
            return
        with run_ctx as active_run:
            yield active_run.info.run_id

    def log_params(self, params: dict[str, Any]) -> None:
        if not self.config.enabled or not params:
            return
        normalized: dict[str, Any] = {}
        for key, value in params.items():
            if isinstance(value, (dict, list)):
                normalized[key] = json.dumps(value, ensure_ascii=True, sort_keys=True)
            else:
                normalized[key] = value
        try:
            mlflow.log_params(normalized)
        except Exception as exc:
            self._on_error(exc)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self.config.enabled and metrics:
            try:
                mlflow.log_metrics(metrics, step=step)
            except Exception as exc:
                self._on_error(exc)

    def log_artifacts(self, local_dir: str, artifact_path: str | None = None) -> None:
        if self.config.enabled:
            try:
                mlflow.log_artifacts(local_dir, artifact_path=artifact_path)
            except Exception as exc:
                self._on_error(exc)

    def log_artifact(self, file_path: str, artifact_path: str | None = None) -> None:
        if self.config.enabled:
            try:
                mlflow.log_artifact(file_path, artifact_path=artifact_path)
            except Exception as exc:
                self._on_error(exc)

    def log_dataset_metadata(
        self,
        *,
        dataset_name: str,
        row_count: int,
        start_ts: str | None = None,
        end_ts: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "dataset_name": dataset_name,
            "row_count": int(row_count),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "metadata": metadata or {},
        }
        self.log_params({"dataset_metadata": payload})

    def log_feature_schema_version(self, *, feature_family: str, schema_version: str) -> None:
        self.log_params({"feature_family": feature_family, "feature_schema_version": schema_version})

    def log_model_artifact_metadata(
        self,
        *,
        model_name: str,
        model_version: str,
        artifact_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.log_params(
            {
                "model_name": model_name,
                "model_version": model_version,
                "model_artifact_path": artifact_path,
                "model_metadata": metadata or {},
            }
        )

    def log_comparison_decision(
        self,
        *,
        candidate_run_id: str,
        champion_run_id: str | None,
        decision: str,
        reason: str,
        metrics: dict[str, float] | None = None,
    ) -> None:
        self.log_params(
            {
                "candidate_run_id": candidate_run_id,
                "champion_run_id": champion_run_id or "",
                "comparison_decision": decision,
                "comparison_reason": reason,
            }
        )
        if metrics:
            self.log_metrics(metrics)


def build_mlflow_tracker(params: dict[str, Any], stage_params: dict[str, Any] | None = None) -> MLflowTracker:
    return MLflowTracker(config=MLflowConfig.from_params(params=params, stage_params=stage_params))


def maybe_log_file_artifact(tracker: MLflowTracker, file_path: Path, artifact_path: str | None = None) -> None:
    if file_path.exists():
        tracker.log_artifact(str(file_path), artifact_path=artifact_path)
