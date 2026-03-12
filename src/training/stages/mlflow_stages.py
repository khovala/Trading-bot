from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.domain.schemas import StageResult
from src.training.pipeline.base import PipelineStage, StageContext
from src.training.tracking.mlflow_utils import build_mlflow_tracker, maybe_log_file_artifact


class _MLflowLoggingStage(PipelineStage):
    model_family: str = "generic"
    report_name: str = ""

    def _stage_metrics(self) -> dict[str, float]:
        return {"placeholder": 1.0}

    def _dataset_name(self) -> str:
        return "unknown_dataset"

    def _feature_family(self) -> str:
        return "unknown"

    def _feature_schema_version(self, ctx: StageContext) -> str:
        family = self._feature_family()
        return str(ctx.params.get("features", {}).get(family, {}).get("schema_version", "v1"))

    def _after_log(self, tracker, run_id: str | None, ctx: StageContext) -> None:
        _ = (tracker, run_id, ctx)

    def run(self, ctx: StageContext) -> StageResult:
        started = datetime.now(tz=timezone.utc)
        tracker = build_mlflow_tracker(ctx.params, ctx.stage_params)
        report_name = self.report_name or f"{ctx.stage_name}.json"

        metrics = self._stage_metrics()
        with tracker.run(
            run_name=ctx.stage_name,
            tags={
                "stage": ctx.stage_name,
                "pipeline_run_id": ctx.run_id,
                "model_family": self.model_family,
            },
            nested=True,
        ) as mlflow_run_id:
            tracker.log_params(
                {
                    "stage_name": ctx.stage_name,
                    "model_family": self.model_family,
                    "tracking_enabled": tracker.config.enabled,
                }
            )
            tracker.log_dataset_metadata(
                dataset_name=self._dataset_name(),
                row_count=0,
                metadata={"stage": ctx.stage_name},
            )
            tracker.log_feature_schema_version(
                feature_family=self._feature_family(),
                schema_version=self._feature_schema_version(ctx),
            )
            tracker.log_metrics(metrics)
            maybe_log_file_artifact(tracker, Path(ctx.workspace) / "reports" / report_name, artifact_path="reports")
            self._after_log(tracker, mlflow_run_id, ctx)
            if tracker.last_error:
                metrics["mlflow_error"] = 1.0
                tracker.log_metrics({"mlflow_error": 1.0})

        finished = datetime.now(tz=timezone.utc)
        return StageResult(
            run_id=ctx.run_id,
            stage_name=ctx.stage_name,
            success=True,
            started_at=started,
            finished_at=finished,
            metrics=metrics,
            artifacts=[f"reports/{report_name}"],
        )


class TrainBaseModelsStage(_MLflowLoggingStage):
    model_family = "base_models"

    def _dataset_name(self) -> str:
        return "merged_train_dataset"

    def _feature_family(self) -> str:
        return "market"

    def _after_log(self, tracker, run_id: str | None, ctx: StageContext) -> None:
        tracker.log_model_artifact_metadata(
            model_name="base_models",
            model_version="candidate",
            artifact_path="models/base",
            metadata={"run_id": run_id or "", "stage": ctx.stage_name},
        )


class TrainNewsModelStage(_MLflowLoggingStage):
    model_family = "news_model"

    def _dataset_name(self) -> str:
        return "news_feature_dataset"

    def _feature_family(self) -> str:
        return "news"

    def _after_log(self, tracker, run_id: str | None, ctx: StageContext) -> None:
        tracker.log_model_artifact_metadata(
            model_name="news_model",
            model_version="candidate",
            artifact_path="models/news",
            metadata={"run_id": run_id or "", "stage": ctx.stage_name},
        )


class TrainEnsembleModelStage(_MLflowLoggingStage):
    model_family = "ensemble_model"

    def _dataset_name(self) -> str:
        return "merged_train_dataset"

    def _feature_family(self) -> str:
        return "merged"

    def _after_log(self, tracker, run_id: str | None, ctx: StageContext) -> None:
        tracker.log_model_artifact_metadata(
            model_name="ensemble_model",
            model_version="candidate",
            artifact_path="models/ensemble",
            metadata={"run_id": run_id or "", "stage": ctx.stage_name},
        )


class EvaluateModelsStage(_MLflowLoggingStage):
    model_family = "evaluation"

    def _dataset_name(self) -> str:
        return "validation_dataset"

    def _feature_family(self) -> str:
        return "merged"


class BacktestStrategyStage(_MLflowLoggingStage):
    model_family = "backtest"

    def _dataset_name(self) -> str:
        return "backtest_dataset"

    def _feature_family(self) -> str:
        return "merged"


class CompareWithProductionStage(_MLflowLoggingStage):
    model_family = "comparison"

    def _dataset_name(self) -> str:
        return "comparison_inputs"

    def _feature_family(self) -> str:
        return "merged"

    def _after_log(self, tracker, run_id: str | None, ctx: StageContext) -> None:
        _ = run_id
        tracker.log_comparison_decision(
            candidate_run_id=ctx.run_id,
            champion_run_id=None,
            decision="candidate",
            reason="placeholder comparison stage",
            metrics={"comparison_score_delta": 0.0},
        )


class PromoteModelStage(_MLflowLoggingStage):
    model_family = "promotion"

    def _dataset_name(self) -> str:
        return "promotion_inputs"

    def _feature_family(self) -> str:
        return "merged"

    def _after_log(self, tracker, run_id: str | None, ctx: StageContext) -> None:
        tracker.log_model_artifact_metadata(
            model_name="champion_model",
            model_version="promoted",
            artifact_path="models/registry",
            metadata={"run_id": run_id or "", "stage": ctx.stage_name, "status": "promoted_placeholder"},
        )
