from __future__ import annotations

from datetime import datetime, timezone

from src.data.feature_store.io import read_parquet
from src.data.feature_store.paths import merged_train_parquet_path
from src.data.ingestion.news.storage import news_features_parquet_path
from src.domain.schemas import StageResult
from src.models.classification.binary_direction import BinaryDirectionClassifier
from src.models.classification.multiclass_action import MulticlassActionClassifier
from src.models.ensemble.stacking_placeholder import StackingMetaModelPlaceholder
from src.models.ensemble.weighted import WeightedEnsembleModel
from src.models.news.news_feature_model import NewsFeatureModel
from src.models.registry.repository import write_model_metadata
from src.models.regression.gru_skeleton import GRURegressionSkeleton
from src.models.regression.tabular_baseline import TabularRegressionBaseline
from src.training.pipeline.base import PipelineStage, StageContext
from src.training.tracking.mlflow_utils import build_mlflow_tracker


def _prediction_horizon_from_params(params: dict) -> str:
    minutes = int(params.get("models", {}).get("prediction_horizon_minutes", 60))
    return f"{minutes}m"


class TrainBaseModelsStage(PipelineStage):
    def run(self, ctx: StageContext) -> StageResult:
        started = datetime.now(tz=timezone.utc)
        rows = read_parquet(merged_train_parquet_path(ctx.workspace))
        tracker = build_mlflow_tracker(ctx.params, ctx.stage_params)

        tabular = TabularRegressionBaseline()
        gru = GRURegressionSkeleton()
        binary = BinaryDirectionClassifier()
        multi = MulticlassActionClassifier()
        horizon = _prediction_horizon_from_params(ctx.params)
        for model in [tabular, gru, binary, multi]:
            model.prediction_horizon = horizon

        metrics = {}
        with tracker.run(ctx.stage_name, tags={"stage": ctx.stage_name, "run_id": ctx.run_id}, nested=True):
            tracker.log_dataset_metadata(
                dataset_name="merged_train",
                row_count=len(rows),
                metadata={"stage": ctx.stage_name},
            )
            tracker.log_feature_schema_version(
                feature_family="merged",
                schema_version=str(ctx.params.get("features", {}).get("merged", {}).get("schema_version", "v1")),
            )
            for model in [tabular, gru, binary, multi]:
                model_metrics = model.fit(rows, target_key="return_1")
                model_path = ctx.workspace / "models" / "base" / f"{model.model_name}.pkl"
                model.save(model_path)
                metrics.update({f"{model.model_name}_{k}": float(v) for k, v in model_metrics.items()})
                tracker.log_model_artifact_metadata(
                    model_name=model.model_name,
                    model_version=model.model_version,
                    artifact_path=str(model_path.relative_to(ctx.workspace)),
                    metadata=model.get_metadata(),
                )
                tracker.log_artifact(str(model_path), artifact_path="models/base")

            tracker.log_params({"stage_name": ctx.stage_name, "model_count": 4})
            tracker.log_metrics(metrics)

        write_model_metadata(
            ctx.workspace / "models" / "base" / "metadata.json",
            {
                "run_id": ctx.run_id,
                "trained_models": [tabular.model_name, gru.model_name, binary.model_name, multi.model_name],
            },
        )

        finished = datetime.now(tz=timezone.utc)
        return StageResult(
            run_id=ctx.run_id,
            stage_name=ctx.stage_name,
            success=True,
            started_at=started,
            finished_at=finished,
            metrics=metrics | {"train_rows": float(len(rows))},
            artifacts=["models/base"],
        )


class TrainNewsModelStage(PipelineStage):
    def run(self, ctx: StageContext) -> StageResult:
        started = datetime.now(tz=timezone.utc)
        rows = read_parquet(news_features_parquet_path(ctx.workspace))
        tracker = build_mlflow_tracker(ctx.params, ctx.stage_params)
        model = NewsFeatureModel()
        model.prediction_horizon = _prediction_horizon_from_params(ctx.params)

        with tracker.run(ctx.stage_name, tags={"stage": ctx.stage_name, "run_id": ctx.run_id}, nested=True):
            tracker.log_dataset_metadata(
                dataset_name="news_features",
                row_count=len(rows),
                metadata={"stage": ctx.stage_name},
            )
            tracker.log_feature_schema_version(
                feature_family="news",
                schema_version=str(ctx.params.get("features", {}).get("news", {}).get("schema_version", "v1")),
            )
            metrics = model.fit(rows, target_key="sentiment_mean")
            model_path = ctx.workspace / "models" / "news" / f"{model.model_name}.pkl"
            model.save(model_path)
            tracker.log_metrics({f"news_model_{k}": float(v) for k, v in metrics.items()})
            tracker.log_model_artifact_metadata(
                model_name=model.model_name,
                model_version=model.model_version,
                artifact_path=str(model_path.relative_to(ctx.workspace)),
                metadata=model.get_metadata(),
            )
            tracker.log_artifact(str(model_path), artifact_path="models/news")
            tracker.log_params({"stage_name": ctx.stage_name})

        write_model_metadata(
            ctx.workspace / "models" / "news" / "metadata.json",
            {"run_id": ctx.run_id, "trained_model": model.model_name},
        )
        finished = datetime.now(tz=timezone.utc)
        return StageResult(
            run_id=ctx.run_id,
            stage_name=ctx.stage_name,
            success=True,
            started_at=started,
            finished_at=finished,
            metrics={"news_train_rows": float(len(rows))},
            artifacts=["models/news"],
        )


class TrainEnsembleModelStage(PipelineStage):
    def run(self, ctx: StageContext) -> StageResult:
        started = datetime.now(tz=timezone.utc)
        model_cfg = ctx.params.get("models", {}).get("weights", {})
        tracker = build_mlflow_tracker(ctx.params, ctx.stage_params)
        horizon = _prediction_horizon_from_params(ctx.params)

        ensemble = WeightedEnsembleModel(
            weights={
                "tabular_regression_baseline": float(model_cfg.get("tabular_regression", 0.2)),
                "gru_regression_skeleton": float(model_cfg.get("rnn_regression", 0.2)),
                "binary_direction_classifier": float(model_cfg.get("binary_classifier", 0.2)),
                "multiclass_action_classifier": float(model_cfg.get("multiclass_classifier", 0.2)),
                "news_feature_model": float(model_cfg.get("news_model", 0.2)),
            }
        )
        stacking = StackingMetaModelPlaceholder()
        ensemble.prediction_horizon = horizon
        stacking.prediction_horizon = horizon
        ensemble_metrics = ensemble.fit([])
        stacking_metrics = stacking.fit([])

        with tracker.run(ctx.stage_name, tags={"stage": ctx.stage_name, "run_id": ctx.run_id}, nested=True):
            tracker.log_feature_schema_version(
                feature_family="merged",
                schema_version=str(ctx.params.get("features", {}).get("merged", {}).get("schema_version", "v1")),
            )
            ensemble_path = ctx.workspace / "models" / "ensemble" / f"{ensemble.model_name}.pkl"
            stacking_path = ctx.workspace / "models" / "ensemble" / f"{stacking.model_name}.pkl"
            ensemble.save(ensemble_path)
            stacking.save(stacking_path)
            tracker.log_metrics(
                {
                    "ensemble_weight_count": float(ensemble_metrics.get("weight_count", 0.0)),
                    "stacking_placeholder": float(stacking_metrics.get("stacking_placeholder", 1.0)),
                }
            )
            tracker.log_model_artifact_metadata(
                model_name=ensemble.model_name,
                model_version=ensemble.model_version,
                artifact_path=str(ensemble_path.relative_to(ctx.workspace)),
                metadata=ensemble.get_metadata(),
            )
            tracker.log_model_artifact_metadata(
                model_name=stacking.model_name,
                model_version=stacking.model_version,
                artifact_path=str(stacking_path.relative_to(ctx.workspace)),
                metadata=stacking.get_metadata(),
            )
            tracker.log_artifact(str(ensemble_path), artifact_path="models/ensemble")
            tracker.log_artifact(str(stacking_path), artifact_path="models/ensemble")

        write_model_metadata(
            ctx.workspace / "models" / "ensemble" / "metadata.json",
            {"run_id": ctx.run_id, "trained_models": [ensemble.model_name, stacking.model_name]},
        )
        finished = datetime.now(tz=timezone.utc)
        return StageResult(
            run_id=ctx.run_id,
            stage_name=ctx.stage_name,
            success=True,
            started_at=started,
            finished_at=finished,
            metrics={"ensemble_models": 2.0},
            artifacts=["models/ensemble"],
        )
