from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from src.data.feature_store.io import read_parquet
from src.data.feature_store.paths import merged_train_parquet_path
from src.data.ingestion.news.storage import news_features_parquet_path
from src.domain.schemas import StageResult
from src.models.classification.binary_direction import BinaryDirectionClassifier
from src.models.classification.multiclass_action import MulticlassActionClassifier
from src.models.ensemble.stacking_placeholder import StackingMetaModelPlaceholder
from src.models.ensemble.weighted import WeightedEnsembleModel
from src.models.news.news_feature_model import NewsFeatureModel
from src.models.base.serialization import load_pickle
from src.models.registry.repository import write_model_metadata
from src.models.regression.gru_skeleton import GRURegressionSkeleton
from src.models.regression.tabular_baseline import TabularRegressionBaseline
from src.training.pipeline.base import PipelineStage, StageContext
from src.training.tracking.mlflow_utils import build_mlflow_tracker


def _prediction_horizon_from_params(params: dict) -> str:
    minutes = int(params.get("models", {}).get("prediction_horizon_minutes", 60))
    return f"{minutes}m"


def _load_model_if_exists(path: Path):
    if not path.exists():
        return None
    return load_pickle(path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _model_quality_proxy(rows: list[dict], preds: list) -> dict[str, float]:
    n = min(len(rows), len(preds))
    if n <= 0:
        return {"directional_accuracy": 0.0, "pnl_proxy": 0.0}
    hits = 0
    pnl_proxy = 0.0
    for row, pred in zip(rows[:n], preds[:n]):
        true_ret = float(row.get("return_1", 0.0))
        pred_ret = float(getattr(pred, "expected_return", 0.0))
        if (pred_ret >= 0.0) == (true_ret >= 0.0):
            hits += 1
        pnl_proxy += (1.0 if pred_ret > 0.0 else (-1.0 if pred_ret < 0.0 else 0.0)) * true_ret
    return {"directional_accuracy": float(hits / n), "pnl_proxy": float(pnl_proxy)}


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
        model_v2_cfg = ctx.params.get("models_v2", {}).get("ensemble", {})
        tracker = build_mlflow_tracker(ctx.params, ctx.stage_params)
        horizon = _prediction_horizon_from_params(ctx.params)
        train_rows = read_parquet(merged_train_parquet_path(ctx.workspace))
        max_fit_rows = int(model_v2_cfg.get("max_fit_rows", 120000))
        if max_fit_rows > 0 and len(train_rows) > max_fit_rows:
            train_rows = train_rows[-max_fit_rows:]

        ensemble = WeightedEnsembleModel(
            weights={
                "tabular_regression_baseline": float(model_cfg.get("tabular_regression", 0.2)),
                "gru_regression_skeleton": float(model_cfg.get("rnn_regression", 0.2)),
                "binary_direction_classifier": float(model_cfg.get("binary_classifier", 0.2)),
                "multiclass_action_classifier": float(model_cfg.get("multiclass_classifier", 0.2)),
                "news_feature_model": float(model_cfg.get("news_model", 0.2)),
            },
            uncertainty_penalty=float(model_v2_cfg.get("uncertainty_penalty", 0.25)),
            turnover_penalty=float(model_v2_cfg.get("turnover_penalty", 0.10)),
            min_weight=float(model_v2_cfg.get("min_weight", 0.02)),
        )
        stacking = StackingMetaModelPlaceholder()
        ensemble.prediction_horizon = horizon
        stacking.prediction_horizon = horizon

        component_predictions = {}
        model_candidates = {
            "tabular_regression_baseline": ctx.workspace / "models/base/tabular_regression_baseline.pkl",
            "gru_regression_skeleton": ctx.workspace / "models/base/gru_regression_skeleton.pkl",
            "binary_direction_classifier": ctx.workspace / "models/base/binary_direction_classifier.pkl",
            "multiclass_action_classifier": ctx.workspace / "models/base/multiclass_action_classifier.pkl",
            "news_feature_model": ctx.workspace / "models/news/news_feature_model.pkl",
        }
        for pkl in (ctx.workspace / "models/foundation").glob("*.pkl"):
            model_candidates[pkl.stem] = pkl

        for model_name, model_path in model_candidates.items():
            model = _load_model_if_exists(model_path)
            if model is None:
                continue
            preds = model.predict(train_rows)
            if preds:
                component_predictions[model_name] = preds
                if model_name not in ensemble.weights:
                    ensemble.weights[model_name] = float(model_v2_cfg.get("foundation_default_weight", 0.05))

        selection_enabled = bool(model_v2_cfg.get("selection_enabled", True))
        top_k_models = max(1, int(model_v2_cfg.get("top_k_models", 3)))
        require_positive_pnl_proxy = bool(model_v2_cfg.get("require_positive_pnl_proxy", True))
        min_directional_accuracy = float(model_v2_cfg.get("min_directional_accuracy", 0.50))
        quality_by_model = {name: _model_quality_proxy(train_rows, preds) for name, preds in component_predictions.items()}
        selected_models = sorted(
            component_predictions.keys(),
            key=lambda m: (
                float(quality_by_model.get(m, {}).get("pnl_proxy", 0.0)),
                float(quality_by_model.get(m, {}).get("directional_accuracy", 0.0)),
            ),
            reverse=True,
        )
        if selection_enabled and selected_models:
            filtered = [
                m
                for m in selected_models
                if (
                    (not require_positive_pnl_proxy or float(quality_by_model.get(m, {}).get("pnl_proxy", 0.0)) > 0.0)
                    and float(quality_by_model.get(m, {}).get("directional_accuracy", 0.0)) >= min_directional_accuracy
                )
            ]
            if not filtered:
                filtered = selected_models
            selected_models = filtered[:top_k_models]
            component_predictions = {m: component_predictions[m] for m in selected_models}
            ensemble.weights = {m: w for m, w in ensemble.weights.items() if m in selected_models}

        ensemble_metrics = ensemble.fit(train_rows, predictions_by_model=component_predictions)
        stacking_metrics = stacking.fit([])

        ensemble_predictions = ensemble.combine(component_predictions) if component_predictions else []
        full_abs = (
            sum(abs(float(p.expected_return)) for p in ensemble_predictions) / len(ensemble_predictions)
            if ensemble_predictions
            else 0.0
        )
        ablation: dict[str, dict[str, float]] = {}
        for model_name in component_predictions:
            reduced = {k: v for k, v in component_predictions.items() if k != model_name}
            reduced_preds = ensemble.combine(reduced) if reduced else []
            reduced_abs = (
                sum(abs(float(p.expected_return)) for p in reduced_preds) / len(reduced_preds)
                if reduced_preds
                else 0.0
            )
            ablation[model_name] = {
                "mean_abs_expected_return_delta": float(full_abs - reduced_abs),
                "ensemble_weight": float(ensemble.weights.get(model_name, 0.0)),
            }
        ablation_path = ctx.workspace / "artifacts" / "evaluation" / "ensemble_ablation.json"
        _write_json(
            ablation_path,
            {
                "stage": ctx.stage_name,
                "model_count": len(component_predictions),
                "ensemble_weights": ensemble.weights,
                "ensemble_diagnostics": ensemble.diagnostics,
                "ablation": ablation,
            },
        )

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
                    "ensemble_component_predictions": float(len(component_predictions)),
                    "ensemble_fit_rows_used": float(len(train_rows)),
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
            tracker.log_artifact(str(ablation_path), artifact_path="artifacts/evaluation")

        write_model_metadata(
            ctx.workspace / "models" / "ensemble" / "metadata.json",
            {
                "run_id": ctx.run_id,
                "trained_models": [ensemble.model_name, stacking.model_name],
                "ensemble_component_models": sorted(component_predictions.keys()),
                "quality_proxy_by_model": quality_by_model,
                "ensemble_selection": {
                    "selection_enabled": selection_enabled,
                    "top_k_models": top_k_models,
                    "require_positive_pnl_proxy": require_positive_pnl_proxy,
                    "min_directional_accuracy": min_directional_accuracy,
                    "selected_models": selected_models,
                },
                "adaptive_weighting": {
                    "uncertainty_penalty": ensemble.uncertainty_penalty,
                    "turnover_penalty": ensemble.turnover_penalty,
                    "min_weight": ensemble.min_weight,
                    "max_fit_rows": max_fit_rows,
                },
            },
        )
        finished = datetime.now(tz=timezone.utc)
        return StageResult(
            run_id=ctx.run_id,
            stage_name=ctx.stage_name,
            success=True,
            started_at=started,
            finished_at=finished,
            metrics={
                "ensemble_models": 2.0,
                "ensemble_component_models": float(len(component_predictions)),
                "ensemble_adaptive_weight_count": float(len(ensemble.weights)),
                "ensemble_fit_rows_used": float(len(train_rows)),
            },
            artifacts=["models/ensemble", "artifacts/evaluation/ensemble_ablation.json"],
        )
