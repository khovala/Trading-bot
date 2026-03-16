from __future__ import annotations

from datetime import datetime, timezone

from src.data.feature_store.io import read_parquet
from src.data.feature_store.paths import merged_train_parquet_path
from src.data.ingestion.news.storage import (
    interim_news_items_path,
    read_jsonl,
    write_parquet,
)
from src.domain.schemas import StageResult
from src.models.foundation.registry import build_forecasters_from_params
from src.models.news.news_encoder_pipeline import NewsEncoderPipeline
from src.models.registry.repository import write_model_metadata
from src.training.pipeline.base import PipelineStage, StageContext
from src.training.tracking.mlflow_utils import build_mlflow_tracker


class TrainFoundationModelsStage(PipelineStage):
    def run(self, ctx: StageContext) -> StageResult:
        started = datetime.now(tz=timezone.utc)
        rows = read_parquet(merged_train_parquet_path(ctx.workspace))
        tracker = build_mlflow_tracker(ctx.params, ctx.stage_params)
        forecasters = build_forecasters_from_params(ctx.params)
        metrics: dict[str, float] = {"train_rows": float(len(rows)), "foundation_model_count": float(len(forecasters))}

        with tracker.run(ctx.stage_name, tags={"stage": ctx.stage_name, "run_id": ctx.run_id}, nested=True):
            tracker.log_dataset_metadata(
                dataset_name="merged_train",
                row_count=len(rows),
                metadata={"stage": ctx.stage_name},
            )
            for model in forecasters:
                fit_metrics = model.fit(rows, target_key="return_1")
                model_path = ctx.workspace / "models" / "foundation" / f"{model.model_name}.pkl"
                model.save(model_path)
                tracker.log_model_artifact_metadata(
                    model_name=model.model_name,
                    model_version=model.model_version,
                    artifact_path=str(model_path.relative_to(ctx.workspace)),
                    metadata=model.get_metadata(),
                )
                tracker.log_artifact(str(model_path), artifact_path="models/foundation")
                for key, value in fit_metrics.items():
                    metrics[f"{model.model_name}_{key}"] = float(value)
            tracker.log_params({"stage_name": ctx.stage_name, "enabled_models": len(forecasters)})
            tracker.log_metrics(metrics)

        write_model_metadata(
            ctx.workspace / "models" / "foundation" / "metadata.json",
            {
                "run_id": ctx.run_id,
                "trained_models": [m.model_name for m in forecasters],
            },
        )

        finished = datetime.now(tz=timezone.utc)
        return StageResult(
            run_id=ctx.run_id,
            stage_name=ctx.stage_name,
            success=True,
            started_at=started,
            finished_at=finished,
            metrics=metrics,
            artifacts=["models/foundation"],
        )


class TrainNewsEncoderStage(PipelineStage):
    def run(self, ctx: StageContext) -> StageResult:
        started = datetime.now(tz=timezone.utc)
        news_cfg = ctx.params.get("models_v2", {}).get("news_encoder", {})
        rows = read_jsonl(interim_news_items_path(ctx.workspace))
        tracker = build_mlflow_tracker(ctx.params, ctx.stage_params)
        pipeline = NewsEncoderPipeline(
            encoder_name=str(news_cfg.get("encoder_name", "hashing_tfidf_v1")),
            embedding_dim=int(news_cfg.get("embedding_dim", 16)),
            half_life_minutes=float(news_cfg.get("half_life_minutes", 180.0)),
            min_tokens=int(news_cfg.get("min_tokens", 3)),
            relevance_threshold=float(news_cfg.get("relevance_threshold", 0.01)),
        )
        fit_metrics, encoded_rows = pipeline.fit_transform(rows)
        out_path = ctx.workspace / "data/processed/news/features/news_embeddings.parquet"
        write_parquet(out_path, encoded_rows)

        with tracker.run(ctx.stage_name, tags={"stage": ctx.stage_name, "run_id": ctx.run_id}, nested=True):
            tracker.log_dataset_metadata(dataset_name="interim_news_items", row_count=len(rows), metadata={"stage": ctx.stage_name})
            tracker.log_metrics({f"news_encoder_{k}": float(v) for k, v in fit_metrics.items()})
            tracker.log_params(
                {
                    "stage_name": ctx.stage_name,
                    "encoder_name": pipeline.encoder_name,
                    "embedding_dim": pipeline.embedding_dim,
                }
            )
            tracker.log_artifact(str(out_path), artifact_path="data/processed/news/features")

        write_model_metadata(
            ctx.workspace / "models" / "news_v2" / "metadata.json",
            {
                "run_id": ctx.run_id,
                "encoder": pipeline.get_metadata(),
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
                "encoded_news_rows": float(len(encoded_rows)),
                "news_input_rows": float(len(rows)),
            },
            artifacts=[
                "data/processed/news/features/news_embeddings.parquet",
                "models/news_v2/metadata.json",
            ],
        )
