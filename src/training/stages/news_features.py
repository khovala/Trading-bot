from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone

from src.data.ingestion.news.storage import (
    interim_news_items_path,
    mapped_news_parquet_path,
    news_features_parquet_path,
    read_jsonl,
    read_parquet,
    write_parquet,
)
from src.data.schemas.news import ProcessedNewsItemRecord
from src.domain.schemas import StageResult
from src.features.news.aggregation import generate_news_features
from src.nlp.entity_mapping.dictionary_mapper import DictionaryIssuerTickerMapper
from src.training.pipeline.base import PipelineStage, StageContext


class MapNewsToInstrumentsStage(PipelineStage):
    def run(self, ctx: StageContext) -> StageResult:
        started = datetime.now(tz=timezone.utc)
        rows = read_jsonl(interim_news_items_path(ctx.workspace))
        items = [ProcessedNewsItemRecord.model_validate(row) for row in rows]
        mapper = DictionaryIssuerTickerMapper()
        mapped = [asdict(mapper.map_item(item)) for item in items]
        write_parquet(mapped_news_parquet_path(ctx.workspace), mapped)

        unmapped = sum(1 for row in mapped if row["ticker"] == "UNKNOWN")
        finished = datetime.now(tz=timezone.utc)
        return StageResult(
            run_id=ctx.run_id,
            stage_name=ctx.stage_name,
            success=True,
            started_at=started,
            finished_at=finished,
            metrics={
                "input_item_count": float(len(items)),
                "mapped_item_count": float(len(mapped)),
                "unmapped_item_count": float(unmapped),
                "mapping_success_ratio": float((len(mapped) - unmapped) / len(mapped)) if mapped else 0.0,
            },
            artifacts=["data/processed/news/mapped_news.parquet"],
        )


class GenerateNewsFeaturesStage(PipelineStage):
    def run(self, ctx: StageContext) -> StageResult:
        started = datetime.now(tz=timezone.utc)
        rows = read_parquet(mapped_news_parquet_path(ctx.workspace))
        news_cfg = ctx.params.get("data", {}).get("news", {})
        features_cfg = ctx.params.get("features", {}).get("news", {})
        stage_cfg = ctx.stage_params

        bucket_minutes = int(stage_cfg.get("bucket_minutes", news_cfg.get("aggregation_bucket_minutes", 60)))
        source_weights = dict(stage_cfg.get("source_weights", {}))
        half_life = int(features_cfg.get("recency_half_life_minutes", 180))

        feature_rows = generate_news_features(
            rows,
            bucket_minutes=bucket_minutes,
            source_weights=source_weights,
            recency_half_life_minutes=half_life,
        )
        write_parquet(news_features_parquet_path(ctx.workspace), feature_rows)

        finished = datetime.now(tz=timezone.utc)
        return StageResult(
            run_id=ctx.run_id,
            stage_name=ctx.stage_name,
            success=True,
            started_at=started,
            finished_at=finished,
            metrics={
                "mapped_row_count": float(len(rows)),
                "feature_row_count": float(len(feature_rows)),
                "bucket_minutes": float(bucket_minutes),
            },
            artifacts=["data/processed/news/features/news_features.parquet"],
        )
