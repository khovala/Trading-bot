from __future__ import annotations

from datetime import datetime, timezone

from src.data.feature_store.io import read_parquet, write_parquet
from src.data.feature_store.paths import (
    market_features_parquet_path,
    merged_test_parquet_path,
    merged_train_parquet_path,
    merged_validation_parquet_path,
)
from src.data.market_store.layout import interim_candles_path, read_jsonl
from src.data.schemas.features import MarketFeatureRow
from src.domain.schemas import StageResult
from src.features.market.engineering import generate_market_features, parse_market_candle_rows
from src.features.merge.datasets import chronological_split, merge_market_news_features
from src.training.pipeline.base import PipelineStage, StageContext


class GenerateMarketFeaturesStage(PipelineStage):
    def run(self, ctx: StageContext) -> StageResult:
        started = datetime.now(tz=timezone.utc)
        market_cfg = ctx.params.get("data", {}).get("market", {})
        tickers = [str(x).upper() for x in market_cfg.get("instruments", ["SBER"])]

        candles_by_ticker = {}
        for ticker in tickers:
            rows = read_jsonl(interim_candles_path(ctx.workspace, ticker))
            candles_by_ticker[ticker] = parse_market_candle_rows(rows)

        feature_rows = generate_market_features(candles_by_ticker)
        validated_rows = [MarketFeatureRow.model_validate(row).model_dump(mode="json") for row in feature_rows]
        write_parquet(market_features_parquet_path(ctx.workspace), validated_rows)

        finished = datetime.now(tz=timezone.utc)
        return StageResult(
            run_id=ctx.run_id,
            stage_name=ctx.stage_name,
            success=True,
            started_at=started,
            finished_at=finished,
            metrics={
                "ticker_count": float(len(tickers)),
                "market_feature_row_count": float(len(validated_rows)),
            },
            artifacts=["data/processed/market/features/market_features.parquet"],
        )


class MergeFeatureSetsStage(PipelineStage):
    def run(self, ctx: StageContext) -> StageResult:
        started = datetime.now(tz=timezone.utc)
        stage_cfg = ctx.stage_params
        market_rows = read_parquet(market_features_parquet_path(ctx.workspace))
        news_rows = read_parquet(ctx.workspace / "data/processed/news/features/news_features.parquet")
        news_cfg = ctx.params.get("data", {}).get("news", {})
        bucket_minutes = int(stage_cfg.get("bucket_minutes", news_cfg.get("aggregation_bucket_minutes", 60)))
        news_lag_buckets = int(stage_cfg.get("news_lag_buckets", 1))
        merged = merge_market_news_features(
            market_rows,
            news_rows,
            bucket_minutes=bucket_minutes,
            news_lag_buckets=news_lag_buckets,
        )

        train_ratio = float(stage_cfg.get("train_ratio", 0.70))
        val_ratio = float(stage_cfg.get("validation_ratio", 0.15))
        test_ratio = float(stage_cfg.get("test_ratio", 0.15))
        train_rows, val_rows, test_rows = chronological_split(
            merged,
            train_ratio=train_ratio,
            validation_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        write_parquet(merged_train_parquet_path(ctx.workspace), train_rows)
        write_parquet(merged_validation_parquet_path(ctx.workspace), val_rows)
        write_parquet(merged_test_parquet_path(ctx.workspace), test_rows)

        finished = datetime.now(tz=timezone.utc)
        return StageResult(
            run_id=ctx.run_id,
            stage_name=ctx.stage_name,
            success=True,
            started_at=started,
            finished_at=finished,
            metrics={
                "merged_row_count": float(len(merged)),
                "train_row_count": float(len(train_rows)),
                "validation_row_count": float(len(val_rows)),
                "test_row_count": float(len(test_rows)),
                "merge_bucket_minutes": float(bucket_minutes),
                "news_lag_buckets": float(news_lag_buckets),
            },
            artifacts=[
                "data/processed/merged/train.parquet",
                "data/processed/merged/validation.parquet",
                "data/processed/merged/test.parquet",
            ],
        )
