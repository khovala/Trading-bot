from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.data.ingestion.market.adapters import build_market_data_source
from src.data.ingestion.market.base import HistoricalCandlesDownloader, InstrumentMetadataLoader
from src.data.market_store.layout import (
    interim_candles_path,
    raw_candles_path,
    raw_instruments_path,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from src.data.preprocessing.market.processor import preprocess_market_dataset
from src.domain.schemas import StageResult
from src.training.pipeline.base import PipelineStage, StageContext


class DownloadMarketDataStage(PipelineStage):
    def run(self, ctx: StageContext) -> StageResult:
        now = datetime.now(tz=timezone.utc)
        market_cfg = ctx.params.get("data", {}).get("market", {})
        stage_cfg = ctx.stage_params
        tickers = [str(x).upper() for x in market_cfg.get("instruments", ["SBER"])]
        interval = str(market_cfg.get("interval", "1min"))
        lookback_days = int(market_cfg.get("lookback_days", 30))
        provider = str(stage_cfg.get("provider", market_cfg.get("provider", "mock")))
        seed = int(ctx.params.get("global", {}).get("seed", 42))

        source = build_market_data_source(provider=provider, seed=seed)
        instrument_loader = InstrumentMetadataLoader(source)
        candle_downloader = HistoricalCandlesDownloader(source)

        instruments = instrument_loader.load(tickers)
        write_json(raw_instruments_path(ctx.workspace), [row.model_dump(mode="json") for row in instruments])

        end_at = now
        start_at = end_at - timedelta(days=lookback_days)
        total_candles = 0
        for ticker in tickers:
            candles = candle_downloader.download(
                ticker=ticker,
                start_at=start_at,
                end_at=end_at,
                interval=interval,
            )
            total_candles += len(candles)
            write_jsonl(raw_candles_path(ctx.workspace, ticker), [row.model_dump(mode="json") for row in candles])

        finished = datetime.now(tz=timezone.utc)
        return StageResult(
            run_id=ctx.run_id,
            stage_name=ctx.stage_name,
            success=True,
            started_at=now,
            finished_at=finished,
            metrics={
                "instrument_count": float(len(instruments)),
                "ticker_count": float(len(tickers)),
                "candle_count": float(total_candles),
            },
            artifacts=["data/raw/market/instruments.json", "data/raw/market/candles"],
        )


class PreprocessMarketDataStage(PipelineStage):
    def run(self, ctx: StageContext) -> StageResult:
        started = datetime.now(tz=timezone.utc)
        market_cfg = ctx.params.get("data", {}).get("market", {})
        stage_cfg = ctx.stage_params
        tickers = [str(x).upper() for x in market_cfg.get("instruments", ["SBER"])]
        interval = str(market_cfg.get("interval", "1min"))
        interval_minutes = int(stage_cfg.get("interval_minutes", 1 if interval == "1min" else 5))

        instrument_rows = read_json(raw_instruments_path(ctx.workspace))
        candles_by_ticker: dict[str, list[dict]] = {}
        for ticker in tickers:
            candles_by_ticker[ticker] = read_jsonl(raw_candles_path(ctx.workspace, ticker))

        instruments, normalized_by_ticker, metrics = preprocess_market_dataset(
            instrument_rows=instrument_rows,
            candles_by_ticker=candles_by_ticker,
            interval_minutes=interval_minutes,
        )

        for ticker, rows in normalized_by_ticker.items():
            write_jsonl(interim_candles_path(ctx.workspace, ticker), [row.model_dump(mode="json") for row in rows])
        write_json(
            ctx.workspace / "data/interim/market/instruments.json",
            [row.model_dump(mode="json") for row in instruments],
        )

        finished = datetime.now(tz=timezone.utc)
        return StageResult(
            run_id=ctx.run_id,
            stage_name=ctx.stage_name,
            success=True,
            started_at=started,
            finished_at=finished,
            metrics=metrics,
            artifacts=[
                "data/interim/market/instruments.json",
                "data/interim/market/candles",
            ],
        )
