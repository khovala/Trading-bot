from __future__ import annotations

from datetime import datetime, timedelta, timezone
from random import Random

from src.data.ingestion.market.base import MarketDataSource
from src.data.schemas.market import CandleRecord, InstrumentRecord


class MockMarketDataSource(MarketDataSource):
    """Offline deterministic source for trainable datasets and tests."""

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def fetch_instruments(self, tickers: list[str]) -> list[InstrumentRecord]:
        records: list[InstrumentRecord] = []
        for ticker in tickers:
            records.append(
                InstrumentRecord(
                    ticker=ticker,
                    name=f"{ticker.upper()} PJSC",
                    lot=10 if ticker.upper() == "SBER" else 1,
                    figi=f"FIGI-{ticker.upper()}",
                    isin=f"ISIN-{ticker.upper()}",
                )
            )
        return records

    def fetch_candles(
        self,
        *,
        ticker: str,
        start_at: datetime,
        end_at: datetime,
        interval: str,
    ) -> list[CandleRecord]:
        step = timedelta(minutes=1 if interval == "1min" else 5)
        start = start_at.astimezone(timezone.utc)
        end = end_at.astimezone(timezone.utc)
        rng = Random(self._seed + sum(ord(ch) for ch in ticker.upper()))
        records: list[CandleRecord] = []

        price = 100.0 + (rng.random() * 10.0)
        ts = start
        while ts <= end:
            drift = (rng.random() - 0.5) * 0.4
            next_price = max(1.0, price + drift)
            high = max(price, next_price) + abs(rng.random() * 0.2)
            low = min(price, next_price) - abs(rng.random() * 0.2)
            records.append(
                CandleRecord(
                    ticker=ticker,
                    timestamp=ts,
                    interval=interval,
                    open=round(price, 4),
                    high=round(high, 4),
                    low=round(max(0.01, low), 4),
                    close=round(next_price, 4),
                    volume=int(500 + rng.random() * 2000),
                    source="mock_market_source",
                )
            )
            price = next_price
            ts += step
        return records


def build_market_data_source(provider: str, seed: int = 42) -> MarketDataSource:
    # Real integrations stay behind this factory boundary in future phases.
    if provider in {"mock", "t_invest_sandbox"}:
        return MockMarketDataSource(seed=seed)
    raise ValueError(f"Unsupported market provider: {provider}")
