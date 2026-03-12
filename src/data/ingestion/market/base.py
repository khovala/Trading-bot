from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from src.data.schemas.market import CandleRecord, InstrumentRecord


class MarketDataSource(ABC):
    @abstractmethod
    def fetch_instruments(self, tickers: list[str]) -> list[InstrumentRecord]:
        raise NotImplementedError

    @abstractmethod
    def fetch_candles(
        self,
        *,
        ticker: str,
        start_at: datetime,
        end_at: datetime,
        interval: str,
    ) -> list[CandleRecord]:
        raise NotImplementedError


class HistoricalCandlesDownloader:
    def __init__(self, source: MarketDataSource) -> None:
        self.source = source

    def download(
        self,
        *,
        ticker: str,
        start_at: datetime,
        end_at: datetime,
        interval: str,
    ) -> list[CandleRecord]:
        return self.source.fetch_candles(
            ticker=ticker,
            start_at=start_at,
            end_at=end_at,
            interval=interval,
        )


class InstrumentMetadataLoader:
    def __init__(self, source: MarketDataSource) -> None:
        self.source = source

    def load(self, tickers: list[str]) -> list[InstrumentRecord]:
        return self.source.fetch_instruments(tickers)
