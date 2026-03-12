from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator


class MarketFeatureRow(BaseModel):
    ticker: str = Field(min_length=1)
    timestamp: datetime
    close: float
    return_1: float
    log_return_1: float
    rolling_volatility_20: float
    momentum_10: float
    rsi_14: float
    macd: float
    macd_signal: float
    atr_14: float
    zscore_20: float
    volume: float
    volume_ratio_20: float
    volume_zscore_20: float
    trend_regime: float
    volatility_regime: float

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("timestamp")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
