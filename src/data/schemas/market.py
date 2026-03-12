from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class InstrumentRecord(BaseModel):
    ticker: str = Field(min_length=1)
    name: str = Field(min_length=1)
    exchange: Literal["MOEX"] = "MOEX"
    currency: str = Field(default="RUB", min_length=3, max_length=3)
    lot: int = Field(default=1, ge=1)
    figi: str | None = None
    isin: str | None = None
    is_tradable: bool = True

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()


class CandleRecord(BaseModel):
    ticker: str = Field(min_length=1)
    timestamp: datetime
    interval: str = "1min"
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: int = Field(ge=0)
    source: str = "unknown"

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("timestamp")
    @classmethod
    def ensure_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_ohlc(self) -> "CandleRecord":
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low is inconsistent with OHLC values")
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high is inconsistent with OHLC values")
        return self
