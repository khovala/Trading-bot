from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


REQUIRED_MARKET_FIELDS: tuple[str, ...] = (
    "ticker",
    "timestamp",
    "close",
    "return_1",
    "log_return_1",
    "rolling_volatility_20",
    "momentum_10",
    "volume",
)

REQUIRED_NEWS_FIELDS: tuple[str, ...] = (
    "news_article_count",
    "news_sentiment_mean",
    "news_weighted_sentiment_mean",
)


class FeatureStoreRow(BaseModel):
    ticker: str = Field(min_length=1)
    timestamp: datetime
    close: float
    return_1: float
    log_return_1: float
    rolling_volatility_20: float
    momentum_10: float
    volume: float
    news_article_count: float = 0.0
    news_sentiment_mean: float = 0.0
    news_weighted_sentiment_mean: float = 0.0
    split: str | None = None

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()


def validate_feature_row(row: dict[str, Any], *, require_news: bool = False) -> FeatureStoreRow:
    for key in REQUIRED_MARKET_FIELDS:
        if key not in row:
            raise ValueError(f"Missing required market field: {key}")
    if require_news:
        for key in REQUIRED_NEWS_FIELDS:
            if key not in row:
                raise ValueError(f"Missing required news field: {key}")
    return FeatureStoreRow.model_validate(row)
