from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.data.schemas.market import CandleRecord
from src.features.market.engineering import generate_market_features
from src.features.merge.datasets import chronological_split, merge_market_news_features


def _candles(count: int) -> list[CandleRecord]:
    base = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    rows: list[CandleRecord] = []
    price = 100.0
    for i in range(count):
        price += 0.2
        rows.append(
            CandleRecord(
                ticker="SBER",
                timestamp=base + timedelta(minutes=i),
                interval="1min",
                open=price - 0.1,
                high=price + 0.2,
                low=price - 0.3,
                close=price,
                volume=1000 + i,
                source="test",
            )
        )
    return rows


def test_market_feature_calculation_contains_required_columns() -> None:
    rows = generate_market_features({"SBER": _candles(40)})
    assert len(rows) == 40
    sample = rows[-1]
    required = {
        "return_1",
        "log_return_1",
        "rolling_volatility_20",
        "momentum_10",
        "rsi_14",
        "macd",
        "macd_signal",
        "atr_14",
        "zscore_20",
        "volume_ratio_20",
        "trend_regime",
        "volatility_regime",
    }
    assert required.issubset(sample.keys())


def test_merge_and_chronological_split_preserve_order_and_no_leakage() -> None:
    market_rows = [
        {"ticker": "SBER", "timestamp": f"2026-01-01T10:0{i}:00+00:00", "close": 100 + i}
        for i in range(6)
    ]
    news_rows = [
        {
            "ticker": "SBER",
            "timestamp_bucket": f"2026-01-01T10:0{i}:00+00:00",
            "article_count": 1.0,
            "positive_count": 1.0,
            "negative_count": 0.0,
            "sentiment_mean": 1.0,
            "weighted_sentiment_mean": 1.0,
            "abnormal_news_volume": 0.0,
            "recency_weighted_sentiment": 1.0,
            "breaking_news_flag": 0.0,
            "event_mna_flag": 0.0,
            "event_sanctions_flag": 0.0,
            "event_management_flag": 0.0,
        }
        for i in range(0, 2)
    ]
    merged = merge_market_news_features(
        market_rows,
        news_rows,
        bucket_minutes=60,
        news_lag_buckets=0,
    )
    train, val, test = chronological_split(
        merged,
        train_ratio=0.5,
        validation_ratio=0.25,
        test_ratio=0.25,
    )
    assert len(train) + len(val) + len(test) == len(merged)
    assert train[-1]["timestamp"] < val[0]["timestamp"]
    assert val[-1]["timestamp"] < test[0]["timestamp"]


def test_merge_uses_lagged_news_bucket_to_avoid_lookahead() -> None:
    market_rows = [
        {"ticker": "SBER", "timestamp": "2026-01-01T10:30:00+00:00", "close": 100.0},
    ]
    news_rows = [
        {
            "ticker": "SBER",
            "timestamp_bucket": "2026-01-01T10:00:00+00:00",
            "article_count": 9.0,
            "positive_count": 9.0,
            "negative_count": 0.0,
            "sentiment_mean": 1.0,
            "weighted_sentiment_mean": 1.0,
            "abnormal_news_volume": 0.0,
            "recency_weighted_sentiment": 1.0,
            "breaking_news_flag": 0.0,
            "event_mna_flag": 0.0,
            "event_sanctions_flag": 0.0,
            "event_management_flag": 0.0,
        },
        {
            "ticker": "SBER",
            "timestamp_bucket": "2026-01-01T09:00:00+00:00",
            "article_count": 4.0,
            "positive_count": 4.0,
            "negative_count": 0.0,
            "sentiment_mean": 1.0,
            "weighted_sentiment_mean": 1.0,
            "abnormal_news_volume": 0.0,
            "recency_weighted_sentiment": 1.0,
            "breaking_news_flag": 0.0,
            "event_mna_flag": 0.0,
            "event_sanctions_flag": 0.0,
            "event_management_flag": 0.0,
        },
    ]
    merged = merge_market_news_features(
        market_rows,
        news_rows,
        bucket_minutes=60,
        news_lag_buckets=1,
    )
    assert len(merged) == 1
    assert merged[0]["news_article_count"] == 4.0


def test_split_ratio_validation() -> None:
    import pytest

    with pytest.raises(ValueError):
        chronological_split([{"ticker": "SBER", "timestamp": "2026-01-01T10:00:00+00:00"}], train_ratio=0.6, validation_ratio=0.3, test_ratio=0.2)
