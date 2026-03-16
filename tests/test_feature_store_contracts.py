from __future__ import annotations

import pytest

from src.data.feature_store.contracts import validate_feature_row


def test_validate_feature_row_accepts_valid_market_row() -> None:
    row = {
        "ticker": "sber",
        "timestamp": "2026-01-01T10:00:00+00:00",
        "close": 100.0,
        "return_1": 0.001,
        "log_return_1": 0.00099,
        "rolling_volatility_20": 0.01,
        "momentum_10": 0.1,
        "volume": 1000.0,
    }
    parsed = validate_feature_row(row)
    assert parsed.ticker == "SBER"


def test_validate_feature_row_requires_news_fields_when_requested() -> None:
    row = {
        "ticker": "SBER",
        "timestamp": "2026-01-01T10:00:00+00:00",
        "close": 100.0,
        "return_1": 0.001,
        "log_return_1": 0.00099,
        "rolling_volatility_20": 0.01,
        "momentum_10": 0.1,
        "volume": 1000.0,
    }
    with pytest.raises(ValueError):
        validate_feature_row(row, require_news=True)
