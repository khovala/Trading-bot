from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.data.preprocessing.market.validation import (
    find_duplicate_bars,
    find_missing_bars,
    normalize_timestamp,
    validate_candles,
    validate_instruments,
)


def _candle(ts: str) -> dict:
    return {
        "ticker": "SBER",
        "timestamp": ts,
        "interval": "1min",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000,
        "source": "test",
    }


def test_timestamp_normalization_to_utc_minute() -> None:
    ts = datetime(2026, 1, 1, 10, 0, 45, 123456, tzinfo=timezone.utc)
    normalized = normalize_timestamp(ts)
    assert normalized.second == 0
    assert normalized.microsecond == 0
    assert normalized.tzinfo == timezone.utc


def test_validate_candles_accepts_z_suffix_timestamp() -> None:
    candles = validate_candles([_candle("2026-01-01T10:00:00Z")])
    assert candles[0].timestamp.tzinfo == timezone.utc


def test_duplicate_and_missing_bar_detection() -> None:
    rows = [
        _candle("2026-01-01T10:00:00+00:00"),
        _candle("2026-01-01T10:00:00+00:00"),
        _candle("2026-01-01T10:02:00+00:00"),
    ]
    candles = validate_candles(rows)
    duplicates = find_duplicate_bars(candles)
    missing = find_missing_bars(candles, interval_minutes=1)
    assert len(duplicates) == 1
    assert missing["SBER"] == 1


def test_schema_validation_rejects_bad_ohlc() -> None:
    bad = _candle("2026-01-01T10:00:00+00:00")
    bad["low"] = 200.0
    with pytest.raises(ValueError):
        validate_candles([bad])


def test_instrument_schema_validation() -> None:
    instruments = validate_instruments([{"ticker": "sber", "name": "Sberbank"}])
    assert instruments[0].ticker == "SBER"
