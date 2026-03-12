from __future__ import annotations

from src.data.preprocessing.market.validation import (
    find_duplicate_bars,
    find_missing_bars,
    validate_candles,
    validate_instruments,
)
from src.data.schemas.market import CandleRecord, InstrumentRecord


def preprocess_market_dataset(
    *,
    instrument_rows: list[dict],
    candles_by_ticker: dict[str, list[dict]],
    interval_minutes: int,
) -> tuple[list[InstrumentRecord], dict[str, list[CandleRecord]], dict[str, float]]:
    instruments = validate_instruments(instrument_rows)

    normalized_by_ticker: dict[str, list[CandleRecord]] = {}
    duplicate_total = 0
    missing_total = 0
    for ticker, rows in candles_by_ticker.items():
        validated = validate_candles(rows)
        duplicates = find_duplicate_bars(validated)
        duplicate_total += len(duplicates)

        deduped = {
            f"{row.ticker}|{row.timestamp.isoformat()}|{row.interval}": row for row in validated
        }
        ordered = sorted(deduped.values(), key=lambda x: x.timestamp)
        missing_total += find_missing_bars(ordered, interval_minutes).get(ticker.upper(), 0)
        normalized_by_ticker[ticker.upper()] = ordered

    metrics = {
        "instrument_count": float(len(instruments)),
        "ticker_count": float(len(normalized_by_ticker)),
        "duplicate_bar_count": float(duplicate_total),
        "missing_bar_count": float(missing_total),
    }
    return instruments, normalized_by_ticker, metrics
