from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from src.data.schemas.market import CandleRecord, InstrumentRecord


def normalize_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    value = value.astimezone(timezone.utc)
    return value.replace(second=0, microsecond=0)


def validate_instruments(rows: list[dict]) -> list[InstrumentRecord]:
    return [InstrumentRecord.model_validate(row) for row in rows]


def validate_candles(rows: list[dict]) -> list[CandleRecord]:
    normalized: list[CandleRecord] = []
    for row in rows:
        payload = dict(row)
        if "timestamp" in payload:
            raw_ts = str(payload["timestamp"]).replace("Z", "+00:00")
            payload["timestamp"] = normalize_timestamp(datetime.fromisoformat(raw_ts))
        normalized.append(CandleRecord.model_validate(payload))
    return normalized


def find_duplicate_bars(candles: list[CandleRecord]) -> list[str]:
    keys = [f"{row.ticker}|{row.timestamp.isoformat()}|{row.interval}" for row in candles]
    counts = Counter(keys)
    return sorted([key for key, count in counts.items() if count > 1])


def find_missing_bars(candles: list[CandleRecord], interval_minutes: int) -> dict[str, int]:
    grouped: dict[str, list[datetime]] = defaultdict(list)
    for row in candles:
        grouped[row.ticker].append(row.timestamp)

    missing_by_ticker: dict[str, int] = {}
    step = timedelta(minutes=interval_minutes)
    for ticker, ts_values in grouped.items():
        ordered = sorted(set(ts_values))
        missing = 0
        for idx in range(1, len(ordered)):
            gap = ordered[idx] - ordered[idx - 1]
            if gap > step:
                missing += int(gap.total_seconds() // step.total_seconds()) - 1
        missing_by_ticker[ticker] = missing
    return missing_by_ticker
