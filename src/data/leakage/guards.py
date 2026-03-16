from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


def _to_utc(ts: str | datetime) -> datetime:
    if isinstance(ts, datetime):
        value = ts
    else:
        value = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def assert_monotonic_by_ticker(
    rows: list[dict[str, Any]],
    *,
    ticker_key: str = "ticker",
    ts_key: str = "timestamp",
) -> None:
    per_ticker: dict[str, list[datetime]] = {}
    for row in rows:
        ticker = str(row[ticker_key]).upper()
        per_ticker.setdefault(ticker, []).append(_to_utc(row[ts_key]))

    for ticker, seq in per_ticker.items():
        for i in range(1, len(seq)):
            if seq[i] < seq[i - 1]:
                raise ValueError(f"Non-monotonic timestamps for ticker={ticker}")


def assert_no_split_overlap(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    ticker_key: str = "ticker",
    ts_key: str = "timestamp",
    embargo_minutes: int = 0,
) -> None:
    embargo = timedelta(minutes=max(0, embargo_minutes))

    by_ticker: dict[str, dict[str, list[datetime]]] = {}
    for split_name, rows in (
        ("train", train_rows),
        ("validation", validation_rows),
        ("test", test_rows),
    ):
        for row in rows:
            ticker = str(row[ticker_key]).upper()
            by_ticker.setdefault(ticker, {}).setdefault(split_name, []).append(_to_utc(row[ts_key]))

    for ticker, chunks in by_ticker.items():
        train_ts = sorted(chunks.get("train", []))
        val_ts = sorted(chunks.get("validation", []))
        test_ts = sorted(chunks.get("test", []))

        if train_ts and val_ts and not (train_ts[-1] + embargo < val_ts[0]):
            raise ValueError(f"Train/validation overlap or embargo violation for ticker={ticker}")
        if val_ts and test_ts and not (val_ts[-1] + embargo < test_ts[0]):
            raise ValueError(f"Validation/test overlap or embargo violation for ticker={ticker}")


def chronological_split_with_embargo(
    rows: list[dict[str, Any]],
    *,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    ticker_key: str = "ticker",
    ts_key: str = "timestamp",
    embargo_minutes: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if abs((train_ratio + validation_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    by_ticker: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        ticker = str(row[ticker_key]).upper()
        by_ticker.setdefault(ticker, []).append(row)

    embargo = timedelta(minutes=max(0, embargo_minutes))
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    for ticker, ticker_rows in by_ticker.items():
        ordered = sorted(ticker_rows, key=lambda x: _to_utc(x[ts_key]))
        n = len(ordered)
        if n == 0:
            continue
        cut1 = int(n * train_ratio)
        cut2 = cut1 + int(n * validation_ratio)
        cut1 = max(1, min(cut1, n))
        cut2 = max(cut1, min(cut2, n))
        t = ordered[:cut1]
        v = ordered[cut1:cut2]
        s = ordered[cut2:]

        if embargo > timedelta(0):
            if t and v:
                min_val_ts = _to_utc(v[0][ts_key])
                t = [r for r in t if _to_utc(r[ts_key]) + embargo < min_val_ts]
            if v and s:
                min_test_ts = _to_utc(s[0][ts_key])
                v = [r for r in v if _to_utc(r[ts_key]) + embargo < min_test_ts]

        train.extend(t)
        val.extend(v)
        test.extend(s)

    train.sort(key=lambda x: (str(x[ticker_key]).upper(), _to_utc(x[ts_key])))
    val.sort(key=lambda x: (str(x[ticker_key]).upper(), _to_utc(x[ts_key])))
    test.sort(key=lambda x: (str(x[ticker_key]).upper(), _to_utc(x[ts_key])))
    return train, val, test


def asof_join(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    left_ts_key: str = "timestamp",
    right_ts_key: str = "timestamp",
    by_key: str = "ticker",
    max_lag_minutes: int = 60,
    allow_exact: bool = True,
) -> list[dict[str, Any]]:
    max_lag = timedelta(minutes=max(0, max_lag_minutes))
    right_index: dict[str, list[dict[str, Any]]] = {}
    for row in right_rows:
        key = str(row[by_key]).upper()
        right_index.setdefault(key, []).append(row)
    for key in right_index:
        right_index[key].sort(key=lambda r: _to_utc(r[right_ts_key]))

    out: list[dict[str, Any]] = []
    for lrow in left_rows:
        key = str(lrow[by_key]).upper()
        lts = _to_utc(lrow[left_ts_key])
        candidates = right_index.get(key, [])
        best: dict[str, Any] | None = None
        for rrow in candidates:
            rts = _to_utc(rrow[right_ts_key])
            if (allow_exact and rts <= lts) or (not allow_exact and rts < lts):
                if lts - rts <= max_lag:
                    best = rrow
            else:
                break
        merged = dict(lrow)
        if best:
            for rk, rv in best.items():
                merged[f"asof_{rk}"] = rv
        out.append(merged)
    return out
