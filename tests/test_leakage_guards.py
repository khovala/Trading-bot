from __future__ import annotations

import pytest

from src.data.leakage.guards import (
    asof_join,
    assert_monotonic_by_ticker,
    assert_no_split_overlap,
    chronological_split_with_embargo,
)


def test_assert_monotonic_by_ticker_raises_for_descending_timestamps() -> None:
    rows = [
        {"ticker": "SBER", "timestamp": "2026-01-01T10:01:00+00:00"},
        {"ticker": "SBER", "timestamp": "2026-01-01T10:00:00+00:00"},
    ]
    with pytest.raises(ValueError):
        assert_monotonic_by_ticker(rows)


def test_chronological_split_with_embargo_produces_non_overlapping_splits() -> None:
    rows = [
        {"ticker": "SBER", "timestamp": f"2026-01-01T10:{i:02d}:00+00:00", "x": i}
        for i in range(30)
    ]
    train, val, test = chronological_split_with_embargo(
        rows,
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
        embargo_minutes=1,
    )
    assert train and val and test
    assert_no_split_overlap(train, val, test, embargo_minutes=1)


def test_asof_join_never_joins_future_rows() -> None:
    left = [{"ticker": "SBER", "timestamp": "2026-01-01T10:10:00+00:00"}]
    right = [
        {"ticker": "SBER", "timestamp": "2026-01-01T10:09:00+00:00", "value": 1},
        {"ticker": "SBER", "timestamp": "2026-01-01T10:11:00+00:00", "value": 2},
    ]
    out = asof_join(left, right, max_lag_minutes=5)
    assert out[0]["asof_value"] == 1
