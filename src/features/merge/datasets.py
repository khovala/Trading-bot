from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _parse_ts(value: str) -> datetime:
    ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _floor_to_bucket(ts: datetime, bucket_minutes: int) -> datetime:
    minute = (ts.minute // bucket_minutes) * bucket_minutes
    return ts.replace(minute=minute, second=0, microsecond=0)


def merge_market_news_features(
    market_rows: list[dict],
    news_rows: list[dict],
    *,
    bucket_minutes: int,
    news_lag_buckets: int = 1,
) -> list[dict]:
    if bucket_minutes <= 0:
        raise ValueError("bucket_minutes must be > 0")
    if news_lag_buckets < 0:
        raise ValueError("news_lag_buckets must be >= 0")

    news_index: dict[tuple[str, str], dict] = {}
    for row in news_rows:
        ticker = str(row["ticker"]).upper()
        ts = _parse_ts(str(row["timestamp_bucket"])).isoformat()
        news_index[(ticker, ts)] = row

    merged: list[dict] = []
    for m in market_rows:
        ticker = str(m["ticker"]).upper()
        ts_dt = _parse_ts(str(m["timestamp"]))
        ts = ts_dt.isoformat()
        bucket_ts = _floor_to_bucket(ts_dt, bucket_minutes) - timedelta(minutes=bucket_minutes * news_lag_buckets)
        n = news_index.get((ticker, bucket_ts.isoformat()), {})
        row = dict(m)
        row["ticker"] = ticker
        row["timestamp"] = ts
        row["news_article_count"] = float(n.get("article_count", 0.0))
        row["news_positive_count"] = float(n.get("positive_count", 0.0))
        row["news_negative_count"] = float(n.get("negative_count", 0.0))
        row["news_sentiment_mean"] = float(n.get("sentiment_mean", 0.0))
        row["news_weighted_sentiment_mean"] = float(n.get("weighted_sentiment_mean", 0.0))
        row["news_abnormal_news_volume"] = float(n.get("abnormal_news_volume", 0.0))
        row["news_recency_weighted_sentiment"] = float(n.get("recency_weighted_sentiment", 0.0))
        row["news_breaking_news_flag"] = float(n.get("breaking_news_flag", 0.0))
        row["news_event_mna_flag"] = float(n.get("event_mna_flag", 0.0))
        row["news_event_sanctions_flag"] = float(n.get("event_sanctions_flag", 0.0))
        row["news_event_management_flag"] = float(n.get("event_management_flag", 0.0))
        merged.append(row)

    merged.sort(key=lambda x: (x["ticker"], x["timestamp"]))
    return merged


def chronological_split(
    rows: list[dict],
    *,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    if not rows:
        return [], [], []
    if abs((train_ratio + validation_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    by_ticker: dict[str, list[dict]] = {}
    for row in rows:
        by_ticker.setdefault(str(row["ticker"]).upper(), []).append(row)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    for ticker, ticker_rows in by_ticker.items():
        ordered = sorted(ticker_rows, key=lambda x: x["timestamp"])
        n = len(ordered)
        n_train = max(1, int(n * train_ratio)) if n >= 3 else max(0, int(n * train_ratio))
        n_val = max(1, int(n * validation_ratio)) if n >= 3 else max(0, int(n * validation_ratio))
        if n_train + n_val >= n and n >= 3:
            n_val = max(1, n - n_train - 1)
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
        split1 = n_train
        split2 = n_train + n_val
        train.extend(ordered[:split1])
        val.extend(ordered[split1:split2])
        test.extend(ordered[split2:])

    train.sort(key=lambda x: (x["ticker"], x["timestamp"]))
    val.sort(key=lambda x: (x["ticker"], x["timestamp"]))
    test.sort(key=lambda x: (x["ticker"], x["timestamp"]))
    return train, val, test
