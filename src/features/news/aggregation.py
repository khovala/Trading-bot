from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import math


POSITIVE_WORDS = ("рост", "прибыль", "дивиденд", "улучш", "positive", "gain")
NEGATIVE_WORDS = ("паден", "убыт", "санкц", "снижен", "negative", "loss")
BREAKING_WORDS = ("срочно", "breaking", "urgent")
EVENT_WORDS = {
    "event_mna_flag": ("слиян", "поглощен", "m&a"),
    "event_sanctions_flag": ("санкц", "ограничен", "санкции"),
    "event_management_flag": ("гендиректор", "совет директоров", "назначен"),
}


def sentiment_score(text: str) -> float:
    lowered = text.lower()
    pos = sum(1 for token in POSITIVE_WORDS if token in lowered)
    neg = sum(1 for token in NEGATIVE_WORDS if token in lowered)
    total = pos + neg
    if total == 0:
        return 0.0
    return float((pos - neg) / total)


def floor_to_bucket(ts: datetime, bucket_minutes: int) -> datetime:
    utc = ts.astimezone(timezone.utc)
    minute = (utc.minute // bucket_minutes) * bucket_minutes
    return utc.replace(minute=minute, second=0, microsecond=0)


def generate_news_features(
    rows: list[dict],
    *,
    bucket_minutes: int,
    source_weights: dict[str, float],
    recency_half_life_minutes: int,
) -> list[dict]:
    grouped: dict[tuple[str, datetime], list[dict]] = defaultdict(list)
    for row in rows:
        ticker = str(row.get("ticker", "UNKNOWN")).upper()
        if ticker == "UNKNOWN":
            continue
        ts = datetime.fromisoformat(str(row["published_at"]).replace("Z", "+00:00"))
        bucket_ts = floor_to_bucket(ts, bucket_minutes)
        grouped[(ticker, bucket_ts)].append(row)

    ordered_keys = sorted(grouped.keys(), key=lambda x: (x[0], x[1]))
    history_counts: dict[str, list[int]] = defaultdict(list)
    results: list[dict] = []
    for ticker, bucket_ts in ordered_keys:
        items = grouped[(ticker, bucket_ts)]
        scores = [sentiment_score(str(x.get("cleaned_text", ""))) for x in items]
        source_weighted_scores = []
        recency_weighted_scores = []

        bucket_end = bucket_ts + timedelta(minutes=bucket_minutes)
        for item, score in zip(items, scores):
            weight = float(source_weights.get(str(item.get("source", "")).lower(), 1.0))
            source_weighted_scores.append(score * weight)

            published = datetime.fromisoformat(str(item["published_at"]).replace("Z", "+00:00"))
            age_minutes = max(0.0, (bucket_end - published).total_seconds() / 60.0)
            decay = math.exp(-age_minutes / max(1.0, float(recency_half_life_minutes)))
            recency_weighted_scores.append(score * decay)

        article_count = len(items)
        pos_count = sum(1 for s in scores if s > 0)
        neg_count = sum(1 for s in scores if s < 0)
        sentiment_mean = sum(scores) / article_count if article_count else 0.0
        weighted_sentiment_mean = (
            sum(source_weighted_scores) / len(source_weighted_scores) if source_weighted_scores else 0.0
        )
        recency_weighted_sent = (
            sum(recency_weighted_scores) / len(recency_weighted_scores) if recency_weighted_scores else 0.0
        )
        breaking_news_flag = 1.0 if any(any(k in str(x.get("title", "")).lower() for k in BREAKING_WORDS) for x in items) else 0.0

        hist = history_counts[ticker]
        if len(hist) >= 3:
            mean = sum(hist) / len(hist)
            variance = sum((c - mean) ** 2 for c in hist) / len(hist)
            std = variance ** 0.5
            abnormal = 1.0 if article_count > mean + 2.0 * std else 0.0
        else:
            abnormal = 0.0
        hist.append(article_count)

        event_flags = {}
        joined_text = " ".join(str(x.get("cleaned_text", "")).lower() for x in items)
        for flag_name, keywords in EVENT_WORDS.items():
            event_flags[flag_name] = 1.0 if any(k in joined_text for k in keywords) else 0.0

        results.append(
            {
                "ticker": ticker,
                "timestamp_bucket": bucket_ts.isoformat(),
                "article_count": float(article_count),
                "positive_count": float(pos_count),
                "negative_count": float(neg_count),
                "sentiment_mean": float(sentiment_mean),
                "weighted_sentiment_mean": float(weighted_sentiment_mean),
                "abnormal_news_volume": float(abnormal),
                "recency_weighted_sentiment": float(recency_weighted_sent),
                "breaking_news_flag": float(breaking_news_flag),
                **event_flags,
            }
        )
    return results
