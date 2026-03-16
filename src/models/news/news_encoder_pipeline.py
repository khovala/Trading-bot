from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import md5
from math import exp
from typing import Any


POSITIVE_TOKENS = {
    "growth",
    "profit",
    "beats",
    "upgrade",
    "strong",
    "buyback",
    "record",
    "increase",
    "surge",
}
NEGATIVE_TOKENS = {
    "loss",
    "downgrade",
    "weak",
    "decline",
    "drop",
    "risk",
    "sanction",
    "lawsuit",
    "fall",
}


def _to_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _tokenize(text: str) -> list[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [token for token in cleaned.split() if token]


def _embedding_bucket(token: str, dim: int) -> int:
    digest = md5(token.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(digest[:8], 16) % max(1, dim)


@dataclass
class NewsEncoderPipeline:
    encoder_name: str = "hashing_tfidf_v1"
    embedding_dim: int = 16
    half_life_minutes: float = 180.0
    min_tokens: int = 3
    relevance_threshold: float = 0.01

    fitted_samples: int = 0
    average_sentiment: float = 0.0
    average_relevance: float = 0.0
    latest_timestamp: str | None = None

    def _sentiment(self, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        pos = sum(1 for t in tokens if t in POSITIVE_TOKENS)
        neg = sum(1 for t in tokens if t in NEGATIVE_TOKENS)
        return (pos - neg) / max(1, len(tokens))

    def _relevance(self, tokens: list[str]) -> float:
        if len(tokens) < self.min_tokens:
            return 0.0
        signal = sum(1 for t in tokens if t in POSITIVE_TOKENS or t in NEGATIVE_TOKENS)
        return min(1.0, signal / max(1, len(tokens)))

    def _recency_weight(self, published_at: datetime, reference_ts: datetime) -> float:
        delta_minutes = max(0.0, (reference_ts - published_at).total_seconds() / 60.0)
        return exp(-delta_minutes / max(1.0, self.half_life_minutes))

    def fit(self, rows: list[dict[str, Any]]) -> dict[str, float]:
        transformed = self.transform(rows)
        self.fitted_samples = len(transformed)
        if not transformed:
            self.average_sentiment = 0.0
            self.average_relevance = 0.0
            self.latest_timestamp = None
            return {"samples": 0.0, "avg_sentiment": 0.0, "avg_relevance": 0.0}

        self.average_sentiment = sum(float(r["sentiment"]) for r in transformed) / len(transformed)
        self.average_relevance = sum(float(r["relevance"]) for r in transformed) / len(transformed)
        self.latest_timestamp = max(str(r["published_at"]) for r in transformed)
        return {
            "samples": float(len(transformed)),
            "avg_sentiment": float(self.average_sentiment),
            "avg_relevance": float(self.average_relevance),
        }

    def transform(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        parsed: list[tuple[dict[str, Any], datetime, list[str]]] = []
        for row in rows:
            text = str(row.get("cleaned_text") or row.get("raw_text") or row.get("title") or "")
            published_at = _to_utc(row.get("published_at", datetime.now(timezone.utc).isoformat()))
            tokens = _tokenize(text)
            parsed.append((row, published_at, tokens))
        ref_ts = max(published_at for _, published_at, _ in parsed)

        output: list[dict[str, Any]] = []
        for row, published_at, tokens in parsed:
            relevance = self._relevance(tokens)
            if relevance < self.relevance_threshold:
                continue
            sentiment = self._sentiment(tokens)
            recency_weight = self._recency_weight(published_at, ref_ts)
            embedding = [0.0 for _ in range(self.embedding_dim)]
            for token in tokens:
                embedding[_embedding_bucket(token, self.embedding_dim)] += 1.0
            norm = max(1.0, sum(embedding))
            normalized_embedding = [v / norm for v in embedding]
            output.append(
                {
                    "source": row.get("source", "unknown"),
                    "title": row.get("title", ""),
                    "published_at": published_at.isoformat(),
                    "sentiment": float(sentiment),
                    "relevance": float(relevance),
                    "recency_weight": float(recency_weight),
                    "weighted_sentiment": float(sentiment * recency_weight),
                    "embedding": normalized_embedding,
                    "encoder_name": self.encoder_name,
                }
            )
        return output

    def fit_transform(self, rows: list[dict[str, Any]]) -> tuple[dict[str, float], list[dict[str, Any]]]:
        metrics = self.fit(rows)
        return metrics, self.transform(rows)

    def get_metadata(self) -> dict[str, Any]:
        return {
            "encoder_name": self.encoder_name,
            "embedding_dim": self.embedding_dim,
            "half_life_minutes": self.half_life_minutes,
            "min_tokens": self.min_tokens,
            "relevance_threshold": self.relevance_threshold,
            "fitted_samples": self.fitted_samples,
            "average_sentiment": self.average_sentiment,
            "average_relevance": self.average_relevance,
            "latest_timestamp": self.latest_timestamp or "",
        }
