from __future__ import annotations

from src.models.news.news_encoder_pipeline import NewsEncoderPipeline


def test_news_encoder_pipeline_fit_transform() -> None:
    pipeline = NewsEncoderPipeline(embedding_dim=8, relevance_threshold=0.0)
    rows = [
        {
            "source": "rbc",
            "title": "Strong profit growth expected",
            "cleaned_text": "Strong profit growth expected from company",
            "published_at": "2026-01-01T10:00:00+00:00",
        },
        {
            "source": "ifax",
            "title": "Risk of decline remains",
            "cleaned_text": "Risk of decline remains for sector",
            "published_at": "2026-01-01T11:00:00+00:00",
        },
    ]

    metrics, encoded = pipeline.fit_transform(rows)

    assert metrics["samples"] == 2.0
    assert len(encoded) == 2
    assert len(encoded[0]["embedding"]) == 8
    assert "weighted_sentiment" in encoded[0]
    assert encoded[0]["encoder_name"] == "hashing_tfidf_v1"
