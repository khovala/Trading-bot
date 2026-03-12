from __future__ import annotations

from datetime import datetime, timezone

from src.data.schemas.news import ProcessedNewsItemRecord
from src.features.news.aggregation import generate_news_features
from src.nlp.entity_mapping.dictionary_mapper import DictionaryIssuerTickerMapper


def _item(title: str, cleaned_text: str, published_at: str) -> ProcessedNewsItemRecord:
    return ProcessedNewsItemRecord(
        source="rbc_rss",
        title=title,
        url=None,
        published_at=published_at,
        raw_text=cleaned_text,
        snippet=None,
        body=None,
        cleaned_text=cleaned_text,
    )


def test_dictionary_mapper_maps_alias_and_sector_fallback() -> None:
    mapper = DictionaryIssuerTickerMapper()

    mapped_alias = mapper.map_item(
        _item("Сбербанк увеличил прибыль", "Сбербанк сообщил о росте прибыли", "2026-03-11T07:20:00+00:00")
    )
    assert mapped_alias.ticker == "SBER"
    assert mapped_alias.mapping_method in {"issuer_alias", "ticker_exact"}

    mapped_sector = mapper.map_item(
        _item("Нефтяной сектор под давлением", "Нефтяные компании теряют капитализацию", "2026-03-11T07:30:00+00:00")
    )
    assert mapped_sector.mapping_method == "sector_fallback"
    assert mapped_sector.ticker == "GAZP"


def test_time_bucket_aggregation_is_deterministic_and_leakage_safe() -> None:
    rows = [
        {
            "source": "rbc_rss",
            "title": "СРОЧНО: Сбербанк повышает дивиденды",
            "url": "https://x/1",
            "published_at": "2026-03-11T07:05:00+00:00",
            "raw_text": "рост прибыли",
            "cleaned_text": "срочно рост прибыль дивиденд",
            "ticker": "SBER",
            "issuer_name": "Sberbank",
            "sector": None,
            "mapping_method": "issuer_alias",
            "mapping_confidence": 0.95,
        },
        {
            "source": "interfax_rss",
            "title": "Сбербанк под давлением",
            "url": "https://x/2",
            "published_at": "2026-03-11T07:40:00+00:00",
            "raw_text": "падение",
            "cleaned_text": "паден убыток санкц",
            "ticker": "SBER",
            "issuer_name": "Sberbank",
            "sector": None,
            "mapping_method": "issuer_alias",
            "mapping_confidence": 0.95,
        },
        {
            "source": "rbc_rss",
            "title": "Сбербанк стабилен",
            "url": "https://x/3",
            "published_at": "2026-03-11T08:10:00+00:00",
            "raw_text": "нейтрально",
            "cleaned_text": "без существенных изменений",
            "ticker": "SBER",
            "issuer_name": "Sberbank",
            "sector": None,
            "mapping_method": "issuer_alias",
            "mapping_confidence": 0.95,
        },
    ]
    features = generate_news_features(
        rows,
        bucket_minutes=60,
        source_weights={"rbc_rss": 1.0, "interfax_rss": 1.2},
        recency_half_life_minutes=180,
    )
    assert len(features) == 2
    first, second = features[0], features[1]
    assert first["ticker"] == "SBER"
    assert first["article_count"] == 2.0
    assert first["breaking_news_flag"] == 1.0
    assert second["article_count"] == 1.0
    assert second["abnormal_news_volume"] == 0.0

    # Ensure deterministic ordering by ticker and bucket timestamp.
    first_ts = datetime.fromisoformat(first["timestamp_bucket"]).astimezone(timezone.utc)
    second_ts = datetime.fromisoformat(second["timestamp_bucket"]).astimezone(timezone.utc)
    assert first_ts < second_ts
