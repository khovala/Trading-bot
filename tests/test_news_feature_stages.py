from __future__ import annotations

from src.data.ingestion.news.storage import read_parquet, write_jsonl
from src.training.pipeline.registry import StageRegistry
from src.training.pipeline.runner import PipelineRunner
from src.training.stages.register import register_default_stages


def test_map_and_generate_news_features_stages(tmp_path) -> None:
    manifest = {
        "stages": {
            "map_news_to_instruments": {
                "purpose": "map",
                "deps": ["data/interim/news"],
                "outs": ["data/processed/news/mapped_news.parquet"],
                "reports": ["reports/map_news_to_instruments.json"],
            },
            "generate_news_features": {
                "purpose": "features",
                "deps": ["data/processed/news/mapped_news.parquet"],
                "outs": ["data/processed/news/features"],
                "reports": ["reports/generate_news_features.json"],
            },
        }
    }
    params = {
        "data": {"news": {"aggregation_bucket_minutes": 60}},
        "features": {"news": {"recency_half_life_minutes": 180}},
        "stages": {"generate_news_features": {"bucket_minutes": 60, "source_weights": {"rbc_rss": 1.0}}},
    }

    write_jsonl(
        tmp_path / "data/interim/news/items.jsonl",
        [
            {
                "source": "rbc_rss",
                "title": "Сбербанк увеличил прибыль",
                "url": "https://example.com/n1",
                "published_at": "2026-03-11T07:20:00Z",
                "raw_text": "рост прибыли",
                "snippet": "рост",
                "body": None,
                "cleaned_text": "сбербанк рост прибыль",
            }
        ],
    )

    registry = StageRegistry()
    register_default_stages(registry)
    runner = PipelineRunner(workspace=tmp_path, manifest=manifest, params=params, registry=registry)

    map_result = runner.run_stage("map_news_to_instruments", fail_on_missing_inputs=True)
    feat_result = runner.run_stage("generate_news_features", fail_on_missing_inputs=True)

    assert map_result.success is True
    assert feat_result.success is True
    mapped_rows = read_parquet(tmp_path / "data/processed/news/mapped_news.parquet")
    feature_rows = read_parquet(tmp_path / "data/processed/news/features/news_features.parquet")
    assert len(mapped_rows) == 1
    assert mapped_rows[0]["ticker"] == "SBER"
    assert len(feature_rows) == 1
    assert feature_rows[0]["article_count"] == 1.0
