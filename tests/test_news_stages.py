from __future__ import annotations

from src.training.pipeline.registry import StageRegistry
from src.training.pipeline.runner import PipelineRunner
from src.training.stages.register import register_default_stages


def test_news_stages_run_and_write_outputs(tmp_path) -> None:
    manifest = {
        "stages": {
            "download_news_data": {
                "purpose": "download",
                "deps": [],
                "outs": ["data/raw/news"],
                "reports": ["reports/download_news_data.json"],
            },
            "preprocess_news_data": {
                "purpose": "preprocess",
                "deps": ["data/raw/news"],
                "outs": ["data/interim/news"],
                "reports": ["reports/preprocess_news_data.json"],
            },
        }
    }
    params = {
        "data": {"news": {"enabled": True, "sources": ["rbc_rss"]}},
        "stages": {
            "download_news_data": {
                "feed_urls": {"rbc_rss": "https://example.com/rss"},
                "limit_per_source": 10,
                "fail_if_all_sources_failed": False,
            }
        },
    }
    registry = StageRegistry()
    register_default_stages(registry)
    runner = PipelineRunner(workspace=tmp_path, manifest=manifest, params=params, registry=registry)

    import src.training.stages.news as news_stage

    original_builder = news_stage.build_news_source_adapter

    class _FakeAdapter:
        def fetch(self, limit: int = 200):
            return [
                {
                    "source": "rbc_rss",
                    "title": "Headline",
                    "url": "https://example.com/1",
                    "published_at": "2026-03-11T07:20:00Z",
                    "raw_text": "<b>Headline</b>",
                    "snippet": "Snippet",
                    "body": None,
                }
            ]

    def _fake_builder(source_name: str, feed_url: str):
        class _Adapter:
            def fetch(self, limit: int = 200):
                from src.data.schemas.news import RawNewsItemRecord

                return [RawNewsItemRecord.model_validate(x) for x in _FakeAdapter().fetch(limit=limit)]

        return _Adapter()

    news_stage.build_news_source_adapter = _fake_builder
    try:
        download_result = runner.run_stage("download_news_data")
        preprocess_result = runner.run_stage("preprocess_news_data", fail_on_missing_inputs=True)
    finally:
        news_stage.build_news_source_adapter = original_builder

    assert download_result.success is True
    assert preprocess_result.success is True
    assert (tmp_path / "data/raw/news/items.jsonl").exists()
    assert (tmp_path / "data/interim/news/items.jsonl").exists()


def test_download_news_stage_fails_when_all_sources_fail(tmp_path) -> None:
    manifest = {
        "stages": {
            "download_news_data": {
                "purpose": "download",
                "deps": [],
                "outs": ["data/raw/news"],
                "reports": [],
            }
        }
    }
    params = {
        "data": {"news": {"enabled": True, "sources": ["rbc_rss"]}},
        "stages": {"download_news_data": {"fail_if_all_sources_failed": True}},
    }
    registry = StageRegistry()
    register_default_stages(registry)
    runner = PipelineRunner(workspace=tmp_path, manifest=manifest, params=params, registry=registry)

    import src.training.stages.news as news_stage

    original_builder = news_stage.build_news_source_adapter

    class _BrokenAdapter:
        def fetch(self, limit: int = 200):
            raise RuntimeError("boom")

    news_stage.build_news_source_adapter = lambda source_name, feed_url: _BrokenAdapter()
    try:
        result = runner.run_stage("download_news_data")
    finally:
        news_stage.build_news_source_adapter = original_builder

    assert result.success is False
