from __future__ import annotations

from datetime import datetime, timezone

from src.data.ingestion.news.rss import build_news_source_adapter
from src.data.ingestion.news.storage import (
    interim_news_items_path,
    raw_news_items_path,
    read_jsonl,
    write_jsonl,
)
from src.data.preprocessing.news.cleaning import deduplicate_items, preprocess_items
from src.data.schemas.news import RawNewsItemRecord
from src.domain.schemas import StageResult
from src.training.pipeline.base import PipelineStage, StageContext

DEFAULT_NEWS_FEEDS: dict[str, str] = {
    "rbc_rss": "https://rssexport.rbc.ru/rbcnews/news/30/full.rss",
    "interfax_rss": "https://www.interfax.ru/rss.asp",
    "cbr_rss": "https://www.cbr.ru/rss/RssNews",
}


class DownloadNewsDataStage(PipelineStage):
    def run(self, ctx: StageContext) -> StageResult:
        started = datetime.now(tz=timezone.utc)
        news_cfg = ctx.params.get("data", {}).get("news", {})
        stage_cfg = ctx.stage_params
        if not bool(news_cfg.get("enabled", True)):
            write_jsonl(raw_news_items_path(ctx.workspace), [])
            finished = datetime.now(tz=timezone.utc)
            return StageResult(
                run_id=ctx.run_id,
                stage_name=ctx.stage_name,
                success=True,
                started_at=started,
                finished_at=finished,
                metrics={"raw_item_count": 0.0, "deduped_item_count": 0.0, "source_error_count": 0.0},
                artifacts=["data/raw/news/items.jsonl"],
            )

        source_names = [str(x) for x in news_cfg.get("sources", ["rbc_rss"])]
        feed_urls = {**DEFAULT_NEWS_FEEDS, **dict(stage_cfg.get("feed_urls", {}))}
        limit_per_source = max(1, int(stage_cfg.get("limit_per_source", 100)))
        fail_if_all_sources_failed = bool(stage_cfg.get("fail_if_all_sources_failed", True))

        all_items: list[RawNewsItemRecord] = []
        source_errors = 0
        for source_name in source_names:
            feed_url = feed_urls.get(source_name)
            if not feed_url:
                source_errors += 1
                continue
            try:
                adapter = build_news_source_adapter(source_name=source_name, feed_url=feed_url)
                all_items.extend(adapter.fetch(limit=limit_per_source))
            except Exception:
                source_errors += 1

        deduped = deduplicate_items(all_items)
        dedup_removed = len(all_items) - len(deduped)
        write_jsonl(raw_news_items_path(ctx.workspace), [x.model_dump(mode="json") for x in deduped])

        finished = datetime.now(tz=timezone.utc)
        all_failed = len(source_names) > 0 and source_errors >= len(source_names)
        success = not (fail_if_all_sources_failed and all_failed)
        return StageResult(
            run_id=ctx.run_id,
            stage_name=ctx.stage_name,
            success=success,
            started_at=started,
            finished_at=finished,
            metrics={
                "raw_item_count": float(len(all_items)),
                "deduped_item_count": float(len(deduped)),
                "dedup_removed": float(dedup_removed),
                "source_error_count": float(source_errors),
                "source_count": float(len(source_names)),
            },
            artifacts=["data/raw/news/items.jsonl"],
        )


class PreprocessNewsDataStage(PipelineStage):
    def run(self, ctx: StageContext) -> StageResult:
        started = datetime.now(tz=timezone.utc)
        raw_rows = read_jsonl(raw_news_items_path(ctx.workspace))
        raw_items = [RawNewsItemRecord.model_validate(row) for row in raw_rows]
        deduped_items = deduplicate_items(raw_items)
        processed_items = preprocess_items(deduped_items)

        write_jsonl(interim_news_items_path(ctx.workspace), [x.model_dump(mode="json") for x in processed_items])
        finished = datetime.now(tz=timezone.utc)
        return StageResult(
            run_id=ctx.run_id,
            stage_name=ctx.stage_name,
            success=True,
            started_at=started,
            finished_at=finished,
            metrics={
                "input_item_count": float(len(raw_items)),
                "deduped_item_count": float(len(deduped_items)),
                "processed_item_count": float(len(processed_items)),
            },
            artifacts=["data/interim/news/items.jsonl"],
        )
