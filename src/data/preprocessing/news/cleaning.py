from __future__ import annotations

import hashlib
import re

from src.data.schemas.news import ProcessedNewsItemRecord, RawNewsItemRecord

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_text(value: str) -> str:
    stripped = _TAG_RE.sub(" ", value)
    stripped = _WS_RE.sub(" ", stripped).strip()
    return stripped


def dedup_key(item: RawNewsItemRecord) -> str:
    if item.url:
        return f"url::{item.url.strip().lower()}"
    base = f"{item.source}|{item.title.strip().lower()}|{item.published_at.isoformat()[:16]}"
    return "hash::" + hashlib.sha256(base.encode("utf-8")).hexdigest()


def deduplicate_items(items: list[RawNewsItemRecord]) -> list[RawNewsItemRecord]:
    seen: set[str] = set()
    deduped: list[RawNewsItemRecord] = []
    for item in items:
        key = dedup_key(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def preprocess_items(items: list[RawNewsItemRecord]) -> list[ProcessedNewsItemRecord]:
    processed: list[ProcessedNewsItemRecord] = []
    for item in items:
        cleaned_text = clean_text(item.raw_text)
        if not cleaned_text:
            continue
        processed.append(
            ProcessedNewsItemRecord(
                **item.model_dump(),
                cleaned_text=cleaned_text,
            )
        )
    return processed
