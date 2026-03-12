from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MappedNewsItem:
    source: str
    title: str
    url: str | None
    published_at: str
    raw_text: str
    cleaned_text: str
    ticker: str
    issuer_name: str
    sector: str | None
    mapping_method: str
    mapping_confidence: float
