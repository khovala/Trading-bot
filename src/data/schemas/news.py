from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator


class RawNewsItemRecord(BaseModel):
    source: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: str | None = None
    published_at: datetime
    raw_text: str = Field(min_length=1)
    snippet: str | None = None
    body: str | None = None

    @field_validator("published_at")
    @classmethod
    def normalize_published_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @field_validator("source")
    @classmethod
    def normalize_source(cls, value: str) -> str:
        return value.strip().lower()


class ProcessedNewsItemRecord(RawNewsItemRecord):
    cleaned_text: str = Field(min_length=1)
