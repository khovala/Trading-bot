from __future__ import annotations

from abc import ABC, abstractmethod

from src.data.schemas.news import RawNewsItemRecord


class NewsSourceAdapter(ABC):
    source_name: str = "unknown"

    @abstractmethod
    def fetch(self, limit: int = 200) -> list[RawNewsItemRecord]:
        raise NotImplementedError
