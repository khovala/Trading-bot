from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.schemas import RawNewsItem


class NewsSourceAdapter(ABC):
    source_name: str = "unknown"

    @abstractmethod
    def fetch(self, *, since_iso: str | None = None, limit: int = 100) -> list[RawNewsItem]:
        """Return raw news items from a single source adapter."""
        raise NotImplementedError
