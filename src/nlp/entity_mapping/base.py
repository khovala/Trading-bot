from __future__ import annotations

from abc import ABC, abstractmethod

from src.data.schemas.news import ProcessedNewsItemRecord
from src.nlp.entity_mapping.schemas import MappedNewsItem


class EntityMapper(ABC):
    @abstractmethod
    def map_item(self, item: ProcessedNewsItemRecord) -> MappedNewsItem:
        """Map processed news item to MOEX issuer/ticker."""
        raise NotImplementedError
