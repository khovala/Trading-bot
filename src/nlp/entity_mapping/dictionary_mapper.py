from __future__ import annotations

import re

from src.data.schemas.news import ProcessedNewsItemRecord
from src.nlp.entity_mapping.dictionaries import (
    ISSUER_ALIASES,
    SECTOR_FALLBACK_TICKER,
    SECTOR_KEYWORDS,
    TICKER_TO_ISSUER,
)
from src.nlp.entity_mapping.schemas import MappedNewsItem


class DictionaryIssuerTickerMapper:
    def __init__(self) -> None:
        alias_pairs: list[tuple[str, str]] = []
        for ticker, aliases in ISSUER_ALIASES.items():
            for alias in aliases:
                alias_pairs.append((alias.lower(), ticker))
        alias_pairs.sort(key=lambda x: len(x[0]), reverse=True)
        self._alias_pairs = alias_pairs

    def map_item(self, item: ProcessedNewsItemRecord) -> MappedNewsItem:
        text = f"{item.title} {item.cleaned_text}".lower()

        for ticker in TICKER_TO_ISSUER:
            if re.search(rf"\b{re.escape(ticker.lower())}\b", text):
                return self._mapped(item, ticker=ticker, method="ticker_exact", confidence=1.0, sector=None)

        for alias, ticker in self._alias_pairs:
            if alias in text:
                return self._mapped(item, ticker=ticker, method="issuer_alias", confidence=0.95, sector=None)

        sector = self._detect_sector(text)
        if sector and sector in SECTOR_FALLBACK_TICKER:
            ticker = SECTOR_FALLBACK_TICKER[sector]
            return self._mapped(item, ticker=ticker, method="sector_fallback", confidence=0.55, sector=sector)

        return self._mapped(item, ticker="UNKNOWN", method="unmapped", confidence=0.0, sector=None)

    @staticmethod
    def _detect_sector(text: str) -> str | None:
        for sector, keywords in SECTOR_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return sector
        return None

    @staticmethod
    def _mapped(
        item: ProcessedNewsItemRecord,
        *,
        ticker: str,
        method: str,
        confidence: float,
        sector: str | None,
    ) -> MappedNewsItem:
        return MappedNewsItem(
            source=item.source,
            title=item.title,
            url=item.url,
            published_at=item.published_at.isoformat(),
            raw_text=item.raw_text,
            cleaned_text=item.cleaned_text,
            ticker=ticker,
            issuer_name=TICKER_TO_ISSUER.get(ticker, "Unknown"),
            sector=sector,
            mapping_method=method,
            mapping_confidence=float(confidence),
        )
