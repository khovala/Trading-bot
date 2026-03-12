from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from src.data.ingestion.news.base import NewsSourceAdapter
from src.data.schemas.news import RawNewsItemRecord


def parse_published_at(value: str | None) -> datetime:
    if not value:
        return datetime.now(tz=timezone.utc)
    cleaned = value.strip().replace("Z", "+00:00")
    try:
        ts = datetime.fromisoformat(cleaned)
    except ValueError:
        ts = parsedate_to_datetime(cleaned)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


class RssNewsSourceAdapter(NewsSourceAdapter):
    def __init__(self, source_name: str, feed_url: str, fetcher=None) -> None:
        self.source_name = source_name
        self.feed_url = feed_url
        self._fetcher = fetcher or self._default_fetcher

    @staticmethod
    def _default_fetcher(feed_url: str) -> str:
        req = Request(feed_url, headers={"User-Agent": "moex-sandbox-bot/0.1"})
        with urlopen(req, timeout=10) as resp:  # nosec B310
            return resp.read().decode("utf-8", errors="ignore")

    def fetch(self, limit: int = 200) -> list[RawNewsItemRecord]:
        payload = self._fetcher(self.feed_url)
        root = ET.fromstring(payload)
        items: list[RawNewsItemRecord] = []
        for item in _iter_feed_items(root):
            title = (item.findtext("title") or "").strip()
            if not title:
                continue
            description = (item.findtext("description") or item.findtext("summary") or "").strip()
            link = _extract_link(item)
            published = parse_published_at(
                item.findtext("pubDate")
                or item.findtext("published")
                or item.findtext("updated")
            )
            raw_text = " ".join(part for part in [title, description] if part).strip()
            items.append(
                RawNewsItemRecord(
                    source=self.source_name,
                    title=title,
                    url=link,
                    published_at=published,
                    raw_text=raw_text or title,
                    snippet=description or None,
                    body=None,
                )
            )
            if len(items) >= limit:
                break
        return items


def _iter_feed_items(root: ET.Element) -> list[ET.Element]:
    rss_items = root.findall(".//item")
    if rss_items:
        return rss_items
    return root.findall(".//{*}entry")


def _extract_link(item: ET.Element) -> str | None:
    link_text = (item.findtext("link") or "").strip()
    if link_text:
        return link_text
    for link in item.findall("{*}link"):
        href = (link.attrib.get("href") or "").strip()
        if href:
            return href
    return None


def build_news_source_adapter(source_name: str, feed_url: str) -> NewsSourceAdapter:
    return RssNewsSourceAdapter(source_name=source_name, feed_url=feed_url)


class TelegramNewsSourceAdapterPhase2(NewsSourceAdapter):
    """Phase-2 placeholder: Telegram ingestion intentionally not implemented."""

    def fetch(self, limit: int = 200) -> list[RawNewsItemRecord]:
        return []
