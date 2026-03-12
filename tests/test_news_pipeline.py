from __future__ import annotations

from datetime import timezone

from src.data.ingestion.news.rss import RssNewsSourceAdapter, parse_published_at
from src.data.preprocessing.news.cleaning import clean_text, deduplicate_items
from src.data.schemas.news import RawNewsItemRecord


def _item(*, title: str, url: str | None, published_at: str) -> RawNewsItemRecord:
    return RawNewsItemRecord(
        source="rbc_rss",
        title=title,
        url=url,
        published_at=published_at,
        raw_text=title,
    )


def test_duplicate_removal_prefers_unique_url_or_hash() -> None:
    items = [
        _item(title="A", url="https://example.com/1", published_at="2026-01-01T10:00:00+00:00"),
        _item(title="A2", url="https://example.com/1", published_at="2026-01-01T10:01:00+00:00"),
        _item(title="B", url=None, published_at="2026-01-01T10:00:00+00:00"),
        _item(title="B", url=None, published_at="2026-01-01T10:00:30+00:00"),
    ]
    deduped = deduplicate_items(items)
    assert len(deduped) == 2


def test_timestamp_parsing_rfc822_and_iso() -> None:
    ts_rss = parse_published_at("Wed, 11 Mar 2026 10:20:00 +0300")
    ts_iso = parse_published_at("2026-03-11T07:20:00Z")
    assert ts_rss.tzinfo == timezone.utc
    assert ts_iso.tzinfo == timezone.utc
    assert ts_rss == ts_iso


def test_basic_text_cleaning() -> None:
    text = "<p>Hello   world</p>\n\n  from <b>MOEX</b>"
    assert clean_text(text) == "Hello world from MOEX"


def test_rss_adapter_parses_atom_entries() -> None:
    atom_payload = """
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>Atom headline</title>
        <summary>Body text</summary>
        <link href="https://example.com/a1" />
        <updated>2026-03-11T07:20:00Z</updated>
      </entry>
    </feed>
    """
    adapter = RssNewsSourceAdapter(
        source_name="atom_test",
        feed_url="https://example.com/feed",
        fetcher=lambda _: atom_payload,
    )
    items = adapter.fetch(limit=10)
    assert len(items) == 1
    assert items[0].url == "https://example.com/a1"
