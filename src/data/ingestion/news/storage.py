from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def raw_news_items_path(workspace: Path) -> Path:
    return workspace / "data/raw/news/items.jsonl"


def interim_news_items_path(workspace: Path) -> Path:
    return workspace / "data/interim/news/items.jsonl"


def mapped_news_parquet_path(workspace: Path) -> Path:
    return workspace / "data/processed/news/mapped_news.parquet"


def news_features_parquet_path(workspace: Path) -> Path:
    return workspace / "data/processed/news/features/news_features.parquet"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, ensure_ascii=True) for row in rows)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({}) if not rows else pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def read_parquet(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return table.to_pylist()
