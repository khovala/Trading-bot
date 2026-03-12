from __future__ import annotations

from pathlib import Path
from typing import Any


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

    return pq.read_table(path).to_pylist()
