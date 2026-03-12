from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def raw_market_root(workspace: Path) -> Path:
    return workspace / "data/raw/market"


def interim_market_root(workspace: Path) -> Path:
    return workspace / "data/interim/market"


def raw_instruments_path(workspace: Path) -> Path:
    return raw_market_root(workspace) / "instruments.json"


def raw_candles_path(workspace: Path, ticker: str) -> Path:
    return raw_market_root(workspace) / "candles" / f"{ticker.upper()}.jsonl"


def interim_candles_path(workspace: Path, ticker: str) -> Path:
    return interim_market_root(workspace) / "candles" / f"{ticker.upper()}.jsonl"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, ensure_ascii=True) for row in rows)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
