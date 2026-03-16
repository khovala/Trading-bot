from __future__ import annotations

import json
from pathlib import Path

from src.research.baseline_snapshot import build_baseline_snapshot


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def test_build_baseline_snapshot_collects_metrics(tmp_path: Path) -> None:
    _write_json(tmp_path / "reports/evaluate_models.json", {"metrics": {"directional_accuracy": 0.6}})
    _write_json(tmp_path / "reports/backtest_strategy.json", {"metrics": {"sharpe": 1.2, "turnover": 100.0}})
    _write_json(tmp_path / "reports/compare_with_production.json", {"metrics": {"candidate_sharpe": 1.2}})

    out_path = build_baseline_snapshot(tmp_path)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "metrics" in payload
    assert payload["metrics"]["directional_accuracy"] == 0.6
    assert payload["metrics"]["sharpe"] == 1.2
    assert payload["metrics"]["turnover"] == 100.0
