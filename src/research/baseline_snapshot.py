from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def _load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def _extract_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in metrics.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


def build_baseline_snapshot(
    workspace: Path,
    *,
    output_path: Path | None = None,
    source_reports: tuple[str, ...] = (
        "reports/evaluate_models.json",
        "reports/backtest_strategy.json",
        "reports/compare_with_production.json",
    ),
) -> Path:
    workspace = workspace.resolve()
    output_path = output_path or (workspace / "reports" / "baseline_snapshot.json")
    snapshot: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_reports": list(source_reports),
        "metrics": {},
    }

    metrics_out: dict[str, float] = {}
    for rel in source_reports:
        payload = _load_report(workspace / rel)
        for key, value in _extract_metrics(payload).items():
            if key not in metrics_out:
                metrics_out[key] = value
            else:
                metrics_out[f"{Path(rel).stem}_{key}"] = value

    snapshot["metrics"] = metrics_out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return output_path
