from __future__ import annotations

import json
from pathlib import Path

from src.domain.schemas import StageResult
from src.training.pipeline.base import StageSpec


def write_stage_reports(spec: StageSpec, result: StageResult, workspace: Path) -> list[Path]:
    written: list[Path] = []
    payload = {
        "run_id": result.run_id,
        "stage_name": result.stage_name,
        "success": result.success,
        "metrics": result.metrics,
        "artifacts": result.artifacts,
    }
    for relative_path in spec.reports:
        target = (workspace / relative_path).resolve()
        root = workspace.resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Report path escapes workspace: {relative_path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix == ".json":
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            written.append(target)
        elif target.suffix == ".md":
            target.write_text(
                "\n".join(
                    [
                        "# Stage Report",
                        "",
                        f"- run_id: `{result.run_id}`",
                        f"- stage: `{result.stage_name}`",
                        f"- success: `{result.success}`",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            written.append(target)
    return written
