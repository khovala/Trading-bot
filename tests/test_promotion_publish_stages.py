from __future__ import annotations

import json
from pathlib import Path

from src.training.pipeline.base import StageContext, StageSpec
from src.training.stages.evaluation_reporting import PromoteModelStage, PublishArtifactsStage


def _ctx(tmp_path: Path, stage_name: str) -> StageContext:
    return StageContext(
        stage_name=stage_name,
        params={"stages": {}},
        stage_params={"enabled": False},
        run_id="run-promo",
        workspace=tmp_path,
        reports_dir=tmp_path / "reports",
        artifacts_dir=tmp_path / "artifacts",
    )
def test_promote_and_publish_include_gate_reasons(tmp_path: Path) -> None:
    decision_payload = {
        "decision": "promote_candidate",
        "checks": {"promotion_criteria": {"walk_forward_sharpe_mean_min": True}, "stage3_signals": {"ablation_positive_ratio": True}},
        "details": {"walk_forward_sharpe_mean": 1.2, "ablation_positive_ratio": 0.8},
    }
    decision_path = tmp_path / "artifacts" / "comparison" / "decision.json"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(json.dumps(decision_payload), encoding="utf-8")

    promote_result = PromoteModelStage(StageSpec(name="promote_model", purpose="x")).run(_ctx(tmp_path, "promote_model"))
    publish_result = PublishArtifactsStage(StageSpec(name="publish_artifacts", purpose="x")).run(
        _ctx(tmp_path, "publish_artifacts")
    )

    assert promote_result.success is True
    assert publish_result.success is True

    champion = json.loads((tmp_path / "models/registry/champion.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "artifacts/published/bundle_manifest.json").read_text(encoding="utf-8"))

    assert "promotion_checks" in champion
    assert "promotion_details" in champion
    assert "promotion_summary" in manifest
    assert "checks" in manifest["promotion_summary"]
