from __future__ import annotations

from pathlib import Path

from src.data.feature_store.io import write_parquet
from src.data.feature_store.paths import merged_train_parquet_path
from src.training.pipeline.base import StageContext, StageSpec
from src.training.stages.policy_layer import TrainPolicyLayerStage


def _ctx(tmp_path: Path, stage_name: str) -> StageContext:
    return StageContext(
        stage_name=stage_name,
        params={
            "models_v2": {
                "policy_layer": {
                    "risk_aversion": 2.0,
                    "turnover_penalty": 0.2,
                    "drawdown_penalty": 0.1,
                    "uncertainty_penalty": 0.2,
                    "max_position": 1.0,
                    "min_confidence": 0.5,
                }
            },
            "stages": {},
        },
        stage_params={"enabled": False},
        run_id="policy-run",
        workspace=tmp_path,
        reports_dir=tmp_path / "reports",
        artifacts_dir=tmp_path / "artifacts",
    )


def test_train_policy_layer_stage_outputs_artifacts(tmp_path: Path) -> None:
    rows = [
        {"ticker": "SBER", "timestamp": "2026-01-01T10:00:00+00:00", "return_1": 0.01, "rolling_volatility_20": 0.2},
        {"ticker": "SBER", "timestamp": "2026-01-01T11:00:00+00:00", "return_1": -0.02, "rolling_volatility_20": 0.3},
    ]
    write_parquet(merged_train_parquet_path(tmp_path), rows)

    result = TrainPolicyLayerStage(StageSpec(name="train_policy_layer", purpose="x")).run(_ctx(tmp_path, "train_policy_layer"))

    assert result.success is True
    assert (tmp_path / "models/policy/offline_policy_layer.pkl").exists()
    assert (tmp_path / "models/policy/metadata.json").exists()
    assert (tmp_path / "artifacts/evaluation/policy_layer_summary.json").exists()
