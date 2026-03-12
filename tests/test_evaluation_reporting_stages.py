from __future__ import annotations

from pathlib import Path

from src.data.feature_store.io import write_parquet
from src.data.feature_store.paths import merged_test_parquet_path, merged_validation_parquet_path
from src.training.pipeline.base import StageContext, StageSpec
from src.training.stages.evaluation_reporting import BacktestStrategyStage, EvaluateModelsStage


def _ctx(tmp_path: Path, stage_name: str) -> StageContext:
    return StageContext(
        stage_name=stage_name,
        params={"features": {"merged": {"schema_version": "v1"}}, "stages": {}, "backtest": {}},
        stage_params={"enabled": False, "walk_forward_folds": 2, "signal_column": "momentum_10"},
        run_id="run-x",
        workspace=tmp_path,
        reports_dir=tmp_path / "reports",
        artifacts_dir=tmp_path / "artifacts",
    )


def test_evaluate_and_backtest_stages_write_reports(tmp_path: Path) -> None:
    rows = [
        {"ticker": "SBER", "timestamp": f"2026-01-01T10:0{i}:00+00:00", "close": 100 + i, "return_1": 0.01, "momentum_10": 0.01}
        for i in range(6)
    ]
    write_parquet(merged_validation_parquet_path(tmp_path), rows)
    write_parquet(merged_test_parquet_path(tmp_path), rows)

    ev = EvaluateModelsStage(StageSpec(name="evaluate_models", purpose="x")).run(_ctx(tmp_path, "evaluate_models"))
    bt = BacktestStrategyStage(StageSpec(name="backtest_strategy", purpose="x")).run(_ctx(tmp_path, "backtest_strategy"))

    assert ev.success is True
    assert bt.success is True
    assert (tmp_path / "reports/evaluate_models.json").exists()
    assert (tmp_path / "reports/backtest_strategy.json").exists()
    assert (tmp_path / "artifacts/backtests/backtest_summary.json").exists()
    assert "walk_forward_sharpe_mean" in bt.metrics
