from __future__ import annotations

import json
from pathlib import Path

from src.data.feature_store.io import write_parquet
from src.data.feature_store.paths import merged_test_parquet_path, merged_validation_parquet_path
from src.models.policy.offline_policy import OfflinePolicyLayer
from src.training.pipeline.base import StageContext, StageSpec
from src.training.stages.evaluation_reporting import (
    BacktestStrategyStage,
    CompareWithProductionStage,
    EvaluateModelsStage,
)


def _ctx(tmp_path: Path, stage_name: str) -> StageContext:
    return StageContext(
        stage_name=stage_name,
        params={
            "features": {"merged": {"schema_version": "v1"}},
            "stages": {},
            "backtest": {},
            "evaluation": {"promotion_criteria": {"walk_forward_sharpe_mean_min": -10.0}},
        },
        stage_params={"enabled": False, "walk_forward_folds": 2, "signal_column": "momentum_10", "min_ablation_positive_ratio": 0.0},
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
    (tmp_path / "artifacts/evaluation").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts/evaluation/ensemble_ablation.json").write_text(
        json.dumps(
            {
                "ensemble_weights": {"a": 0.6, "b": 0.4},
                "ensemble_diagnostics": {"model_stats": {"a": {"avg_confidence": 0.6, "turnover_proxy": 0.1}}},
                "ablation": {
                    "a": {"mean_abs_expected_return_delta": 0.01},
                    "news_feature_model": {"mean_abs_expected_return_delta": 0.02},
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "artifacts/evaluation/policy_layer_summary.json").write_text(
        json.dumps(
            {
                "summary_metrics": {
                    "policy_avg_utility": 0.1,
                    "policy_avg_turnover_proxy": 0.2,
                    "policy_avg_abs_position": 0.3,
                    "decision_samples": 6.0,
                }
            }
        ),
        encoding="utf-8",
    )

    policy = OfflinePolicyLayer(min_confidence=1.1)
    policy.fit(rows)
    policy_path = tmp_path / "models" / "policy" / "offline_policy_layer.pkl"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy.save(policy_path)

    ev = EvaluateModelsStage(StageSpec(name="evaluate_models", purpose="x")).run(_ctx(tmp_path, "evaluate_models"))
    bt = BacktestStrategyStage(StageSpec(name="backtest_strategy", purpose="x")).run(_ctx(tmp_path, "backtest_strategy"))

    assert ev.success is True
    assert bt.success is True
    assert (tmp_path / "reports/evaluate_models.json").exists()
    assert (tmp_path / "reports/evaluate_models_detailed.json").exists()
    assert (tmp_path / "reports/backtest_strategy.json").exists()
    assert (tmp_path / "artifacts/backtests/backtest_summary.json").exists()
    assert "walk_forward_sharpe_mean" in bt.metrics
    assert bt.metrics.get("policy_backtest_enabled") == 1.0
    assert "ablation_positive_ratio" in ev.metrics
    assert "policy_avg_utility" in ev.metrics
    assert "policy_hit_ratio" in ev.metrics
    assert "policy_utility_adjusted_score" in ev.metrics
    assert "policy_turnover_budget_violations" in ev.metrics


def test_compare_stage_uses_stage3_signals_in_decision(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "evaluate_models.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "directional_accuracy": 0.55,
                    "ablation_positive_ratio": 0.8,
                    "ensemble_weight_concentration_hhi": 0.4,
                    "ablation_news_model_delta": 0.01,
                    "policy_avg_utility": 0.2,
                    "policy_avg_turnover_proxy": 0.1,
                }
            }
        ),
        encoding="utf-8",
    )
    (reports / "backtest_strategy.json").write_text(
        json.dumps({"metrics": {"sharpe": 0.2, "pnl": 1000.0, "drawdown": 0.1, "turnover": 0.1}}),
        encoding="utf-8",
    )

    ctx = _ctx(tmp_path, "compare_with_production")
    ctx.stage_params = {
        "promotion_sharpe_threshold": 0.0,
        "min_ablation_positive_ratio": 0.5,
        "max_weight_concentration_hhi": 0.6,
        "min_policy_avg_utility": 0.0,
        "max_policy_turnover_proxy": 0.5,
    }
    result = CompareWithProductionStage(StageSpec(name="compare_with_production", purpose="x")).run(ctx)

    assert result.success is True
    payload = json.loads((tmp_path / "artifacts/comparison/decision.json").read_text(encoding="utf-8"))
    assert "stage3_signals" in payload["checks"]
    assert "policy_avg_utility" in payload["checks"]["stage3_signals"]


def test_evaluate_models_computes_policy_execution_metrics(tmp_path: Path) -> None:
    rows = [
        {"ticker": "SBER", "timestamp": f"2026-01-01T10:0{i}:00+00:00", "close": 100 + i, "return_1": (0.01 if i % 2 == 0 else -0.01), "momentum_10": (0.01 if i % 2 == 0 else -0.01)}
        for i in range(8)
    ]
    write_parquet(merged_validation_parquet_path(tmp_path), rows)
    (tmp_path / "artifacts/evaluation").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts/evaluation/ensemble_ablation.json").write_text(json.dumps({}), encoding="utf-8")
    (tmp_path / "artifacts/evaluation/policy_layer_summary.json").write_text(json.dumps({}), encoding="utf-8")

    policy = OfflinePolicyLayer(min_confidence=0.0)
    policy.fit(rows)
    policy_path = tmp_path / "models" / "policy" / "offline_policy_layer.pkl"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy.save(policy_path)

    ctx = _ctx(tmp_path, "evaluate_models")
    ctx.stage_params = {"turnover_budget_per_step": 0.01, "utility_scale": 50.0}
    ev = EvaluateModelsStage(StageSpec(name="evaluate_models", purpose="x")).run(ctx)

    assert ev.success is True
    assert ev.metrics["policy_active_decisions"] >= 1.0
    assert "policy_turnover_budget_violation_ratio" in ev.metrics
    assert 0.0 <= ev.metrics["policy_turnover_budget_violation_ratio"] <= 1.0
