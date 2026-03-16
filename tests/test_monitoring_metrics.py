from __future__ import annotations

import json

from prometheus_client import generate_latest

from src.monitoring.metrics import refresh_operational_metrics


def test_refresh_operational_metrics_exports_pipeline_and_backtest_gauges(tmp_path) -> None:
    (tmp_path / "artifacts/backtests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports/final").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts/comparison").mkdir(parents=True, exist_ok=True)

    (tmp_path / "artifacts/backtests/backtest_summary.json").write_text(
        json.dumps(
            {
                "pnl": 123.0,
                "drawdown": 0.1,
                "sharpe": 1.2,
                "sortino": 1.4,
                "calmar": 0.9,
                "hit_ratio": 0.55,
                "turnover": 1000.0,
                "exposure": 0.3,
                "walk_forward_sharpe_mean": 0.8,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "reports/evaluate_models.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "directional_accuracy": 0.61,
                    "mae_proxy": 0.02,
                    "policy_hit_ratio": 0.57,
                    "policy_utility_adjusted_score": 0.71,
                    "policy_turnover_budget_violation_ratio": 0.08,
                    "ensemble_weight_concentration_hhi": 0.22,
                    "ablation_positive_ratio": 0.75,
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "artifacts/comparison/decision.json").write_text(
        json.dumps({"decision": "promote_candidate"}),
        encoding="utf-8",
    )
    (tmp_path / "reports/final/retrain_report.json").write_text(
        json.dumps({"reports": {"a.json": {}, "b.json": {}}}),
        encoding="utf-8",
    )

    refresh_operational_metrics(tmp_path)
    payload = generate_latest().decode("utf-8")

    for metric_name in (
        "strategy_pnl_rub",
        "strategy_sharpe",
        "strategy_turnover",
        "model_directional_accuracy",
        "policy_utility_adjusted_score",
        "ensemble_weight_concentration_hhi",
        "promotion_decision_flag",
        "pipeline_report_count",
        "pipeline_last_update_unix",
    ):
        assert metric_name in payload
