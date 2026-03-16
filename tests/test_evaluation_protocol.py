from __future__ import annotations

from src.research.evaluation_protocol import (
    PromotionCriteria,
    check_promotion_criteria,
    load_evaluation_protocol,
)


def test_load_evaluation_protocol_defaults() -> None:
    protocol = load_evaluation_protocol({})
    assert protocol.walk_forward_folds == 5
    assert protocol.purged_cv_enabled
    assert protocol.embargo_minutes == 60


def test_check_promotion_criteria_passes_when_metrics_meet_thresholds() -> None:
    criteria = PromotionCriteria(
        walk_forward_sharpe_mean_min=0.5,
        max_drawdown_max=0.2,
        min_positive_folds_ratio=0.5,
        min_turnover_reduction_vs_baseline=0.2,
        require_cost_stress_pass=True,
        require_calibration_report=True,
        require_leakage_checks=True,
    )
    result = check_promotion_criteria(
        metrics={"pnl": 10.0, "drawdown": 0.1, "walk_forward_sharpe_mean": 0.8, "turnover": 100.0},
        baseline_metrics={"turnover": 200.0},
        criteria=criteria,
        positive_fold_count=3,
        total_fold_count=4,
        cost_stress_passed=True,
        has_calibration_report=True,
        has_leakage_report=True,
    )
    assert result.passed
    assert all(result.checks.values())


def test_check_promotion_criteria_fails_on_drawdown_and_turnover() -> None:
    criteria = PromotionCriteria(
        walk_forward_sharpe_mean_min=1.0,
        max_drawdown_max=0.15,
        min_positive_folds_ratio=0.5,
        min_turnover_reduction_vs_baseline=0.5,
    )
    result = check_promotion_criteria(
        metrics={"pnl": -1.0, "drawdown": 0.4, "walk_forward_sharpe_mean": 0.1, "turnover": 1000.0},
        baseline_metrics={"turnover": 1000.0},
        criteria=criteria,
        positive_fold_count=1,
        total_fold_count=5,
    )
    assert not result.passed
    assert not result.checks["max_drawdown_max"]
    assert not result.checks["min_turnover_reduction_vs_baseline"]
