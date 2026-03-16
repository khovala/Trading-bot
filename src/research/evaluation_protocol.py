from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluationProtocol:
    walk_forward_folds: int = 5
    purged_cv_enabled: bool = True
    embargo_minutes: int = 60
    regime_wise_evaluation: bool = True
    stress_cost_multipliers: tuple[float, ...] = (1.0, 1.5, 2.0)
    stress_slippage_multipliers: tuple[float, ...] = (1.0, 1.5, 2.0)


@dataclass(frozen=True, slots=True)
class PromotionCriteria:
    walk_forward_sharpe_mean_min: float = 1.0
    max_drawdown_max: float = 0.15
    min_positive_folds_ratio: float = 0.5
    min_turnover_reduction_vs_baseline: float = 0.50
    require_news_ablation_benefit: bool = False
    require_cost_stress_pass: bool = True
    require_calibration_report: bool = True
    require_leakage_checks: bool = True


@dataclass(frozen=True, slots=True)
class AcceptanceCheckResult:
    passed: bool
    checks: dict[str, bool]
    details: dict[str, float]


def _tuple_of_floats(value: Any, default: tuple[float, ...]) -> tuple[float, ...]:
    if not isinstance(value, list):
        return default
    out: list[float] = []
    for item in value:
        try:
            out.append(float(item))
        except Exception:
            continue
    return tuple(out) if out else default


def load_evaluation_protocol(params: dict[str, Any]) -> EvaluationProtocol:
    cfg = params.get("evaluation", {}).get("protocol", {})
    if not isinstance(cfg, dict):
        cfg = {}
    return EvaluationProtocol(
        walk_forward_folds=max(1, int(cfg.get("walk_forward_folds", 5))),
        purged_cv_enabled=bool(cfg.get("purged_cv_enabled", True)),
        embargo_minutes=max(0, int(cfg.get("embargo_minutes", 60))),
        regime_wise_evaluation=bool(cfg.get("regime_wise_evaluation", True)),
        stress_cost_multipliers=_tuple_of_floats(
            cfg.get("stress_cost_multipliers"),
            (1.0, 1.5, 2.0),
        ),
        stress_slippage_multipliers=_tuple_of_floats(
            cfg.get("stress_slippage_multipliers"),
            (1.0, 1.5, 2.0),
        ),
    )


def load_promotion_criteria(params: dict[str, Any]) -> PromotionCriteria:
    cfg = params.get("evaluation", {}).get("promotion_criteria", {})
    if not isinstance(cfg, dict):
        cfg = {}
    return PromotionCriteria(
        walk_forward_sharpe_mean_min=float(cfg.get("walk_forward_sharpe_mean_min", 1.0)),
        max_drawdown_max=float(cfg.get("max_drawdown_max", 0.15)),
        min_positive_folds_ratio=float(cfg.get("min_positive_folds_ratio", 0.5)),
        min_turnover_reduction_vs_baseline=float(cfg.get("min_turnover_reduction_vs_baseline", 0.50)),
        require_news_ablation_benefit=bool(cfg.get("require_news_ablation_benefit", False)),
        require_cost_stress_pass=bool(cfg.get("require_cost_stress_pass", True)),
        require_calibration_report=bool(cfg.get("require_calibration_report", True)),
        require_leakage_checks=bool(cfg.get("require_leakage_checks", True)),
    )


def _safe_float(metrics: dict[str, float], key: str, default: float = 0.0) -> float:
    try:
        return float(metrics.get(key, default))
    except Exception:
        return default


def check_promotion_criteria(
    *,
    metrics: dict[str, float],
    criteria: PromotionCriteria,
    baseline_metrics: dict[str, float] | None = None,
    positive_fold_count: int | None = None,
    total_fold_count: int | None = None,
    cost_stress_passed: bool | None = None,
    has_calibration_report: bool | None = None,
    has_leakage_report: bool | None = None,
    news_ablation_benefit: float | None = None,
) -> AcceptanceCheckResult:
    baseline_metrics = baseline_metrics or {}
    pnl = _safe_float(metrics, "pnl")
    dd = _safe_float(metrics, "drawdown", default=1.0)
    wf_sharpe = _safe_float(metrics, "walk_forward_sharpe_mean")
    turnover = _safe_float(metrics, "turnover")
    baseline_turnover = _safe_float(baseline_metrics, "turnover", default=turnover if turnover > 0 else 1.0)
    turnover_reduction = 1.0 - (turnover / baseline_turnover if baseline_turnover > 0 else 1.0)

    if total_fold_count and total_fold_count > 0 and positive_fold_count is not None:
        positive_fold_ratio = float(positive_fold_count) / float(total_fold_count)
    else:
        positive_fold_ratio = 1.0 if pnl > 0 else 0.0

    checks = {
        "walk_forward_sharpe_mean_min": wf_sharpe >= criteria.walk_forward_sharpe_mean_min,
        "max_drawdown_max": dd <= criteria.max_drawdown_max,
        "min_positive_folds_ratio": positive_fold_ratio >= criteria.min_positive_folds_ratio,
        "min_turnover_reduction_vs_baseline": turnover_reduction >= criteria.min_turnover_reduction_vs_baseline,
    }

    if criteria.require_news_ablation_benefit:
        checks["news_ablation_benefit"] = (news_ablation_benefit or 0.0) > 0.0
    if criteria.require_cost_stress_pass:
        checks["cost_stress_passed"] = bool(cost_stress_passed)
    if criteria.require_calibration_report:
        checks["has_calibration_report"] = bool(has_calibration_report)
    if criteria.require_leakage_checks:
        checks["has_leakage_report"] = bool(has_leakage_report)

    details = {
        "pnl": pnl,
        "drawdown": dd,
        "walk_forward_sharpe_mean": wf_sharpe,
        "turnover": turnover,
        "baseline_turnover": baseline_turnover,
        "turnover_reduction_vs_baseline": turnover_reduction,
        "positive_fold_ratio": positive_fold_ratio,
    }
    return AcceptanceCheckResult(
        passed=all(checks.values()),
        checks=checks,
        details=details,
    )
