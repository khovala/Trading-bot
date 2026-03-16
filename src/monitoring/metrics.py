from __future__ import annotations

import json
from pathlib import Path
from time import time

from prometheus_client import Counter, Gauge, Histogram


REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["path", "method", "status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "API request latency", ["path", "method"])
INFERENCE_LATENCY = Histogram("inference_latency_seconds", "Inference latency by model", ["model"])
MODEL_CONFIDENCE = Gauge("model_confidence", "Latest model confidence", ["model"])
SIGNAL_COUNT = Counter("signal_count_total", "Signals emitted by action", ["action"])
ORDER_SUBMISSION_TOTAL = Counter("order_submission_total", "Order submissions", ["status"])
ORDER_FAILURE_TOTAL = Counter("order_failure_total", "Order failures", ["reason"])
PNL_GAUGE = Gauge("strategy_pnl_rub", "Strategy pnl in RUB")
DRAWDOWN_GAUGE = Gauge("strategy_drawdown", "Current drawdown")
STALE_DATA_GAUGE = Gauge("stale_data_flag", "Stale data flag (1 stale)")
NEWS_FRESHNESS_SECONDS = Gauge("news_freshness_seconds", "Seconds since last news item")
NEWS_ITEMS_BY_SOURCE = Gauge("news_items_by_source", "News item count by source", ["source"])
TICKER_MAPPING_SUCCESS_RATIO = Gauge("ticker_mapping_success_ratio", "Ticker mapping success ratio")
STRATEGY_SHARPE = Gauge("strategy_sharpe", "Backtest Sharpe ratio")
STRATEGY_SORTINO = Gauge("strategy_sortino", "Backtest Sortino ratio")
STRATEGY_CALMAR = Gauge("strategy_calmar", "Backtest Calmar ratio")
STRATEGY_HIT_RATIO = Gauge("strategy_hit_ratio", "Backtest hit ratio")
STRATEGY_TURNOVER = Gauge("strategy_turnover", "Backtest turnover")
STRATEGY_EXPOSURE = Gauge("strategy_exposure", "Backtest exposure")
WALK_FORWARD_SHARPE_MEAN = Gauge("walk_forward_sharpe_mean", "Walk-forward average Sharpe ratio")
MODEL_DIRECTIONAL_ACCURACY = Gauge("model_directional_accuracy", "Validation directional accuracy")
MODEL_MAE_PROXY = Gauge("model_mae_proxy", "Validation MAE proxy")
POLICY_HIT_RATIO = Gauge("policy_hit_ratio", "Policy decision hit ratio on validation")
POLICY_UTILITY_ADJUSTED_SCORE = Gauge("policy_utility_adjusted_score", "Policy utility-adjusted score")
POLICY_TURNOVER_BUDGET_VIOLATION_RATIO = Gauge(
    "policy_turnover_budget_violation_ratio",
    "Policy turnover budget violation ratio",
)
ENSEMBLE_WEIGHT_CONCENTRATION_HHI = Gauge(
    "ensemble_weight_concentration_hhi",
    "Ensemble weight concentration HHI",
)
ENSEMBLE_ABLATION_POSITIVE_RATIO = Gauge(
    "ensemble_ablation_positive_ratio",
    "Ratio of positive ablation contributions",
)
PROMOTION_DECISION_FLAG = Gauge(
    "promotion_decision_flag",
    "Promotion decision flag (1=promote_candidate, 0=keep_champion)",
)
PIPELINE_REPORT_COUNT = Gauge("pipeline_report_count", "Final report count from generate_reports")
PIPELINE_LAST_UPDATE_UNIX = Gauge("pipeline_last_update_unix", "Last pipeline artifact update timestamp")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def refresh_operational_metrics(workspace: Path) -> None:
    backtest_summary = workspace / "artifacts/backtests/backtest_summary.json"
    if backtest_summary.exists():
        payload = _load_json(backtest_summary)
        PNL_GAUGE.set(_safe_float(payload.get("pnl", 0.0)))
        DRAWDOWN_GAUGE.set(_safe_float(payload.get("drawdown", 0.0)))
        STRATEGY_SHARPE.set(_safe_float(payload.get("sharpe", 0.0)))
        STRATEGY_SORTINO.set(_safe_float(payload.get("sortino", 0.0)))
        STRATEGY_CALMAR.set(_safe_float(payload.get("calmar", 0.0)))
        STRATEGY_HIT_RATIO.set(_safe_float(payload.get("hit_ratio", 0.0)))
        STRATEGY_TURNOVER.set(_safe_float(payload.get("turnover", 0.0)))
        STRATEGY_EXPOSURE.set(_safe_float(payload.get("exposure", 0.0)))
        WALK_FORWARD_SHARPE_MEAN.set(_safe_float(payload.get("walk_forward_sharpe_mean", 0.0)))

    map_report = workspace / "reports/map_news_to_instruments.json"
    if map_report.exists():
        payload = _load_json(map_report)
        ratio = _safe_float(payload.get("metrics", {}).get("mapping_success_ratio", 0.0))
        TICKER_MAPPING_SUCCESS_RATIO.set(ratio)

    raw_news = workspace / "data/raw/news/items.jsonl"
    if raw_news.exists():
        counts: dict[str, int] = {}
        latest_ts = None
        for line in raw_news.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            src = str(row.get("source", "unknown")).lower()
            counts[src] = counts.get(src, 0) + 1
            latest_ts = row.get("published_at", latest_ts)
        for src, cnt in counts.items():
            NEWS_ITEMS_BY_SOURCE.labels(source=src).set(float(cnt))
        if latest_ts:
            from datetime import datetime, timezone

            ts = datetime.fromisoformat(str(latest_ts).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            NEWS_FRESHNESS_SECONDS.set(max(0.0, time() - ts.timestamp()))

    eval_report = _load_json(workspace / "reports/evaluate_models.json")
    eval_metrics = eval_report.get("metrics", {}) if isinstance(eval_report.get("metrics"), dict) else {}
    MODEL_DIRECTIONAL_ACCURACY.set(_safe_float(eval_metrics.get("directional_accuracy", 0.0)))
    MODEL_MAE_PROXY.set(_safe_float(eval_metrics.get("mae_proxy", 0.0)))
    POLICY_HIT_RATIO.set(_safe_float(eval_metrics.get("policy_hit_ratio", 0.0)))
    POLICY_UTILITY_ADJUSTED_SCORE.set(_safe_float(eval_metrics.get("policy_utility_adjusted_score", 0.0)))
    POLICY_TURNOVER_BUDGET_VIOLATION_RATIO.set(
        _safe_float(eval_metrics.get("policy_turnover_budget_violation_ratio", 0.0))
    )
    ENSEMBLE_WEIGHT_CONCENTRATION_HHI.set(_safe_float(eval_metrics.get("ensemble_weight_concentration_hhi", 0.0)))
    ENSEMBLE_ABLATION_POSITIVE_RATIO.set(_safe_float(eval_metrics.get("ablation_positive_ratio", 0.0)))

    decision_report = _load_json(workspace / "artifacts/comparison/decision.json")
    decision_value = 1.0 if str(decision_report.get("decision", "")) == "promote_candidate" else 0.0
    PROMOTION_DECISION_FLAG.set(decision_value)

    final_report = _load_json(workspace / "reports/final/retrain_report.json")
    if isinstance(final_report.get("reports"), dict):
        PIPELINE_REPORT_COUNT.set(float(len(final_report.get("reports", {}))))
    latest_mtime = 0.0
    for candidate in (
        workspace / "reports/final/retrain_report.json",
        workspace / "reports/backtest_strategy.json",
        workspace / "reports/evaluate_models.json",
        workspace / "artifacts/backtests/backtest_summary.json",
    ):
        if candidate.exists():
            latest_mtime = max(latest_mtime, candidate.stat().st_mtime)
    PIPELINE_LAST_UPDATE_UNIX.set(latest_mtime)


def set_stale_data(is_stale: bool) -> None:
    STALE_DATA_GAUGE.set(1.0 if is_stale else 0.0)
