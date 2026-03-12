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


def refresh_operational_metrics(workspace: Path) -> None:
    backtest_summary = workspace / "artifacts/backtests/backtest_summary.json"
    if backtest_summary.exists():
        payload = json.loads(backtest_summary.read_text(encoding="utf-8"))
        PNL_GAUGE.set(float(payload.get("pnl", 0.0)))
        DRAWDOWN_GAUGE.set(float(payload.get("drawdown", 0.0)))

    map_report = workspace / "reports/map_news_to_instruments.json"
    if map_report.exists():
        payload = json.loads(map_report.read_text(encoding="utf-8"))
        ratio = float(payload.get("metrics", {}).get("mapping_success_ratio", 0.0))
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


def set_stale_data(is_stale: bool) -> None:
    STALE_DATA_GAUGE.set(1.0 if is_stale else 0.0)
