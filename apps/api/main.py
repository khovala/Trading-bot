from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.config.settings import get_settings
from src.domain.enums import RunMode
from src.monitoring.metrics import (
    MODEL_CONFIDENCE,
    ORDER_SUBMISSION_TOTAL,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    SIGNAL_COUNT,
    refresh_operational_metrics,
)

APP_VERSION = "0.1.0"

app = FastAPI(title="MOEX Sandbox Platform API", version=APP_VERSION)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    started = perf_counter()
    path = request.url.path
    method = request.method
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        latency = perf_counter() - started
        REQUEST_COUNT.labels(path=path, method=method, status=str(status_code)).inc()
        REQUEST_LATENCY.labels(path=path, method=method).observe(latency)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    settings = get_settings()
    if settings.trading_mode == RunMode.LIVE:
        if not settings.t_invest_token or not settings.t_invest_account_id:
            return {"status": "not_ready", "mode": settings.trading_mode.value}
    return {"status": "ready", "mode": settings.trading_mode.value}


@app.get("/models")
def models() -> dict[str, list[dict[str, str]]]:
    return {"models": []}


@app.get("/signals/latest")
def latest_signal() -> dict[str, str]:
    SIGNAL_COUNT.labels(action="HOLD").inc()
    MODEL_CONFIDENCE.labels(model="orchestrator").set(0.5)
    return {"status": "not_implemented"}


@app.post("/trade/run-once")
def trade_run_once() -> JSONResponse:
    ORDER_SUBMISSION_TOTAL.labels(status="accepted").inc()
    return JSONResponse({"status": "accepted", "mode": "dry_run_placeholder"}, status_code=202)


@app.post("/retrain/run-once")
def retrain_run_once() -> JSONResponse:
    return JSONResponse({"status": "accepted", "pipeline": "placeholder"}, status_code=202)


@app.get("/backtests/latest")
def backtests_latest() -> dict[str, str]:
    return {"status": "not_implemented"}


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    refresh_operational_metrics(Path("."))
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "moex-sandbox-platform-api", "ts": datetime.now(timezone.utc).isoformat()}
