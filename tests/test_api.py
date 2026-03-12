from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.main import app
from src.config.settings import get_settings


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_returns_not_ready_for_live_without_credentials(monkeypatch) -> None:
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("REAL_TRADING_ENABLED", "true")
    monkeypatch.setenv("DRY_RUN", "false")
    monkeypatch.delenv("T_INVEST_TOKEN", raising=False)
    monkeypatch.delenv("T_INVEST_ACCOUNT_ID", raising=False)
    get_settings.cache_clear()

    client = TestClient(app)
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "not_ready"

    get_settings.cache_clear()
