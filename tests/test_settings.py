from __future__ import annotations

import pytest

from src.config.settings import Settings


def test_live_mode_requires_explicit_enable_and_no_dry_run() -> None:
    with pytest.raises(ValueError):
        Settings(TRADING_MODE="live", REAL_TRADING_ENABLED=False, DRY_RUN=False)

    with pytest.raises(ValueError):
        Settings(TRADING_MODE="live", REAL_TRADING_ENABLED=True, DRY_RUN=True)


def test_sandbox_rejects_real_trading_flag() -> None:
    with pytest.raises(ValueError):
        Settings(TRADING_MODE="sandbox", REAL_TRADING_ENABLED=True, DRY_RUN=True)
