from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class BacktestConfig:
    initial_cash: float = 1_000_000.0
    commission_bps: float = 5.0
    slippage_bps: float = 5.0
    lot_size: int = 1
    execution_delay_bars: int = 1
    signal_threshold: float = 0.0
    position_size_pct: float = 0.2
    signal_column: str = "expected_return"


def _signal_from_row(row: dict[str, Any], threshold: float, signal_column: str) -> int:
    raw = float(
        row.get(
            signal_column,
            row.get("signal", row.get("expected_return", row.get("momentum_10", 0.0))),
        )
    )
    if raw > threshold:
        return 1
    if raw < -threshold:
        return -1
    return 0


def run_backtest(rows: list[dict[str, Any]], cfg: BacktestConfig) -> dict[str, Any]:
    if not rows:
        return {
            "equity_curve": [],
            "trade_log": [],
            "summary": {"pnl": 0.0, "turnover": 0.0, "exposure_mean": 0.0},
        }
    ordered = sorted(rows, key=lambda x: x["timestamp"])
    initial_price = float(ordered[0]["close"])

    cash = cfg.initial_cash
    position = 0
    pending_signal: tuple[int, int] | None = None
    trade_log: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    turnover = 0.0
    exposures: list[float] = []

    for i, row in enumerate(ordered):
        ts = str(row["timestamp"])
        price = float(row["close"])
        signal = _signal_from_row(row, cfg.signal_threshold, cfg.signal_column)
        if pending_signal is None and signal != 0:
            pending_signal = (signal, i + cfg.execution_delay_bars)

        if pending_signal and i >= pending_signal[1]:
            exec_signal = pending_signal[0]
            pending_signal = None
            target_notional = max(0.0, cash * cfg.position_size_pct)
            raw_qty = int(target_notional / max(1e-9, price))
            qty = max(0, (raw_qty // max(1, cfg.lot_size)) * max(1, cfg.lot_size))
            if exec_signal < 0:
                qty = -qty

            delta = qty - position
            if delta != 0:
                fill_price = price * (1.0 + (cfg.slippage_bps / 10_000.0) * (1.0 if delta > 0 else -1.0))
                notional = abs(delta) * fill_price
                commission = notional * (cfg.commission_bps / 10_000.0)
                cash -= delta * fill_price
                cash -= commission
                turnover += notional
                position = qty
                trade_log.append(
                    {
                        "timestamp": ts,
                        "price": float(fill_price),
                        "delta_qty": int(delta),
                        "position": int(position),
                        "commission": float(commission),
                    }
                )

        equity = cash + position * price
        benchmark_equity = cfg.initial_cash * (price / initial_price if initial_price else 1.0)
        exposure = abs(position * price) / max(1e-9, equity)
        exposures.append(exposure)
        equity_curve.append(
            {
                "timestamp": ts,
                "equity": float(equity),
                "benchmark_equity": float(benchmark_equity),
                "exposure": float(exposure),
            }
        )

    return {
        "equity_curve": equity_curve,
        "trade_log": trade_log,
        "summary": {
            "pnl": float(equity_curve[-1]["equity"] - cfg.initial_cash),
            "turnover": float(turnover),
            "exposure_mean": float(sum(exposures) / len(exposures) if exposures else 0.0),
        },
    }
