from __future__ import annotations

from src.backtesting.engine import BacktestConfig, run_backtest
from src.backtesting.metrics import compute_backtest_metrics


def test_backtest_engine_outputs_equity_and_trades() -> None:
    rows = [
        {"timestamp": f"2026-01-01T10:0{i}:00+00:00", "close": 100.0 + i, "signal": 1 if i == 0 else 0}
        for i in range(6)
    ]
    result = run_backtest(rows, BacktestConfig(initial_cash=10000, execution_delay_bars=1, lot_size=1))
    assert len(result["equity_curve"]) == 6
    assert isinstance(result["trade_log"], list)
    metrics = compute_backtest_metrics(
        equity_curve=result["equity_curve"],
        trade_log=result["trade_log"],
        turnover=float(result["summary"]["turnover"]),
        exposure_mean=float(result["summary"]["exposure_mean"]),
    )
    assert "pnl" in metrics
    assert "sharpe" in metrics


def test_backtest_uses_configured_signal_column_not_return_leakage() -> None:
    rows = [
        {
            "timestamp": "2026-01-01T10:00:00+00:00",
            "close": 100.0,
            "expected_return": 0.0,
            "return_1": 1.0,
        },
        {
            "timestamp": "2026-01-01T10:01:00+00:00",
            "close": 99.0,
            "expected_return": 0.0,
            "return_1": 1.0,
        },
    ]
    res = run_backtest(
        rows,
        BacktestConfig(
            initial_cash=10000,
            execution_delay_bars=0,
            lot_size=1,
            signal_column="expected_return",
            signal_threshold=0.0,
        ),
    )
    assert len(res["trade_log"]) == 0
