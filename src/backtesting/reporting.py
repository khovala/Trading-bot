from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.market_store.layout import write_json


def write_backtest_outputs(
    *,
    out_dir: Path,
    summary: dict[str, Any],
    equity_curve: list[dict[str, Any]],
    trade_log: list[dict[str, Any]],
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    drawdown_curve: list[dict[str, Any]] = []
    peak = 0.0
    for row in equity_curve:
        eq = float(row["equity"])
        peak = max(peak, eq)
        dd = (peak - eq) / peak if peak > 0 else 0.0
        drawdown_curve.append({"timestamp": row["timestamp"], "drawdown": dd})

    rolling_perf = []
    for i in range(1, len(equity_curve)):
        prev = float(equity_curve[i - 1]["equity"])
        cur = float(equity_curve[i]["equity"])
        rolling_perf.append({"timestamp": equity_curve[i]["timestamp"], "ret_1": cur / prev - 1.0 if prev else 0.0})

    trade_distribution = {"count": len(trade_log), "sizes": [abs(int(t.get("delta_qty", 0))) for t in trade_log]}

    write_json(out_dir / "backtest_summary.json", summary)
    write_json(out_dir / "equity_curve.json", equity_curve)
    write_json(out_dir / "trade_log.json", trade_log)
    write_json(plots_dir / "equity_vs_benchmark.json", equity_curve)
    write_json(plots_dir / "drawdown_curve.json", drawdown_curve)
    write_json(plots_dir / "rolling_performance.json", rolling_perf)
    write_json(plots_dir / "trade_distribution.json", trade_distribution)
    return [
        "artifacts/backtests/backtest_summary.json",
        "artifacts/backtests/equity_curve.json",
        "artifacts/backtests/trade_log.json",
        "artifacts/backtests/plots/equity_vs_benchmark.json",
        "artifacts/backtests/plots/drawdown_curve.json",
        "artifacts/backtests/plots/rolling_performance.json",
        "artifacts/backtests/plots/trade_distribution.json",
    ]
