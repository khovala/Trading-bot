from __future__ import annotations

import math
from typing import Any


def _returns(equity_curve: list[dict[str, Any]]) -> list[float]:
    if len(equity_curve) < 2:
        return []
    out: list[float] = []
    for i in range(1, len(equity_curve)):
        prev = float(equity_curve[i - 1]["equity"])
        cur = float(equity_curve[i]["equity"])
        out.append(cur / prev - 1.0 if prev else 0.0)
    return out


def _max_drawdown(equity_curve: list[dict[str, Any]]) -> float:
    peak = 0.0
    max_dd = 0.0
    for row in equity_curve:
        eq = float(row["equity"])
        peak = max(peak, eq)
        if peak > 0:
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)
    return max_dd


def compute_backtest_metrics(
    *,
    equity_curve: list[dict[str, Any]],
    trade_log: list[dict[str, Any]],
    turnover: float,
    exposure_mean: float,
) -> dict[str, float]:
    rets = _returns(equity_curve)
    if rets:
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / len(rets)
        std = math.sqrt(var)
        neg = [r for r in rets if r < 0]
        neg_std = math.sqrt(sum(r * r for r in neg) / len(neg)) if neg else 0.0
    else:
        mean = std = neg_std = 0.0

    sharpe = mean / std * math.sqrt(252) if std > 0 else 0.0
    sortino = mean / neg_std * math.sqrt(252) if neg_std > 0 else 0.0
    max_dd = _max_drawdown(equity_curve)
    total_return = (
        (float(equity_curve[-1]["equity"]) / float(equity_curve[0]["equity"]) - 1.0)
        if len(equity_curve) > 1
        else 0.0
    )
    calmar = total_return / max_dd if max_dd > 0 else 0.0

    wins = sum(1 for t in trade_log if float(t.get("delta_qty", 0)) > 0)
    losses = sum(1 for t in trade_log if float(t.get("delta_qty", 0)) < 0)
    hit_ratio = wins / (wins + losses) if (wins + losses) else 0.0

    pnl = float(equity_curve[-1]["equity"] - equity_curve[0]["equity"]) if equity_curve else 0.0
    return {
        "pnl": pnl,
        "drawdown": max_dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "hit_ratio": hit_ratio,
        "turnover": float(turnover),
        "exposure": float(exposure_mean),
    }
