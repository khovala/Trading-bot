from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.market_store.layout import write_json


def _build_histogram(values: list[int]) -> tuple[list[int], list[int]]:
    if not values:
        return ([], [])
    vmax = max(values)
    if vmax <= 0:
        return ([0], [len(values)])
    bins = max(5, min(25, int(len(values) ** 0.5)))
    width = max(1, (vmax + bins - 1) // bins)
    edges = list(range(0, vmax + width + 1, width))
    counts = [0 for _ in range(len(edges) - 1)]
    for v in values:
        idx = min(len(counts) - 1, max(0, v // width))
        counts[idx] += 1
    centers = [int((edges[i] + edges[i + 1]) / 2) for i in range(len(counts))]
    return centers, counts


def _write_png_line_chart(
    path: Path,
    *,
    x: list[int],
    ys: list[tuple[list[float], str]],
    title: str,
    y_label: str,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(12, 6))
    for y_values, label in ys:
        ax.plot(x, y_values, label=label)
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    if len(ys) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _write_png_bar_chart(
    path: Path,
    *,
    x: list[int],
    y: list[int],
    title: str,
    x_label: str,
    y_label: str,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, y, width=max(1, int((x[1] - x[0]) * 0.8) if len(x) > 1 else 1))
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _write_html_line_chart(
    path: Path,
    *,
    x: list[int],
    ys: list[tuple[list[float], str]],
    title: str,
    y_label: str,
) -> bool:
    try:
        import plotly.graph_objects as go
    except Exception:
        return False
    fig = go.Figure()
    for y_values, label in ys:
        fig.add_trace(go.Scatter(x=x, y=y_values, mode="lines", name=label))
    fig.update_layout(title=title, xaxis_title="Index", yaxis_title=y_label, template="plotly_white")
    fig.write_html(str(path), include_plotlyjs="cdn")
    return True


def _write_html_bar_chart(
    path: Path,
    *,
    x: list[int],
    y: list[int],
    title: str,
    x_label: str,
    y_label: str,
) -> bool:
    try:
        import plotly.graph_objects as go
    except Exception:
        return False
    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, template="plotly_white")
    fig.write_html(str(path), include_plotlyjs="cdn")
    return True


def _render_backtest_visuals(
    *,
    plots_dir: Path,
    equity_curve: list[dict[str, Any]],
    drawdown_curve: list[dict[str, Any]],
    rolling_perf: list[dict[str, Any]],
    trade_distribution: dict[str, Any],
) -> list[str]:
    artifacts: list[str] = []
    x_equity = list(range(len(equity_curve)))
    equity_values = [float(row.get("equity", 0.0)) for row in equity_curve]
    benchmark_values = [float(row.get("benchmark_equity", 0.0)) for row in equity_curve]

    x_dd = list(range(len(drawdown_curve)))
    drawdown_values = [float(row.get("drawdown", 0.0)) for row in drawdown_curve]

    x_roll = list(range(len(rolling_perf)))
    rolling_values = [float(row.get("ret_1", 0.0)) for row in rolling_perf]

    sizes = [int(v) for v in trade_distribution.get("sizes", [])]
    hist_x, hist_y = _build_histogram(sizes)

    if x_equity:
        if _write_png_line_chart(
            plots_dir / "equity_vs_benchmark.png",
            x=x_equity,
            ys=[(equity_values, "Equity"), (benchmark_values, "Benchmark")],
            title="Equity vs Benchmark",
            y_label="RUB",
        ):
            artifacts.append("artifacts/backtests/plots/equity_vs_benchmark.png")
        if _write_html_line_chart(
            plots_dir / "equity_vs_benchmark.html",
            x=x_equity,
            ys=[(equity_values, "Equity"), (benchmark_values, "Benchmark")],
            title="Equity vs Benchmark",
            y_label="RUB",
        ):
            artifacts.append("artifacts/backtests/plots/equity_vs_benchmark.html")

    if x_dd:
        if _write_png_line_chart(
            plots_dir / "drawdown_curve.png",
            x=x_dd,
            ys=[(drawdown_values, "Drawdown")],
            title="Drawdown Curve",
            y_label="Drawdown",
        ):
            artifacts.append("artifacts/backtests/plots/drawdown_curve.png")
        if _write_html_line_chart(
            plots_dir / "drawdown_curve.html",
            x=x_dd,
            ys=[(drawdown_values, "Drawdown")],
            title="Drawdown Curve",
            y_label="Drawdown",
        ):
            artifacts.append("artifacts/backtests/plots/drawdown_curve.html")

    if x_roll:
        if _write_png_line_chart(
            plots_dir / "rolling_performance.png",
            x=x_roll,
            ys=[(rolling_values, "ret_1")],
            title="Rolling Performance",
            y_label="Return",
        ):
            artifacts.append("artifacts/backtests/plots/rolling_performance.png")
        if _write_html_line_chart(
            plots_dir / "rolling_performance.html",
            x=x_roll,
            ys=[(rolling_values, "ret_1")],
            title="Rolling Performance",
            y_label="Return",
        ):
            artifacts.append("artifacts/backtests/plots/rolling_performance.html")

    if hist_x and hist_y:
        if _write_png_bar_chart(
            plots_dir / "trade_distribution.png",
            x=hist_x,
            y=hist_y,
            title="Trade Size Distribution",
            x_label="Trade size bucket",
            y_label="Count",
        ):
            artifacts.append("artifacts/backtests/plots/trade_distribution.png")
        if _write_html_bar_chart(
            plots_dir / "trade_distribution.html",
            x=hist_x,
            y=hist_y,
            title="Trade Size Distribution",
            x_label="Trade size bucket",
            y_label="Count",
        ):
            artifacts.append("artifacts/backtests/plots/trade_distribution.html")
    return artifacts


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
    visual_artifacts = _render_backtest_visuals(
        plots_dir=plots_dir,
        equity_curve=equity_curve,
        drawdown_curve=drawdown_curve,
        rolling_perf=rolling_perf,
        trade_distribution=trade_distribution,
    )
    return [
        "artifacts/backtests/backtest_summary.json",
        "artifacts/backtests/equity_curve.json",
        "artifacts/backtests/trade_log.json",
        "artifacts/backtests/plots/equity_vs_benchmark.json",
        "artifacts/backtests/plots/drawdown_curve.json",
        "artifacts/backtests/plots/rolling_performance.json",
        "artifacts/backtests/plots/trade_distribution.json",
    ] + visual_artifacts
