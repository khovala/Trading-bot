from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

from src.backtesting.engine import BacktestConfig, run_backtest
from src.backtesting.metrics import compute_backtest_metrics
from src.data.feature_store.io import read_parquet
from src.data.feature_store.paths import merged_test_parquet_path
from src.domain.schemas import StageResult
from src.models.policy.offline_policy import OfflinePolicyLayer
from src.training.pipeline.base import PipelineStage, StageContext
from src.training.stages.evaluation_reporting import _inject_ensemble_signal_rows
from src.training.tracking.mlflow_utils import build_mlflow_tracker


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _grid_values(params: dict, key: str, default: list[float]) -> list[float]:
    value = params.get(key, default)
    if not isinstance(value, list):
        return default
    out: list[float] = []
    for item in value:
        try:
            out.append(float(item))
        except Exception:
            continue
    return out or default


@dataclass(frozen=True, slots=True)
class CandidateConfig:
    min_confidence: float
    signal_deadband: float
    max_turnover_step: float
    signal_to_position_scale: float
    backtest_position_size_pct: float
    backtest_signal_threshold: float


class TunePolicyBacktestStage(PipelineStage):
    def run(self, ctx: StageContext) -> StageResult:
        started = datetime.now(tz=timezone.utc)
        rows = read_parquet(merged_test_parquet_path(ctx.workspace))
        rows = _inject_ensemble_signal_rows(ctx.workspace, rows)

        policy_cfg = ctx.params.get("models_v2", {}).get("policy_layer", {})
        bt_cfg = ctx.params.get("backtest", {})
        tuner_cfg = ctx.stage_params

        min_conf_grid = _grid_values(tuner_cfg, "min_confidence_grid", [0.0, 0.05, 0.1, 0.2])
        deadband_grid = _grid_values(tuner_cfg, "signal_deadband_grid", [0.0, 0.00005, 0.0001, 0.0002])
        max_turnover_grid = _grid_values(tuner_cfg, "max_turnover_step_grid", [0.05, 0.1, 0.2])
        scale_grid = _grid_values(tuner_cfg, "signal_to_position_scale_grid", [200.0, 350.0, 500.0, 750.0])
        pos_pct_grid = _grid_values(tuner_cfg, "backtest_position_size_pct_grid", [0.01, 0.02, 0.03, 0.05])
        signal_th_grid = _grid_values(tuner_cfg, "backtest_signal_threshold_grid", [0.0, 0.00005, 0.0001, 0.0002])
        max_candidates = int(tuner_cfg.get("max_candidates", 200))

        candidates: list[CandidateConfig] = []
        for min_conf in min_conf_grid:
            for deadband in deadband_grid:
                for max_turn in max_turnover_grid:
                    for scale in scale_grid:
                        for pos_pct in pos_pct_grid:
                            for sig_th in signal_th_grid:
                                candidates.append(
                                    CandidateConfig(
                                        min_confidence=min_conf,
                                        signal_deadband=deadband,
                                        max_turnover_step=max_turn,
                                        signal_to_position_scale=scale,
                                        backtest_position_size_pct=pos_pct,
                                        backtest_signal_threshold=sig_th,
                                    )
                                )
        if len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]

        results: list[dict] = []
        best_row: dict | None = None
        for idx, candidate in enumerate(candidates):
            policy = OfflinePolicyLayer(
                risk_aversion=float(policy_cfg.get("risk_aversion", 3.0)),
                turnover_penalty=float(policy_cfg.get("turnover_penalty", 0.25)),
                drawdown_penalty=float(policy_cfg.get("drawdown_penalty", 0.2)),
                uncertainty_penalty=float(policy_cfg.get("uncertainty_penalty", 0.35)),
                max_position=float(policy_cfg.get("max_position", 1.0)),
                min_confidence=float(candidate.min_confidence),
                signal_deadband=float(candidate.signal_deadband),
                max_turnover_step=float(candidate.max_turnover_step),
                signal_to_position_scale=float(candidate.signal_to_position_scale),
            )
            policy.fit(rows)
            decisions = policy.decide_batch(rows)
            eval_rows: list[dict] = []
            for row, decision in zip(rows, decisions):
                tp = float(decision.get("target_position", 0.0))
                eval_rows.append(
                    dict(
                        row,
                        policy_target_position=tp,
                        policy_signal=1.0 if tp > 0.0 else (-1.0 if tp < 0.0 else 0.0),
                    )
                )
            cfg = BacktestConfig(
                initial_cash=float(bt_cfg.get("initial_cash_rub", 1_000_000)),
                commission_bps=float(bt_cfg.get("commission_bps", 5)),
                slippage_bps=float(bt_cfg.get("slippage_bps", 5)),
                lot_size=1,
                execution_delay_bars=int(bt_cfg.get("delayed_execution_bars", 1)),
                signal_threshold=float(candidate.backtest_signal_threshold),
                position_size_pct=float(candidate.backtest_position_size_pct),
                signal_column="policy_signal",
                target_position_column="policy_target_position",
                stop_loss_pct=bt_cfg.get("stop_loss_pct"),
                take_profit_pct=bt_cfg.get("take_profit_pct"),
            )
            bt = run_backtest(eval_rows, cfg)
            metrics = compute_backtest_metrics(
                equity_curve=bt["equity_curve"],
                trade_log=bt["trade_log"],
                turnover=float(bt["summary"]["turnover"]),
                exposure_mean=float(bt["summary"]["exposure_mean"]),
            )
            score = (
                1.0 if float(metrics.get("pnl", 0.0)) > 0.0 else 0.0,
                float(metrics.get("pnl", 0.0)),
                float(metrics.get("sharpe", 0.0)),
                -float(metrics.get("drawdown", 1.0)),
            )
            row = {
                "candidate_index": idx,
                "config": asdict(candidate),
                "metrics": {k: float(v) for k, v in metrics.items()},
                "score": list(score),
            }
            results.append(row)
            if best_row is None or tuple(row["score"]) > tuple(best_row["score"]):
                best_row = row

        best_row = best_row or {
            "candidate_index": -1,
            "config": {},
            "metrics": {"pnl": 0.0, "sharpe": 0.0, "drawdown": 0.0},
            "score": [0.0, 0.0, 0.0, -1.0],
        }

        artifacts_dir = ctx.workspace / "artifacts" / "evaluation"
        grid_path = artifacts_dir / "policy_tuning_grid.json"
        best_path = artifacts_dir / "policy_tuning_best.json"
        _write_json(grid_path, {"stage": ctx.stage_name, "candidates": results})
        _write_json(best_path, {"stage": ctx.stage_name, "best": best_row})

        # Persist recommended params for manual/automated transfer to params.yaml.
        recommendations_path = ctx.workspace / "artifacts" / "evaluation" / "policy_tuning_recommendations.json"
        _write_json(
            recommendations_path,
            {
                "models_v2.policy_layer": {
                    "min_confidence": best_row["config"].get("min_confidence"),
                    "signal_deadband": best_row["config"].get("signal_deadband"),
                    "max_turnover_step": best_row["config"].get("max_turnover_step"),
                    "signal_to_position_scale": best_row["config"].get("signal_to_position_scale"),
                },
                "backtest": {
                    "position_size_pct": best_row["config"].get("backtest_position_size_pct"),
                    "signal_threshold": best_row["config"].get("backtest_signal_threshold"),
                },
                "best_metrics": best_row["metrics"],
            },
        )

        tracker = build_mlflow_tracker(ctx.params, ctx.stage_params)
        with tracker.run(ctx.stage_name, tags={"stage": ctx.stage_name, "run_id": ctx.run_id}, nested=True):
            tracker.log_params({"candidate_count": len(results)})
            tracker.log_metrics(
                {
                    "best_pnl": float(best_row["metrics"].get("pnl", 0.0)),
                    "best_sharpe": float(best_row["metrics"].get("sharpe", 0.0)),
                    "best_drawdown": float(best_row["metrics"].get("drawdown", 0.0)),
                }
            )
            tracker.log_artifact(str(grid_path), artifact_path="artifacts/evaluation")
            tracker.log_artifact(str(best_path), artifact_path="artifacts/evaluation")
            tracker.log_artifact(str(recommendations_path), artifact_path="artifacts/evaluation")

        finished = datetime.now(tz=timezone.utc)
        return StageResult(
            run_id=ctx.run_id,
            stage_name=ctx.stage_name,
            success=True,
            started_at=started,
            finished_at=finished,
            metrics={
                "candidate_count": float(len(results)),
                "best_pnl": float(best_row["metrics"].get("pnl", 0.0)),
                "best_sharpe": float(best_row["metrics"].get("sharpe", 0.0)),
                "best_drawdown": float(best_row["metrics"].get("drawdown", 0.0)),
            },
            artifacts=[
                "artifacts/evaluation/policy_tuning_grid.json",
                "artifacts/evaluation/policy_tuning_best.json",
                "artifacts/evaluation/policy_tuning_recommendations.json",
            ],
        )
