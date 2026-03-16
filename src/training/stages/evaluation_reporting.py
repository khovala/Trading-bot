from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from src.backtesting.engine import BacktestConfig, run_backtest
from src.backtesting.metrics import compute_backtest_metrics
from src.backtesting.reporting import write_backtest_outputs
from src.data.feature_store.io import read_parquet
from src.data.feature_store.paths import merged_test_parquet_path, merged_validation_parquet_path
from src.domain.schemas import StageResult
from src.research.evaluation_protocol import check_promotion_criteria, load_promotion_criteria
from src.models.registry.repository import read_model_metadata, write_model_metadata
from src.models.base.serialization import load_pickle
from src.training.pipeline.base import PipelineStage, StageContext
from src.training.tracking.mlflow_utils import build_mlflow_tracker, maybe_log_file_artifact


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_md(path: Path, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = [f"# {title}", ""] + [f"- {line}" for line in lines]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _model_metrics(rows: list[dict], preds: list) -> dict[str, float]:
    n = min(len(rows), len(preds))
    if n <= 0:
        return {
            "samples": 0.0,
            "directional_accuracy": 0.0,
            "mae": 0.0,
            "avg_confidence": 0.0,
            "pnl_proxy": 0.0,
        }
    hits = 0
    mae_sum = 0.0
    conf_sum = 0.0
    pnl_proxy = 0.0
    for row, pred in zip(rows[:n], preds[:n]):
        true_ret = _safe_float(row.get("return_1"))
        pred_ret = _safe_float(getattr(pred, "expected_return", 0.0))
        if (pred_ret >= 0.0) == (true_ret >= 0.0):
            hits += 1
        mae_sum += abs(pred_ret - true_ret)
        conf_sum += _safe_float(getattr(pred, "confidence", 0.0))
        pnl_proxy += (1.0 if pred_ret > 0.0 else (-1.0 if pred_ret < 0.0 else 0.0)) * true_ret
    return {
        "samples": float(n),
        "directional_accuracy": float(hits / n),
        "mae": float(mae_sum / n),
        "avg_confidence": float(conf_sum / n),
        "pnl_proxy": float(pnl_proxy),
    }


def _load_component_models(workspace: Path) -> dict[str, object]:
    models: dict[str, object] = {}
    model_dirs = [
        workspace / "models" / "base",
        workspace / "models" / "news",
        workspace / "models" / "foundation",
    ]
    for model_dir in model_dirs:
        if not model_dir.exists():
            continue
        for pkl in model_dir.glob("*.pkl"):
            try:
                models[pkl.stem] = load_pickle(pkl)
            except Exception:
                continue
    return models


def _inject_ensemble_signal_rows(workspace: Path, rows: list[dict]) -> list[dict]:
    if not rows:
        return rows
    ensemble_path = workspace / "models" / "ensemble" / "weighted_ensemble.pkl"
    if not ensemble_path.exists():
        return rows
    component_models = _load_component_models(workspace)
    if not component_models:
        return rows
    component_preds: dict[str, list] = {}
    for model_name, model in component_models.items():
        if not hasattr(model, "predict"):
            continue
        try:
            preds = model.predict(rows)
        except Exception:
            continue
        if preds:
            component_preds[model_name] = preds
    if not component_preds:
        return rows
    try:
        ensemble_model = load_pickle(ensemble_path)
    except Exception:
        return rows
    if not hasattr(ensemble_model, "combine"):
        return rows
    try:
        ensemble_preds = ensemble_model.combine(component_preds)
    except Exception:
        return rows
    enriched: list[dict] = []
    for row, pred in zip(rows, ensemble_preds):
        enriched.append(
            dict(
                row,
                expected_return=float(getattr(pred, "expected_return", 0.0)),
                confidence=float(getattr(pred, "confidence", 0.0)),
                direction_probability_up=float(getattr(pred, "direction_probability_up", 0.5)),
                direction_probability_down=float(getattr(pred, "direction_probability_down", 0.5)),
            )
        )
    if len(ensemble_preds) < len(rows):
        enriched.extend(rows[len(ensemble_preds) :])
    return enriched


def _write_png_model_charts(
    *,
    out_dir: Path,
    validation_metrics: dict[str, dict[str, float]],
    test_metrics: dict[str, dict[str, float]],
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(set(validation_metrics.keys()) | set(test_metrics.keys()))
    if not names:
        return []

    def _vals(source: dict[str, dict[str, float]], key: str) -> list[float]:
        return [float(source.get(name, {}).get(key, 0.0)) for name in names]

    artifacts: list[str] = []

    # Directional accuracy chart
    x = list(range(len(names)))
    width = 0.4
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar([i - width / 2 for i in x], _vals(validation_metrics, "directional_accuracy"), width=width, label="validation")
    ax.bar([i + width / 2 for i in x], _vals(test_metrics, "directional_accuracy"), width=width, label="test")
    ax.set_title("Directional Accuracy by Model")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    p = out_dir / "directional_accuracy_by_model.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    artifacts.append("artifacts/evaluation/model_plots_png/directional_accuracy_by_model.png")

    # MAE chart
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar([i - width / 2 for i in x], _vals(validation_metrics, "mae"), width=width, label="validation")
    ax.bar([i + width / 2 for i in x], _vals(test_metrics, "mae"), width=width, label="test")
    ax.set_title("MAE by Model")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    p = out_dir / "mae_by_model.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    artifacts.append("artifacts/evaluation/model_plots_png/mae_by_model.png")

    # PnL proxy chart
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(names, _vals(test_metrics, "pnl_proxy"))
    ax.set_title("PnL Proxy by Model (Test)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    p = out_dir / "pnl_proxy_by_model_test.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    artifacts.append("artifacts/evaluation/model_plots_png/pnl_proxy_by_model_test.png")
    return artifacts


class EvaluateModelsStage(PipelineStage):
    def run(self, ctx: StageContext) -> StageResult:
        started = datetime.now(tz=timezone.utc)
        rows = read_parquet(merged_validation_parquet_path(ctx.workspace))
        test_rows = read_parquet(merged_test_parquet_path(ctx.workspace))
        total = len(rows)
        directional_hits = sum(
            1
            for r in rows
            if (float(r.get("return_1", 0.0)) >= 0) == (float(r.get("momentum_10", 0.0)) >= 0)
        )
        accuracy = directional_hits / total if total else 0.0
        mae = (sum(abs(float(r.get("return_1", 0.0))) for r in rows) / total) if total else 0.0
        metrics = {"validation_samples": float(total), "directional_accuracy": float(accuracy), "mae_proxy": float(mae)}

        # Per-model evaluation (validation + test)
        component_models = _load_component_models(ctx.workspace)
        validation_by_model: dict[str, dict[str, float]] = {}
        test_by_model: dict[str, dict[str, float]] = {}
        component_preds_val: dict[str, list] = {}
        component_preds_test: dict[str, list] = {}
        for model_name, model in component_models.items():
            if not hasattr(model, "predict"):
                continue
            try:
                preds_val = model.predict(rows)
                preds_test = model.predict(test_rows)
            except Exception:
                continue
            component_preds_val[model_name] = preds_val
            component_preds_test[model_name] = preds_test
            validation_by_model[model_name] = _model_metrics(rows, preds_val)
            test_by_model[model_name] = _model_metrics(test_rows, preds_test)

        ensemble_metrics_validation: dict[str, float] = {}
        ensemble_metrics_test: dict[str, float] = {}
        ensemble_path = ctx.workspace / "models" / "ensemble" / "weighted_ensemble.pkl"
        if ensemble_path.exists() and component_preds_val and component_preds_test:
            try:
                ensemble_model = load_pickle(ensemble_path)
                if hasattr(ensemble_model, "combine"):
                    ensemble_val_preds = ensemble_model.combine(component_preds_val)
                    ensemble_test_preds = ensemble_model.combine(component_preds_test)
                    ensemble_metrics_validation = _model_metrics(rows, ensemble_val_preds)
                    ensemble_metrics_test = _model_metrics(test_rows, ensemble_test_preds)
                    validation_by_model["weighted_ensemble"] = dict(ensemble_metrics_validation)
                    test_by_model["weighted_ensemble"] = dict(ensemble_metrics_test)
                    # Keep top-level directional_accuracy anchored to ensemble when available.
                    metrics["directional_accuracy"] = float(ensemble_metrics_validation.get("directional_accuracy", accuracy))
                    metrics["mae_proxy"] = float(ensemble_metrics_validation.get("mae", mae))
            except Exception:
                pass

        # Render separate PNG diagnostics directory for model behavior.
        model_plot_dir = ctx.workspace / "artifacts" / "evaluation" / "model_plots_png"
        png_artifacts = _write_png_model_charts(
            out_dir=model_plot_dir,
            validation_metrics=validation_by_model,
            test_metrics=test_by_model,
        )

        detailed_report = {
            "stage": ctx.stage_name,
            "validation_metrics_by_model": validation_by_model,
            "test_metrics_by_model": test_by_model,
            "ensemble_validation": ensemble_metrics_validation,
            "ensemble_test": ensemble_metrics_test,
        }
        detailed_report_path = ctx.workspace / "reports" / "evaluate_models_detailed.json"
        _write_json(detailed_report_path, detailed_report)

        ablation_payload = _load_json(ctx.workspace / "artifacts" / "evaluation" / "ensemble_ablation.json")
        policy_payload = _load_json(ctx.workspace / "artifacts" / "evaluation" / "policy_layer_summary.json")
        ablation = ablation_payload.get("ablation", {})
        weights = ablation_payload.get("ensemble_weights", {})
        diagnostics = ablation_payload.get("ensemble_diagnostics", {})
        if isinstance(ablation, dict):
            deltas = [
                _safe_float(v.get("mean_abs_expected_return_delta"))
                for v in ablation.values()
                if isinstance(v, dict)
            ]
            positive_count = sum(1 for d in deltas if d > 0)
            metrics["ablation_model_count"] = float(len(deltas))
            metrics["ablation_positive_ratio"] = float(positive_count / len(deltas)) if deltas else 0.0
            metrics["ablation_mean_abs_delta"] = float(sum(abs(d) for d in deltas) / len(deltas)) if deltas else 0.0
            if "news_feature_model" in ablation and isinstance(ablation["news_feature_model"], dict):
                metrics["ablation_news_model_delta"] = _safe_float(
                    ablation["news_feature_model"].get("mean_abs_expected_return_delta")
                )
        if isinstance(weights, dict) and weights:
            weight_values = [_safe_float(v) for v in weights.values()]
            metrics["ensemble_weight_max"] = max(weight_values) if weight_values else 0.0
            metrics["ensemble_weight_concentration_hhi"] = float(sum(v * v for v in weight_values))
        if isinstance(diagnostics, dict):
            model_stats = diagnostics.get("model_stats", {})
            if isinstance(model_stats, dict) and model_stats:
                confs = [_safe_float(v.get("avg_confidence")) for v in model_stats.values() if isinstance(v, dict)]
                turnovers = [_safe_float(v.get("turnover_proxy")) for v in model_stats.values() if isinstance(v, dict)]
                metrics["ensemble_component_avg_confidence"] = float(sum(confs) / len(confs)) if confs else 0.0
                metrics["ensemble_component_avg_turnover_proxy"] = float(sum(turnovers) / len(turnovers)) if turnovers else 0.0
        if isinstance(policy_payload, dict):
            summary = policy_payload.get("summary_metrics", {})
            if isinstance(summary, dict):
                for key in ("policy_avg_utility", "policy_avg_turnover_proxy", "policy_avg_abs_position", "decision_samples"):
                    if key in summary:
                        metrics[key] = _safe_float(summary.get(key))

        policy_model_path = ctx.workspace / "models" / "policy" / "offline_policy_layer.pkl"
        if policy_model_path.exists() and rows:
            policy_model = load_pickle(policy_model_path)
            if hasattr(policy_model, "decide_batch"):
                decisions = policy_model.decide_batch(rows)  # type: ignore[attr-defined]
                if isinstance(decisions, list) and decisions:
                    turnover_budget = float(ctx.stage_params.get("turnover_budget_per_step", 0.25))
                    utility_scale = float(ctx.stage_params.get("utility_scale", 100.0))
                    active_count = 0
                    hit_count = 0
                    utility_values: list[float] = []
                    turnover_values: list[float] = []
                    turnover_violations = 0
                    for row, decision in zip(rows, decisions):
                        target_position = _safe_float(decision.get("target_position"))
                        ret_1 = _safe_float(row.get("return_1"))
                        if target_position != 0.0:
                            active_count += 1
                            if target_position * ret_1 > 0.0:
                                hit_count += 1
                        utility = _safe_float(decision.get("expected_utility"))
                        turnover_proxy = _safe_float(decision.get("turnover_proxy"))
                        utility_values.append(utility)
                        turnover_values.append(turnover_proxy)
                        if turnover_proxy > turnover_budget:
                            turnover_violations += 1
                    avg_utility = (sum(utility_values) / len(utility_values)) if utility_values else 0.0
                    policy_hit_ratio = float(hit_count / active_count) if active_count > 0 else 0.0
                    violation_ratio = float(turnover_violations / len(turnover_values)) if turnover_values else 0.0
                    utility_adjusted_score = (
                        policy_hit_ratio * max(0.0, 1.0 + avg_utility * utility_scale) * max(0.0, 1.0 - violation_ratio)
                    )
                    metrics["policy_active_decisions"] = float(active_count)
                    metrics["policy_hit_ratio"] = float(policy_hit_ratio)
                    metrics["policy_avg_decision_utility"] = float(avg_utility)
                    metrics["policy_turnover_budget_per_step"] = float(turnover_budget)
                    metrics["policy_turnover_budget_violations"] = float(turnover_violations)
                    metrics["policy_turnover_budget_violation_ratio"] = float(violation_ratio)
                    metrics["policy_utility_adjusted_score"] = float(utility_adjusted_score)

        report_json = ctx.workspace / "reports" / "evaluate_models.json"
        report_md = ctx.workspace / "reports" / "evaluate_models.md"
        _write_json(report_json, {"stage": ctx.stage_name, "metrics": metrics})
        _write_md(
            report_md,
            "Evaluate Models",
            [f"{k}: {v:.6f}" for k, v in sorted(metrics.items())],
        )

        tracker = build_mlflow_tracker(ctx.params, ctx.stage_params)
        with tracker.run(ctx.stage_name, tags={"stage": ctx.stage_name, "run_id": ctx.run_id}, nested=True):
            tracker.log_dataset_metadata(dataset_name="validation", row_count=total, metadata={"stage": ctx.stage_name})
            tracker.log_feature_schema_version(
                feature_family="merged",
                schema_version=str(ctx.params.get("features", {}).get("merged", {}).get("schema_version", "v1")),
            )
            tracker.log_metrics(metrics)
            maybe_log_file_artifact(tracker, report_json, artifact_path="reports")
            maybe_log_file_artifact(tracker, report_md, artifact_path="reports")

        finished = datetime.now(tz=timezone.utc)
        return StageResult(
            run_id=ctx.run_id,
            stage_name=ctx.stage_name,
            success=True,
            started_at=started,
            finished_at=finished,
            metrics=metrics,
            artifacts=[
                "reports/evaluate_models.json",
                "reports/evaluate_models.md",
                "reports/evaluate_models_detailed.json",
            ]
            + png_artifacts,
        )


class BacktestStrategyStage(PipelineStage):
    def run(self, ctx: StageContext) -> StageResult:
        started = datetime.now(tz=timezone.utc)
        rows = read_parquet(merged_test_parquet_path(ctx.workspace))
        rows = _inject_ensemble_signal_rows(ctx.workspace, rows)
        use_policy_layer = bool(ctx.stage_params.get("use_policy_layer", True))
        policy_model_path = ctx.workspace / "models" / "policy" / "offline_policy_layer.pkl"
        policy_backtest_enabled = False
        if use_policy_layer and policy_model_path.exists() and rows:
            policy_model = load_pickle(policy_model_path)
            if hasattr(policy_model, "decide_batch"):
                policy_decisions = policy_model.decide_batch(rows)  # type: ignore[attr-defined]
                if len(policy_decisions) == len(rows):
                    enriched_rows: list[dict] = []
                    for row, decision in zip(rows, policy_decisions):
                        target_position = _safe_float(decision.get("target_position"))
                        enriched = dict(
                            row,
                            policy_target_position=target_position,
                            policy_signal=1.0 if target_position > 0 else (-1.0 if target_position < 0 else 0.0),
                            policy_expected_utility=_safe_float(decision.get("expected_utility")),
                            policy_turnover_proxy=_safe_float(decision.get("turnover_proxy")),
                        )
                        enriched_rows.append(enriched)
                    rows = enriched_rows
                    policy_backtest_enabled = True

        bt_cfg = BacktestConfig(
            initial_cash=float(ctx.params.get("backtest", {}).get("initial_cash_rub", 1_000_000)),
            commission_bps=float(ctx.params.get("backtest", {}).get("commission_bps", 5)),
            slippage_bps=float(ctx.params.get("backtest", {}).get("slippage_bps", 5)),
            lot_size=1,
            execution_delay_bars=int(ctx.params.get("backtest", {}).get("delayed_execution_bars", 1)),
            signal_threshold=float(ctx.params.get("backtest", {}).get("signal_threshold", 0.0005)),
            position_size_pct=float(ctx.params.get("backtest", {}).get("position_size_pct", 0.05)),
            signal_column=str(ctx.stage_params.get("signal_column", "policy_signal" if policy_backtest_enabled else "expected_return")),
            target_position_column=("policy_target_position" if policy_backtest_enabled else None),
        )

        result = run_backtest(rows, bt_cfg)
        summary = compute_backtest_metrics(
            equity_curve=result["equity_curve"],
            trade_log=result["trade_log"],
            turnover=float(result["summary"]["turnover"]),
            exposure_mean=float(result["summary"]["exposure_mean"]),
        )

        # Optional ablation: with-news vs without-news proxy signal.
        no_news_rows = [dict(r, news_sentiment_mean=0.0) for r in rows]
        no_news_result = run_backtest(no_news_rows, bt_cfg)
        no_news_summary = compute_backtest_metrics(
            equity_curve=no_news_result["equity_curve"],
            trade_log=no_news_result["trade_log"],
            turnover=float(no_news_result["summary"]["turnover"]),
            exposure_mean=float(no_news_result["summary"]["exposure_mean"]),
        )
        summary["ablation_pnl_delta_with_minus_without_news"] = summary["pnl"] - no_news_summary["pnl"]
        summary["policy_backtest_enabled"] = 1.0 if policy_backtest_enabled else 0.0
        if policy_backtest_enabled:
            policy_positions = [abs(_safe_float(r.get("policy_target_position"))) for r in rows]
            summary["policy_avg_abs_target_position"] = (
                float(sum(policy_positions) / len(policy_positions)) if policy_positions else 0.0
            )
            policy_utilities = [_safe_float(r.get("policy_expected_utility")) for r in rows]
            summary["policy_avg_expected_utility"] = (
                float(sum(policy_utilities) / len(policy_utilities)) if policy_utilities else 0.0
            )

        # Simple walk-forward: split chronologically into equal folds.
        fold_count = max(1, int(ctx.stage_params.get("walk_forward_folds", 3)))
        fold_size = max(1, len(rows) // fold_count) if rows else 1
        wf_sharpes: list[float] = []
        for i in range(fold_count):
            start = i * fold_size
            end = len(rows) if i == fold_count - 1 else min(len(rows), (i + 1) * fold_size)
            fold_rows = rows[start:end]
            if len(fold_rows) < 2:
                continue
            fold_result = run_backtest(fold_rows, bt_cfg)
            fold_metrics = compute_backtest_metrics(
                equity_curve=fold_result["equity_curve"],
                trade_log=fold_result["trade_log"],
                turnover=float(fold_result["summary"]["turnover"]),
                exposure_mean=float(fold_result["summary"]["exposure_mean"]),
            )
            wf_sharpes.append(float(fold_metrics["sharpe"]))
        summary["walk_forward_sharpe_mean"] = float(sum(wf_sharpes) / len(wf_sharpes)) if wf_sharpes else 0.0
        summary["walk_forward_fold_count"] = float(len(wf_sharpes))

        out_dir = ctx.workspace / "artifacts" / "backtests"
        artifacts = write_backtest_outputs(
            out_dir=out_dir,
            summary=summary,
            equity_curve=result["equity_curve"],
            trade_log=result["trade_log"],
        )
        _write_json(ctx.workspace / "reports" / "backtest_strategy.json", {"stage": ctx.stage_name, "metrics": summary})
        _write_md(
            ctx.workspace / "reports" / "backtest_strategy.md",
            "Backtest Strategy",
            [f"{k}: {v:.6f}" for k, v in summary.items()],
        )

        tracker = build_mlflow_tracker(ctx.params, ctx.stage_params)
        with tracker.run(ctx.stage_name, tags={"stage": ctx.stage_name, "run_id": ctx.run_id}, nested=True):
            tracker.log_dataset_metadata(dataset_name="test", row_count=len(rows), metadata={"stage": ctx.stage_name})
            tracker.log_metrics({k: float(v) for k, v in summary.items()})
            maybe_log_file_artifact(tracker, ctx.workspace / "reports" / "backtest_strategy.json", artifact_path="reports")
            maybe_log_file_artifact(tracker, ctx.workspace / "reports" / "backtest_strategy.md", artifact_path="reports")
            maybe_log_file_artifact(tracker, out_dir / "backtest_summary.json", artifact_path="backtests")

        finished = datetime.now(tz=timezone.utc)
        return StageResult(
            run_id=ctx.run_id,
            stage_name=ctx.stage_name,
            success=True,
            started_at=started,
            finished_at=finished,
            metrics={k: float(v) for k, v in summary.items()},
            artifacts=artifacts + ["reports/backtest_strategy.json", "reports/backtest_strategy.md"],
        )


class CompareWithProductionStage(PipelineStage):
    def run(self, ctx: StageContext) -> StageResult:
        started = datetime.now(tz=timezone.utc)
        eval_report_path = ctx.workspace / "reports" / "evaluate_models.json"
        backtest_report_path = ctx.workspace / "reports" / "backtest_strategy.json"
        eval_report = _load_json(eval_report_path)
        bt_report = _load_json(backtest_report_path)
        baseline_snapshot = _load_json(ctx.workspace / "reports" / "baseline_snapshot.json")
        candidate_sharpe = float(bt_report.get("metrics", {}).get("sharpe", 0.0))
        candidate_acc = float(eval_report.get("metrics", {}).get("directional_accuracy", 0.0))
        threshold = float(ctx.stage_params.get("promotion_sharpe_threshold", 0.0))
        criteria = load_promotion_criteria(ctx.params)

        eval_metrics = eval_report.get("metrics", {}) if isinstance(eval_report.get("metrics"), dict) else {}
        bt_metrics = bt_report.get("metrics", {}) if isinstance(bt_report.get("metrics"), dict) else {}
        merged_metrics = {**{str(k): _safe_float(v) for k, v in eval_metrics.items()}, **{str(k): _safe_float(v) for k, v in bt_metrics.items()}}
        baseline_metrics = baseline_snapshot.get("metrics", {}) if isinstance(baseline_snapshot.get("metrics"), dict) else {}

        acceptance = check_promotion_criteria(
            metrics=merged_metrics,
            criteria=criteria,
            baseline_metrics={str(k): _safe_float(v) for k, v in baseline_metrics.items()},
            positive_fold_count=int(bt_metrics.get("walk_forward_positive_folds", 0)),
            total_fold_count=int(bt_metrics.get("walk_forward_fold_count", 0)),
            cost_stress_passed=bool(bt_metrics.get("cost_stress_passed", True)),
            has_calibration_report=bool(eval_metrics.get("has_calibration_report", True)),
            has_leakage_report=bool(eval_metrics.get("has_leakage_report", True)),
            news_ablation_benefit=_safe_float(eval_metrics.get("ablation_news_model_delta", 0.0)),
        )

        min_ablation_positive_ratio = float(ctx.stage_params.get("min_ablation_positive_ratio", 0.0))
        max_weight_concentration_hhi = float(ctx.stage_params.get("max_weight_concentration_hhi", 1.0))
        min_policy_avg_utility = float(ctx.stage_params.get("min_policy_avg_utility", -1e9))
        max_policy_turnover_proxy = float(ctx.stage_params.get("max_policy_turnover_proxy", 1e9))
        ablation_positive_ratio = _safe_float(eval_metrics.get("ablation_positive_ratio", 0.0))
        weight_hhi = _safe_float(eval_metrics.get("ensemble_weight_concentration_hhi", 1.0))
        policy_avg_utility = _safe_float(eval_metrics.get("policy_avg_utility", 0.0))
        policy_avg_turnover_proxy = _safe_float(eval_metrics.get("policy_avg_turnover_proxy", 0.0))
        extra_checks = {
            "legacy_promotion_sharpe_threshold": candidate_sharpe >= threshold,
            "ablation_positive_ratio": ablation_positive_ratio >= min_ablation_positive_ratio,
            "weight_concentration_hhi": weight_hhi <= max_weight_concentration_hhi,
            "policy_avg_utility": policy_avg_utility >= min_policy_avg_utility,
            "policy_avg_turnover_proxy": policy_avg_turnover_proxy <= max_policy_turnover_proxy,
        }
        decision = "promote_candidate" if (acceptance.passed and all(extra_checks.values())) else "keep_champion"

        payload = {
            "decision": decision,
            "candidate": {"sharpe": candidate_sharpe, "directional_accuracy": candidate_acc},
            "thresholds": {
                "promotion_sharpe_threshold": threshold,
                "min_ablation_positive_ratio": min_ablation_positive_ratio,
                "max_weight_concentration_hhi": max_weight_concentration_hhi,
                "min_policy_avg_utility": min_policy_avg_utility,
                "max_policy_turnover_proxy": max_policy_turnover_proxy,
            },
            "checks": {
                "promotion_criteria": acceptance.checks,
                "stage3_signals": extra_checks,
            },
            "details": {
                **acceptance.details,
                "ablation_positive_ratio": ablation_positive_ratio,
                "ensemble_weight_concentration_hhi": weight_hhi,
                "policy_avg_utility": policy_avg_utility,
                "policy_avg_turnover_proxy": policy_avg_turnover_proxy,
            },
        }
        _write_json(ctx.workspace / "reports" / "compare_with_production.json", payload)
        _write_json(ctx.workspace / "artifacts" / "comparison" / "decision.json", payload)

        tracker = build_mlflow_tracker(ctx.params, ctx.stage_params)
        with tracker.run(ctx.stage_name, tags={"stage": ctx.stage_name, "run_id": ctx.run_id}, nested=True):
            tracker.log_comparison_decision(
                candidate_run_id=ctx.run_id,
                champion_run_id=None,
                decision=decision,
                reason=f"candidate_sharpe={candidate_sharpe:.4f}",
                metrics={
                    "candidate_sharpe": candidate_sharpe,
                    "candidate_directional_accuracy": candidate_acc,
                    "ablation_positive_ratio": ablation_positive_ratio,
                    "ensemble_weight_concentration_hhi": weight_hhi,
                    "policy_avg_utility": policy_avg_utility,
                    "policy_avg_turnover_proxy": policy_avg_turnover_proxy,
                    "promotion_checks_passed": 1.0 if acceptance.passed else 0.0,
                },
            )

        finished = datetime.now(tz=timezone.utc)
        return StageResult(
            run_id=ctx.run_id,
            stage_name=ctx.stage_name,
            success=True,
            started_at=started,
            finished_at=finished,
            metrics={
                "candidate_sharpe": candidate_sharpe,
                "candidate_directional_accuracy": candidate_acc,
                "ablation_positive_ratio": ablation_positive_ratio,
                "ensemble_weight_concentration_hhi": weight_hhi,
                "policy_avg_utility": policy_avg_utility,
                "policy_avg_turnover_proxy": policy_avg_turnover_proxy,
                "promotion_checks_passed": 1.0 if acceptance.passed else 0.0,
            },
            artifacts=["reports/compare_with_production.json", "artifacts/comparison/decision.json"],
        )


class PromoteModelStage(PipelineStage):
    def run(self, ctx: StageContext) -> StageResult:
        started = datetime.now(tz=timezone.utc)
        decision_path = ctx.workspace / "artifacts" / "comparison" / "decision.json"
        decision_payload = _load_json(decision_path)
        promote = decision_payload.get("decision") == "promote_candidate"
        checks = decision_payload.get("checks", {}) if isinstance(decision_payload.get("checks"), dict) else {}
        details = decision_payload.get("details", {}) if isinstance(decision_payload.get("details"), dict) else {}
        registry_payload = {
            "champion_model": "weighted_ensemble" if promote else "existing_champion",
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "source_run_id": ctx.run_id,
            "decision": decision_payload.get("decision", "unknown"),
            "promotion_checks": checks,
            "promotion_details": details,
            "decision_snapshot_path": "artifacts/comparison/decision.json",
        }
        write_model_metadata(ctx.workspace / "models" / "registry" / "champion.json", registry_payload)
        _write_json(ctx.workspace / "reports" / "promote_model.json", registry_payload)

        tracker = build_mlflow_tracker(ctx.params, ctx.stage_params)
        with tracker.run(ctx.stage_name, tags={"stage": ctx.stage_name, "run_id": ctx.run_id}, nested=True):
            tracker.log_model_artifact_metadata(
                model_name=registry_payload["champion_model"],
                model_version="promoted",
                artifact_path="models/registry/champion.json",
                metadata=registry_payload,
            )

        finished = datetime.now(tz=timezone.utc)
        return StageResult(
            run_id=ctx.run_id,
            stage_name=ctx.stage_name,
            success=True,
            started_at=started,
            finished_at=finished,
            metrics={"promoted": 1.0 if promote else 0.0},
            artifacts=["models/registry/champion.json", "reports/promote_model.json"],
        )


class PublishArtifactsStage(PipelineStage):
    def run(self, ctx: StageContext) -> StageResult:
        started = datetime.now(tz=timezone.utc)
        registry_payload = _load_json(ctx.workspace / "models" / "registry" / "champion.json")
        decision_payload = _load_json(ctx.workspace / "artifacts" / "comparison" / "decision.json")
        manifest = {
            "run_id": ctx.run_id,
            "published_at": datetime.now(timezone.utc).isoformat(),
            "promotion_summary": {
                "decision": decision_payload.get("decision", "unknown"),
                "checks": decision_payload.get("checks", {}),
                "details": decision_payload.get("details", {}),
                "champion_model": registry_payload.get("champion_model", "unknown"),
            },
            "artifacts": [
                "models/base",
                "models/news",
                "models/ensemble",
                "models/registry/champion.json",
                "artifacts/backtests/backtest_summary.json",
                "artifacts/evaluation/ensemble_ablation.json",
                "artifacts/comparison/decision.json",
                "reports/evaluate_models.json",
                "reports/backtest_strategy.json",
                "reports/compare_with_production.json",
                "reports/promote_model.json",
            ],
        }
        _write_json(ctx.workspace / "artifacts" / "published" / "bundle_manifest.json", manifest)
        _write_json(ctx.workspace / "reports" / "publish_artifacts.json", manifest)
        tracker = build_mlflow_tracker(ctx.params, ctx.stage_params)
        with tracker.run(ctx.stage_name, tags={"stage": ctx.stage_name, "run_id": ctx.run_id}, nested=True):
            tracker.log_params({"stage_name": ctx.stage_name, "artifact_count": len(manifest["artifacts"])})
            tracker.log_metrics({"artifact_count": float(len(manifest["artifacts"]))})
            maybe_log_file_artifact(
                tracker,
                ctx.workspace / "artifacts" / "published" / "bundle_manifest.json",
                artifact_path="published",
            )
        finished = datetime.now(tz=timezone.utc)
        return StageResult(
            run_id=ctx.run_id,
            stage_name=ctx.stage_name,
            success=True,
            started_at=started,
            finished_at=finished,
            metrics={"artifact_count": float(len(manifest["artifacts"]))},
            artifacts=["artifacts/published/bundle_manifest.json", "reports/publish_artifacts.json"],
        )


class GenerateReportsStage(PipelineStage):
    def run(self, ctx: StageContext) -> StageResult:
        started = datetime.now(tz=timezone.utc)
        report_names = [
            "evaluate_models.json",
            "backtest_strategy.json",
            "compare_with_production.json",
            "promote_model.json",
            "publish_artifacts.json",
        ]
        consolidated = {"run_id": ctx.run_id, "generated_at": datetime.now(timezone.utc).isoformat(), "reports": {}}
        for name in report_names:
            path = ctx.workspace / "reports" / name
            if path.exists():
                consolidated["reports"][name] = json.loads(path.read_text(encoding="utf-8"))
        _write_json(ctx.workspace / "reports" / "final" / "retrain_report.json", consolidated)
        _write_md(
            ctx.workspace / "reports" / "final" / "retrain_report.md",
            "Retrain Report",
            [
                f"run_id: {ctx.run_id}",
                f"report_count: {len(consolidated['reports'])}",
                "included_reports: " + ", ".join(sorted(consolidated["reports"].keys())),
            ],
        )
        tracker = build_mlflow_tracker(ctx.params, ctx.stage_params)
        with tracker.run(ctx.stage_name, tags={"stage": ctx.stage_name, "run_id": ctx.run_id}, nested=True):
            tracker.log_metrics({"report_count": float(len(consolidated["reports"]))})
            maybe_log_file_artifact(tracker, ctx.workspace / "reports" / "final" / "retrain_report.json", artifact_path="reports/final")
            maybe_log_file_artifact(tracker, ctx.workspace / "reports" / "final" / "retrain_report.md", artifact_path="reports/final")
        finished = datetime.now(tz=timezone.utc)
        return StageResult(
            run_id=ctx.run_id,
            stage_name=ctx.stage_name,
            success=True,
            started_at=started,
            finished_at=finished,
            metrics={"report_count": float(len(consolidated["reports"]))},
            artifacts=["reports/final/retrain_report.json", "reports/final/retrain_report.md"],
        )
