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
from src.models.registry.repository import read_model_metadata, write_model_metadata
from src.training.pipeline.base import PipelineStage, StageContext
from src.training.tracking.mlflow_utils import build_mlflow_tracker, maybe_log_file_artifact


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_md(path: Path, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = [f"# {title}", ""] + [f"- {line}" for line in lines]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


class EvaluateModelsStage(PipelineStage):
    def run(self, ctx: StageContext) -> StageResult:
        started = datetime.now(tz=timezone.utc)
        rows = read_parquet(merged_validation_parquet_path(ctx.workspace))
        total = len(rows)
        directional_hits = sum(
            1
            for r in rows
            if (float(r.get("return_1", 0.0)) >= 0) == (float(r.get("momentum_10", 0.0)) >= 0)
        )
        accuracy = directional_hits / total if total else 0.0
        mae = (sum(abs(float(r.get("return_1", 0.0))) for r in rows) / total) if total else 0.0
        metrics = {"validation_samples": float(total), "directional_accuracy": float(accuracy), "mae_proxy": float(mae)}

        report_json = ctx.workspace / "reports" / "evaluate_models.json"
        report_md = ctx.workspace / "reports" / "evaluate_models.md"
        _write_json(report_json, {"stage": ctx.stage_name, "metrics": metrics})
        _write_md(report_md, "Evaluate Models", [f"directional_accuracy: {accuracy:.4f}", f"mae_proxy: {mae:.6f}"])

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
            artifacts=["reports/evaluate_models.json", "reports/evaluate_models.md"],
        )


class BacktestStrategyStage(PipelineStage):
    def run(self, ctx: StageContext) -> StageResult:
        started = datetime.now(tz=timezone.utc)
        rows = read_parquet(merged_test_parquet_path(ctx.workspace))
        bt_cfg = BacktestConfig(
            initial_cash=float(ctx.params.get("backtest", {}).get("initial_cash_rub", 1_000_000)),
            commission_bps=float(ctx.params.get("backtest", {}).get("commission_bps", 5)),
            slippage_bps=float(ctx.params.get("backtest", {}).get("slippage_bps", 5)),
            lot_size=1,
            execution_delay_bars=int(ctx.params.get("backtest", {}).get("delayed_execution_bars", 1)),
            signal_column=str(ctx.stage_params.get("signal_column", "expected_return")),
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
        eval_report = json.loads(eval_report_path.read_text(encoding="utf-8")) if eval_report_path.exists() else {}
        bt_report = json.loads(backtest_report_path.read_text(encoding="utf-8")) if backtest_report_path.exists() else {}
        candidate_sharpe = float(bt_report.get("metrics", {}).get("sharpe", 0.0))
        candidate_acc = float(eval_report.get("metrics", {}).get("directional_accuracy", 0.0))
        threshold = float(ctx.stage_params.get("promotion_sharpe_threshold", 0.0))
        decision = "promote_candidate" if candidate_sharpe >= threshold else "keep_champion"

        payload = {
            "decision": decision,
            "candidate": {"sharpe": candidate_sharpe, "directional_accuracy": candidate_acc},
            "thresholds": {"promotion_sharpe_threshold": threshold},
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
                metrics={"candidate_sharpe": candidate_sharpe, "candidate_directional_accuracy": candidate_acc},
            )

        finished = datetime.now(tz=timezone.utc)
        return StageResult(
            run_id=ctx.run_id,
            stage_name=ctx.stage_name,
            success=True,
            started_at=started,
            finished_at=finished,
            metrics={"candidate_sharpe": candidate_sharpe, "candidate_directional_accuracy": candidate_acc},
            artifacts=["reports/compare_with_production.json", "artifacts/comparison/decision.json"],
        )


class PromoteModelStage(PipelineStage):
    def run(self, ctx: StageContext) -> StageResult:
        started = datetime.now(tz=timezone.utc)
        decision_path = ctx.workspace / "artifacts" / "comparison" / "decision.json"
        decision_payload = json.loads(decision_path.read_text(encoding="utf-8")) if decision_path.exists() else {}
        promote = decision_payload.get("decision") == "promote_candidate"
        registry_payload = {
            "champion_model": "weighted_ensemble" if promote else "existing_champion",
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "source_run_id": ctx.run_id,
            "decision": decision_payload.get("decision", "unknown"),
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
        manifest = {
            "run_id": ctx.run_id,
            "published_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": [
                "models/base",
                "models/news",
                "models/ensemble",
                "models/registry/champion.json",
                "artifacts/backtests/backtest_summary.json",
                "reports/evaluate_models.json",
                "reports/backtest_strategy.json",
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
