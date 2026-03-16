from __future__ import annotations

from datetime import datetime, timezone
import json

from src.data.feature_store.io import read_parquet
from src.data.feature_store.paths import merged_train_parquet_path
from src.domain.schemas import StageResult
from src.models.policy.offline_policy import OfflinePolicyLayer
from src.models.registry.repository import write_model_metadata
from src.training.pipeline.base import PipelineStage, StageContext
from src.training.tracking.mlflow_utils import build_mlflow_tracker


def _write_json(path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


class TrainPolicyLayerStage(PipelineStage):
    def run(self, ctx: StageContext) -> StageResult:
        started = datetime.now(tz=timezone.utc)
        rows = read_parquet(merged_train_parquet_path(ctx.workspace))
        policy_cfg = ctx.params.get("models_v2", {}).get("policy_layer", {})
        policy = OfflinePolicyLayer(
            risk_aversion=float(policy_cfg.get("risk_aversion", 3.0)),
            turnover_penalty=float(policy_cfg.get("turnover_penalty", 0.25)),
            drawdown_penalty=float(policy_cfg.get("drawdown_penalty", 0.20)),
            uncertainty_penalty=float(policy_cfg.get("uncertainty_penalty", 0.35)),
            max_position=float(policy_cfg.get("max_position", 1.0)),
            min_confidence=float(policy_cfg.get("min_confidence", 0.50)),
            signal_deadband=float(policy_cfg.get("signal_deadband", 0.0005)),
            max_turnover_step=float(policy_cfg.get("max_turnover_step", 0.20)),
            signal_to_position_scale=float(policy_cfg.get("signal_to_position_scale", 500.0)),
        )
        fit_metrics = policy.fit(rows, target_key="return_1")
        decisions = policy.decide_batch(rows[: max(0, min(5000, len(rows)))])
        avg_utility = (
            sum(float(d.get("expected_utility", 0.0)) for d in decisions) / len(decisions)
            if decisions
            else 0.0
        )
        avg_turnover = (
            sum(float(d.get("turnover_proxy", 0.0)) for d in decisions) / len(decisions)
            if decisions
            else 0.0
        )
        avg_abs_position = (
            sum(abs(float(d.get("target_position", 0.0))) for d in decisions) / len(decisions)
            if decisions
            else 0.0
        )
        summary = {
            "decision_samples": float(len(decisions)),
            "policy_avg_utility": float(avg_utility),
            "policy_avg_turnover_proxy": float(avg_turnover),
            "policy_avg_abs_position": float(avg_abs_position),
        }
        model_path = ctx.workspace / "models" / "policy" / "offline_policy_layer.pkl"
        policy.save(model_path)
        write_model_metadata(
            ctx.workspace / "models" / "policy" / "metadata.json",
            {
                "run_id": ctx.run_id,
                "trained_policy": policy.policy_name,
                "policy_metadata": policy.get_metadata(),
            },
        )
        _write_json(
            ctx.workspace / "artifacts" / "evaluation" / "policy_layer_summary.json",
            {
                "stage": ctx.stage_name,
                "fit_metrics": fit_metrics,
                "summary_metrics": summary,
            },
        )

        tracker = build_mlflow_tracker(ctx.params, ctx.stage_params)
        with tracker.run(ctx.stage_name, tags={"stage": ctx.stage_name, "run_id": ctx.run_id}, nested=True):
            tracker.log_dataset_metadata(dataset_name="merged_train", row_count=len(rows), metadata={"stage": ctx.stage_name})
            tracker.log_model_artifact_metadata(
                model_name=policy.policy_name,
                model_version=policy.policy_version,
                artifact_path=str(model_path.relative_to(ctx.workspace)),
                metadata=policy.get_metadata(),
            )
            tracker.log_metrics({**{f"policy_fit_{k}": float(v) for k, v in fit_metrics.items()}, **summary})
            tracker.log_artifact(str(model_path), artifact_path="models/policy")
            tracker.log_artifact(
                str(ctx.workspace / "artifacts" / "evaluation" / "policy_layer_summary.json"),
                artifact_path="artifacts/evaluation",
            )

        finished = datetime.now(tz=timezone.utc)
        return StageResult(
            run_id=ctx.run_id,
            stage_name=ctx.stage_name,
            success=True,
            started_at=started,
            finished_at=finished,
            metrics={**{k: float(v) for k, v in fit_metrics.items()}, **summary},
            artifacts=[
                "models/policy/offline_policy_layer.pkl",
                "models/policy/metadata.json",
                "artifacts/evaluation/policy_layer_summary.json",
            ],
        )
