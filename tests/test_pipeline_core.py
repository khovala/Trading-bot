from __future__ import annotations

from datetime import datetime, timezone

from src.domain.schemas import StageResult
from src.training.pipeline.base import PipelineStage, StageContext
from src.training.pipeline.registry import STAGE_NAMES, StageRegistry
from src.training.pipeline.runner import PipelineRunner


def test_stage_registry_registers_all_placeholders() -> None:
    registry = StageRegistry()
    registry.register_placeholders()
    assert registry.list_names() == sorted(STAGE_NAMES)


def test_registry_registers_missing_manifest_stages() -> None:
    registry = StageRegistry()
    registry.register_placeholders()
    registry.register_from_manifest({"stages": {"run_retrain_pipeline": {}}})
    assert registry.has("run_retrain_pipeline")


def test_runner_dispatch_executes_stage_and_writes_report(tmp_path) -> None:
    manifest = {
        "stages": {
            "download_market_data": {
                "purpose": "stub",
                "deps": [],
                "outs": ["data/raw/market"],
                "reports": ["reports/download_market_data.json"],
            }
        }
    }
    params = {"stages": {"download_market_data": {"limit": 10}}}
    registry = StageRegistry()
    registry.register_placeholders()
    runner = PipelineRunner(workspace=tmp_path, manifest=manifest, params=params, registry=registry)

    result = runner.run_stage("download_market_data")
    assert result.success is True
    assert (tmp_path / "data/raw/market").exists()
    assert (tmp_path / "reports/download_market_data.json").exists()


def test_runner_tracks_missing_inputs_metric(tmp_path) -> None:
    manifest = {
        "stages": {
            "preprocess_market_data": {
                "purpose": "stub",
                "deps": ["data/raw/market/missing.csv"],
                "outs": ["data/interim/market"],
                "reports": [],
            }
        }
    }
    registry = StageRegistry()
    registry.register_placeholders()
    runner = PipelineRunner(workspace=tmp_path, manifest=manifest, params={}, registry=registry)

    result = runner.run_stage("preprocess_market_data")
    assert result.metrics["missing_inputs"] == 1.0


def test_runner_strict_inputs_raises(tmp_path) -> None:
    manifest = {
        "stages": {
            "preprocess_market_data": {
                "purpose": "stub",
                "deps": ["data/raw/market/missing.csv"],
                "outs": [],
                "reports": [],
            }
        }
    }
    registry = StageRegistry()
    registry.register_placeholders()
    runner = PipelineRunner(workspace=tmp_path, manifest=manifest, params={}, registry=registry)

    import pytest

    with pytest.raises(FileNotFoundError):
        runner.run_stage("preprocess_market_data", fail_on_missing_inputs=True)


def test_runner_loads_stage_params_from_params_yaml_shape(tmp_path) -> None:
    class ParamAwareStage(PipelineStage):
        def run(self, ctx: StageContext) -> StageResult:
            now = datetime.now(timezone.utc)
            return StageResult(
                run_id=ctx.run_id,
                stage_name=ctx.stage_name,
                success=True,
                started_at=now,
                finished_at=now,
                metrics={"stage_param_count": float(len(ctx.stage_params))},
                artifacts=[],
            )

    manifest = {
        "stages": {
            "custom_stage": {
                "purpose": "stub",
                "deps": [],
                "outs": [],
                "reports": [],
            }
        }
    }
    params = {"stages": {"custom_stage": {"foo": 1, "bar": 2}}}
    registry = StageRegistry()
    registry.register("custom_stage", ParamAwareStage)
    runner = PipelineRunner(workspace=tmp_path, manifest=manifest, params=params, registry=registry)

    result = runner.run_stage("custom_stage")
    assert result.metrics["stage_param_count"] == 2.0


def test_runner_rejects_paths_outside_workspace(tmp_path) -> None:
    manifest = {
        "stages": {
            "download_market_data": {
                "purpose": "stub",
                "deps": [],
                "outs": ["../outside"],
                "reports": [],
            }
        }
    }
    registry = StageRegistry()
    registry.register_placeholders()
    runner = PipelineRunner(workspace=tmp_path, manifest=manifest, params={}, registry=registry)

    import pytest

    with pytest.raises(ValueError):
        runner.run_stage("download_market_data")
