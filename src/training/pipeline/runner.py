from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from src.domain.schemas import StageResult
from src.training.pipeline.base import StageContext, StageSpec
from src.training.pipeline.registry import StageRegistry
from src.training.pipeline.reports import write_stage_reports


class PipelineRunner:
    def __init__(
        self,
        *,
        workspace: Path,
        manifest: dict[str, Any],
        params: dict[str, Any],
        registry: StageRegistry,
        run_id: str | None = None,
    ) -> None:
        self.workspace = workspace
        self.manifest = manifest
        self.params = params
        self.registry = registry
        self.run_id = run_id or str(uuid4())

    def stage_spec(self, stage_name: str) -> StageSpec:
        stage_cfg = self.manifest.get("stages", {}).get(stage_name)
        if not isinstance(stage_cfg, dict):
            raise KeyError(f"Stage config not found: {stage_name}")

        inputs = stage_cfg.get("deps", [])
        outputs = stage_cfg.get("outs", [])
        reports = stage_cfg.get("reports", [])
        if not isinstance(inputs, list) or not isinstance(outputs, list) or not isinstance(reports, list):
            raise ValueError(f"Stage '{stage_name}' deps/outs/reports must be lists.")

        return StageSpec(
            name=stage_name,
            purpose=stage_cfg.get("purpose", ""),
            inputs=inputs,
            outputs=outputs,
            reports=reports,
            mlflow=stage_cfg.get("mlflow", {}),
        )

    def _resolve_path(self, value: str) -> Path:
        path = (self.workspace / value).resolve()
        workspace = self.workspace.resolve()
        if path != workspace and workspace not in path.parents:
            raise ValueError(f"Path escapes workspace: {value}")
        return path

    def validate_inputs(self, spec: StageSpec) -> list[str]:
        missing: list[str] = []
        for dep in spec.inputs:
            if not isinstance(dep, str):
                missing.append(str(dep))
                continue
            if not self._resolve_path(dep).exists():
                missing.append(dep)
        return missing

    def prepare_outputs(self, spec: StageSpec) -> None:
        for output in spec.outputs:
            if not isinstance(output, str):
                continue
            target = self._resolve_path(output)
            directory = target.parent if target.suffix else target
            directory.mkdir(parents=True, exist_ok=True)

    def stage_context(self, spec: StageSpec) -> StageContext:
        stage_params = self.params.get("stages", {}).get(spec.name, {})
        if not isinstance(stage_params, dict):
            stage_params = {}
        return StageContext(
            stage_name=spec.name,
            params=self.params,
            stage_params=stage_params,
            run_id=self.run_id,
            workspace=self.workspace,
            reports_dir=self.workspace / "reports",
            artifacts_dir=self.workspace / "artifacts",
        )

    def run_stage(self, stage_name: str, *, fail_on_missing_inputs: bool = False) -> StageResult:
        spec = self.stage_spec(stage_name)
        if not self.registry.has(stage_name):
            raise KeyError(f"Stage not registered: {stage_name}")

        missing_inputs = self.validate_inputs(spec)
        if missing_inputs and fail_on_missing_inputs:
            raise FileNotFoundError(f"Missing stage inputs for {stage_name}: {missing_inputs}")

        self.prepare_outputs(spec)
        ctx = self.stage_context(spec)
        ctx.reports_dir.mkdir(parents=True, exist_ok=True)
        ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)

        stage = self.registry.create(spec)
        result = stage.run(ctx)
        if missing_inputs:
            result.metrics["missing_inputs"] = float(len(missing_inputs))

        write_stage_reports(spec, result, self.workspace)
        return result
