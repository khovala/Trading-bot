from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.domain.schemas import StageResult


@dataclass(slots=True)
class StageContext:
    stage_name: str
    params: dict[str, Any]
    stage_params: dict[str, Any]
    run_id: str
    workspace: Path
    reports_dir: Path
    artifacts_dir: Path


@dataclass(slots=True)
class StageSpec:
    name: str
    purpose: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    reports: list[str] = field(default_factory=list)
    mlflow: dict[str, bool] = field(default_factory=dict)

    @property
    def deps(self) -> list[str]:
        return self.inputs

    @property
    def outs(self) -> list[str]:
        return self.outputs


class PipelineStage(ABC):
    def __init__(self, spec: StageSpec) -> None:
        self.spec = spec

    @abstractmethod
    def run(self, ctx: StageContext) -> StageResult:
        raise NotImplementedError


class NoopStage(PipelineStage):
    def run(self, ctx: StageContext) -> StageResult:
        now = datetime.now(tz=timezone.utc)
        return StageResult(
            run_id=ctx.run_id,
            stage_name=ctx.stage_name,
            success=True,
            started_at=now,
            finished_at=now,
            metrics={"noop": 1.0},
            artifacts=[],
        )
