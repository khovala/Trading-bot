from __future__ import annotations

from datetime import datetime, timezone

from src.domain.schemas import StageResult
from src.training.pipeline.base import PipelineStage, StageContext


class PlaceholderStage(PipelineStage):
    """Stub stage implementation for incremental phase delivery."""

    def run(self, ctx: StageContext) -> StageResult:
        now = datetime.now(tz=timezone.utc)
        return StageResult(
            run_id=ctx.run_id,
            stage_name=ctx.stage_name,
            success=True,
            started_at=now,
            finished_at=now,
            metrics={"placeholder": 1.0},
            artifacts=[],
        )
