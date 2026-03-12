from __future__ import annotations

from src.training.pipeline.base import StageSpec
from src.training.pipeline.registry import StageRegistry
from src.training.stages.mlflow_stages import TrainBaseModelsStage
from src.training.stages.register import register_default_stages


def test_mlflow_stages_registered_as_concrete() -> None:
    registry = StageRegistry()
    register_default_stages(registry)
    stage = registry.create(StageSpec(name="train_base_models", purpose="x"))
    assert isinstance(stage, TrainBaseModelsStage)
