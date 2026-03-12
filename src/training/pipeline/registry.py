from __future__ import annotations

from typing import Any, Type

from src.training.pipeline.base import PipelineStage, StageSpec
from src.training.stages.placeholders import PlaceholderStage


STAGE_NAMES: tuple[str, ...] = (
    "download_market_data",
    "preprocess_market_data",
    "download_news_data",
    "preprocess_news_data",
    "map_news_to_instruments",
    "generate_market_features",
    "generate_news_features",
    "merge_feature_sets",
    "train_base_models",
    "train_news_model",
    "train_ensemble_model",
    "evaluate_models",
    "backtest_strategy",
    "compare_with_production",
    "promote_model",
    "publish_artifacts",
    "generate_reports",
)


class StageRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, Type[PipelineStage]] = {}

    def register(self, stage_name: str, stage_cls: Type[PipelineStage]) -> None:
        if stage_name in self._registry:
            raise ValueError(f"Stage already registered: {stage_name}")
        self._registry[stage_name] = stage_cls

    def register_placeholders(self, stage_names: tuple[str, ...] = STAGE_NAMES) -> None:
        for stage_name in stage_names:
            if stage_name not in self._registry:
                self.register(stage_name, PlaceholderStage)

    def register_from_manifest(self, manifest: dict[str, Any]) -> None:
        stages = manifest.get("stages", {})
        if not isinstance(stages, dict):
            return
        for stage_name in stages:
            if isinstance(stage_name, str) and stage_name not in self._registry:
                self.register(stage_name, PlaceholderStage)

    def create(self, spec: StageSpec) -> PipelineStage:
        stage_cls = self._registry.get(spec.name)
        if stage_cls is None:
            raise KeyError(f"Stage not registered: {spec.name}")
        return stage_cls(spec)

    def has(self, stage_name: str) -> bool:
        return stage_name in self._registry

    def list_names(self) -> list[str]:
        return sorted(self._registry)
