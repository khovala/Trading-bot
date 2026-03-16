from __future__ import annotations

from src.training.pipeline.registry import StageRegistry
from src.training.stages.evaluation_reporting import (
    BacktestStrategyStage,
    CompareWithProductionStage,
    EvaluateModelsStage,
    GenerateReportsStage,
    PromoteModelStage,
    PublishArtifactsStage,
)
from src.training.stages.foundation_models import TrainFoundationModelsStage, TrainNewsEncoderStage
from src.training.stages.features import GenerateMarketFeaturesStage, MergeFeatureSetsStage
from src.training.stages.market import DownloadMarketDataStage, PreprocessMarketDataStage
from src.training.stages.model_training import TrainBaseModelsStage, TrainEnsembleModelStage, TrainNewsModelStage
from src.training.stages.news import DownloadNewsDataStage, PreprocessNewsDataStage
from src.training.stages.news_features import GenerateNewsFeaturesStage, MapNewsToInstrumentsStage
from src.training.stages.policy_layer import TrainPolicyLayerStage
from src.training.stages.policy_tuning import TunePolicyBacktestStage


def register_default_stages(registry: StageRegistry) -> None:
    if not registry.has("download_market_data"):
        registry.register("download_market_data", DownloadMarketDataStage)
    if not registry.has("preprocess_market_data"):
        registry.register("preprocess_market_data", PreprocessMarketDataStage)
    if not registry.has("download_news_data"):
        registry.register("download_news_data", DownloadNewsDataStage)
    if not registry.has("preprocess_news_data"):
        registry.register("preprocess_news_data", PreprocessNewsDataStage)
    if not registry.has("map_news_to_instruments"):
        registry.register("map_news_to_instruments", MapNewsToInstrumentsStage)
    if not registry.has("generate_market_features"):
        registry.register("generate_market_features", GenerateMarketFeaturesStage)
    if not registry.has("generate_news_features"):
        registry.register("generate_news_features", GenerateNewsFeaturesStage)
    if not registry.has("merge_feature_sets"):
        registry.register("merge_feature_sets", MergeFeatureSetsStage)
    if not registry.has("train_news_encoder"):
        registry.register("train_news_encoder", TrainNewsEncoderStage)
    if not registry.has("train_foundation_models"):
        registry.register("train_foundation_models", TrainFoundationModelsStage)
    if not registry.has("train_base_models"):
        registry.register("train_base_models", TrainBaseModelsStage)
    if not registry.has("train_news_model"):
        registry.register("train_news_model", TrainNewsModelStage)
    if not registry.has("train_ensemble_model"):
        registry.register("train_ensemble_model", TrainEnsembleModelStage)
    if not registry.has("train_policy_layer"):
        registry.register("train_policy_layer", TrainPolicyLayerStage)
    if not registry.has("tune_policy_backtest"):
        registry.register("tune_policy_backtest", TunePolicyBacktestStage)
    if not registry.has("evaluate_models"):
        registry.register("evaluate_models", EvaluateModelsStage)
    if not registry.has("backtest_strategy"):
        registry.register("backtest_strategy", BacktestStrategyStage)
    if not registry.has("compare_with_production"):
        registry.register("compare_with_production", CompareWithProductionStage)
    if not registry.has("promote_model"):
        registry.register("promote_model", PromoteModelStage)
    if not registry.has("publish_artifacts"):
        registry.register("publish_artifacts", PublishArtifactsStage)
    if not registry.has("generate_reports"):
        registry.register("generate_reports", GenerateReportsStage)
    registry.register_placeholders()
