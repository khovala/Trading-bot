"""News-specific models."""

from src.models.news.news_encoder_pipeline import NewsEncoderPipeline
from src.models.news.news_feature_model import NewsFeatureModel

__all__ = ["NewsFeatureModel", "NewsEncoderPipeline"]
