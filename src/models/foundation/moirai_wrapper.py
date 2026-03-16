from __future__ import annotations

from dataclasses import dataclass

from src.models.foundation.base_forecaster import FoundationForecasterBase


@dataclass
class Moirai2Wrapper(FoundationForecasterBase):
    model_name: str = "moirai2_wrapper"
    model_version: str = "v2"
    backend_name: str = "moirai2"
    backend_available: bool = False
    use_covariates: bool = True
    covariate_columns: tuple[str, ...] = (
        "macd",
        "macd_signal",
        "atr_14",
        "news_recency_weighted_sentiment",
    )
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    calibration_alpha: float = 0.08
