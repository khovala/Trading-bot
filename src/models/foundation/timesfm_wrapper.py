from __future__ import annotations

from dataclasses import dataclass

from src.models.foundation.base_forecaster import FoundationForecasterBase


@dataclass
class TimesFM2Wrapper(FoundationForecasterBase):
    model_name: str = "timesfm2_wrapper"
    model_version: str = "v2"
    backend_name: str = "timesfm2"
    backend_available: bool = False
    use_covariates: bool = True
    covariate_columns: tuple[str, ...] = (
        "return_1",
        "rolling_volatility_20",
        "zscore_20",
    )
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    calibration_alpha: float = 0.10
