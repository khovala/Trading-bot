from __future__ import annotations

from dataclasses import dataclass

from src.models.foundation.base_forecaster import FoundationForecasterBase


@dataclass
class TimeXerWrapper(FoundationForecasterBase):
    model_name: str = "timexer_wrapper"
    model_version: str = "v1"
    backend_name: str = "timexer"
    backend_available: bool = False
    use_covariates: bool = True
    covariate_columns: tuple[str, ...] = (
        "return_1",
        "momentum_10",
        "rolling_volatility_20",
    )
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    calibration_alpha: float = 0.12


@dataclass
class TFTWrapper(FoundationForecasterBase):
    model_name: str = "tft_wrapper"
    model_version: str = "v1"
    backend_name: str = "tft"
    backend_available: bool = False
    use_covariates: bool = True
    covariate_columns: tuple[str, ...] = (
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "rolling_volatility_20",
    )
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    calibration_alpha: float = 0.09


@dataclass
class PatchTSTWrapper(FoundationForecasterBase):
    model_name: str = "patchtst_wrapper"
    model_version: str = "v1"
    backend_name: str = "patchtst"
    backend_available: bool = False
    use_covariates: bool = True
    covariate_columns: tuple[str, ...] = (
        "return_1",
        "return_5",
        "zscore_20",
    )
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    calibration_alpha: float = 0.07
