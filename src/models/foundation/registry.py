from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.models.foundation.chronos2_wrapper import Chronos2Wrapper
from src.models.foundation.deep_sequence_wrappers import PatchTSTWrapper, TFTWrapper, TimeXerWrapper
from src.models.foundation.moirai_wrapper import Moirai2Wrapper
from src.models.foundation.timesfm_wrapper import TimesFM2Wrapper

FORECASTER_FACTORY: dict[str, type] = {
    "chronos2": Chronos2Wrapper,
    "timesfm2": TimesFM2Wrapper,
    "moirai2": Moirai2Wrapper,
    "timexer": TimeXerWrapper,
    "tft": TFTWrapper,
    "patchtst": PatchTSTWrapper,
}


@dataclass(frozen=True, slots=True)
class ForecasterSpec:
    forecaster_type: str
    enabled: bool = True
    prediction_horizon: str = "60m"
    model_version: str = "v1"
    calibration_alpha: float = 0.1
    expected_return_scale: float = 1.0
    covariate_columns: tuple[str, ...] = ()
    extra_params: dict[str, Any] | None = None


def _tuple_of_strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(v) for v in value if isinstance(v, str) and v.strip())


def parse_forecaster_specs(params: dict[str, Any]) -> list[ForecasterSpec]:
    section = params.get("models_v2", {}).get("foundation_forecasters", [])
    if not isinstance(section, list):
        return []
    specs: list[ForecasterSpec] = []
    for item in section:
        if not isinstance(item, dict):
            continue
        ftype = str(item.get("type", "")).strip().lower()
        if not ftype:
            continue
        specs.append(
            ForecasterSpec(
                forecaster_type=ftype,
                enabled=bool(item.get("enabled", True)),
                prediction_horizon=str(item.get("prediction_horizon", "60m")),
                model_version=str(item.get("model_version", "v1")),
                calibration_alpha=float(item.get("calibration_alpha", 0.1)),
                expected_return_scale=float(item.get("expected_return_scale", 1.0)),
                covariate_columns=_tuple_of_strings(item.get("covariate_columns", [])),
                extra_params={
                    str(k): v
                    for k, v in item.items()
                    if str(k)
                    not in {
                        "type",
                        "enabled",
                        "prediction_horizon",
                        "model_version",
                        "calibration_alpha",
                        "expected_return_scale",
                        "covariate_columns",
                    }
                },
            )
        )
    return specs


def build_forecasters_from_params(params: dict[str, Any]) -> list:
    forecasters: list = []
    for spec in parse_forecaster_specs(params):
        if not spec.enabled:
            continue
        cls = FORECASTER_FACTORY.get(spec.forecaster_type)
        if cls is None:
            continue
        forecasters.append(
            cls(
                prediction_horizon=spec.prediction_horizon,
                model_version=spec.model_version,
                calibration_alpha=spec.calibration_alpha,
                expected_return_scale=spec.expected_return_scale,
                covariate_columns=spec.covariate_columns or cls().covariate_columns,
                **(spec.extra_params or {}),
            )
        )
    return forecasters
