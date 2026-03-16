from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.models.base.interface import BaseTradingModel
from src.models.base.schemas import StandardizedPrediction
from src.models.base.serialization import clip01, load_pickle, save_pickle


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


@dataclass
class FoundationForecasterBase(BaseTradingModel):
    """
    Config-driven forecaster abstraction used by foundation model wrappers.

    The class supports optional external backend usage and a robust statistical
    fallback that remains executable in local/offline research environments.
    """

    model_name: str
    model_version: str = "v1"
    prediction_horizon: str = "60m"
    backend_name: str = "internal_proxy"
    backend_available: bool = False
    use_covariates: bool = True
    covariate_columns: tuple[str, ...] = ()
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    calibration_alpha: float = 0.10
    expected_return_scale: float = 1.0

    # Fitted parameters
    fitted_at: str | None = None
    mean_target: float = 0.0
    std_target: float = 1.0
    last_target: float = 0.0
    covariate_weights: dict[str, float] = field(default_factory=dict)
    quantile_offsets: dict[float, float] = field(default_factory=dict)
    train_samples: int = 0

    def _compute_quantiles(self, values: list[float]) -> dict[float, float]:
        if not values:
            return {q: 0.0 for q in self.quantiles}
        ordered = sorted(values)
        n = len(ordered)
        out: dict[float, float] = {}
        for q in self.quantiles:
            qq = min(1.0, max(0.0, float(q)))
            idx = int(round((n - 1) * qq))
            out[qq] = ordered[idx]
        return out

    def fit(self, rows: list[dict[str, Any]], target_key: str = "return_1") -> dict[str, float]:
        targets = [_safe_float(r.get(target_key, 0.0)) for r in rows]
        self.train_samples = len(targets)
        if targets:
            self.mean_target = sum(targets) / len(targets)
            variance = sum((x - self.mean_target) ** 2 for x in targets) / len(targets)
            self.std_target = max(1e-6, variance**0.5)
            self.last_target = targets[-1]
            qvals = self._compute_quantiles(targets)
            self.quantile_offsets = {q: v - self.mean_target for q, v in qvals.items()}
        else:
            self.mean_target = 0.0
            self.std_target = 1.0
            self.last_target = 0.0
            self.quantile_offsets = {q: 0.0 for q in self.quantiles}

        self.covariate_weights = {}
        if self.use_covariates and rows and self.covariate_columns:
            denom = max(1.0, sum(abs(_safe_float(r.get(c, 0.0))) for r in rows for c in self.covariate_columns))
            for col in self.covariate_columns:
                num = sum(_safe_float(r.get(col, 0.0)) * _safe_float(r.get(target_key, 0.0)) for r in rows)
                self.covariate_weights[col] = num / denom

        self.fitted_at = datetime.now(timezone.utc).isoformat()
        return {
            "train_samples": float(self.train_samples),
            "train_mean_target": float(self.mean_target),
            "train_std_target": float(self.std_target),
            "backend_available": 1.0 if self.backend_available else 0.0,
        }

    def predict_distribution(self, rows: list[dict[str, Any]]) -> list[dict[str, float]]:
        out: list[dict[str, float]] = []
        for row in rows:
            covar_term = 0.0
            if self.use_covariates and self.covariate_weights:
                covar_term = sum(_safe_float(row.get(c, 0.0)) * w for c, w in self.covariate_weights.items())
            center = self.mean_target + self.calibration_alpha * self.last_target + covar_term
            sample = {str(q): float(center + self.quantile_offsets.get(float(q), 0.0)) for q in self.quantiles}
            out.append(sample)
        return out

    def predict(self, rows: list[dict[str, Any]]) -> list[StandardizedPrediction]:
        distributions = self.predict_distribution(rows)
        preds: list[StandardizedPrediction] = []
        for item in distributions:
            median = _safe_float(item.get("0.5", self.mean_target))
            lo = _safe_float(item.get("0.1", median - self.std_target))
            hi = _safe_float(item.get("0.9", median + self.std_target))
            spread = max(1e-9, hi - lo)
            confidence = clip01(1.0 - min(1.0, spread / (abs(median) + self.std_target + 1e-9)))
            up = clip01(0.5 + median / (2.0 * (self.std_target + 1e-9)))
            down = clip01(1.0 - up)
            preds.append(
                StandardizedPrediction(
                    expected_return=float(median * self.expected_return_scale),
                    direction_probability_up=up,
                    direction_probability_down=down,
                    confidence=confidence,
                    prediction_horizon=self.prediction_horizon,
                    model_name=self.model_name,
                    model_version=self.model_version,
                )
            )
        return preds

    def save(self, path: Path) -> None:
        save_pickle(path, self)

    @classmethod
    def load(cls, path: Path) -> "FoundationForecasterBase":
        obj = load_pickle(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj

    def get_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "prediction_horizon": self.prediction_horizon,
            "backend_name": self.backend_name,
            "backend_available": self.backend_available,
            "covariate_columns": list(self.covariate_columns),
            "quantiles": list(self.quantiles),
            "calibration_alpha": self.calibration_alpha,
            "fitted_at": self.fitted_at or "",
            "train_samples": self.train_samples,
        }
