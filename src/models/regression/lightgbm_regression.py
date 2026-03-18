from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.models.base.interface import BaseTradingModel
from src.models.base.schemas import StandardizedPrediction
from src.models.base.serialization import clip01, load_pickle, save_pickle


FEATURE_COLUMNS = [
    "rolling_volatility_20",
    "momentum_10",
    "rsi_14",
    "macd",
    "macd_signal",
    "atr_14",
    "zscore_20",
    "volume_ratio_20",
    "volume_zscore_20",
    "trend_regime",
    "volatility_regime",
    "return_lag_1",
    "return_lag_2",
    "return_lag_5",
    "volatility_lag_1",
    "rsi_lag_1",
    "macd_momentum_interaction",
    "volume_volatility_interaction",
]


def _extract_features(row: dict[str, Any]) -> list[float]:
    return [float(row.get(col, 0.0)) for col in FEATURE_COLUMNS]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


@dataclass
class LightGBMRegressionModel(BaseTradingModel):
    model_name: str = "lightgbm_regression"
    model_version: str = "v2"
    prediction_horizon: str = "60m"
    fitted_at: str | None = None
    metrics_: dict[str, float] = field(default_factory=dict)

    _model: Any = field(default=None, repr=False)
    _feature_columns: tuple[str, ...] = field(default_factory=lambda: tuple(FEATURE_COLUMNS))

    def fit(self, rows: list[dict[str, Any]], target_key: str = "return_1") -> dict[str, float]:
        from lightgbm import LGBMRegressor

        X = np.array([_extract_features(r) for r in rows], dtype=np.float32)
        y = np.array([_safe_float(r.get(target_key, 0.0)) for r in rows], dtype=np.float32)

        valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        if not valid_mask.any():
            return self._fallback_fit(rows, target_key)

        X, y = X[valid_mask], y[valid_mask]

        self._model = LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
        )
        self._model.fit(X, y)

        preds = self._model.predict(X)
        mae = float(np.mean(np.abs(y - preds)))
        mse = float(np.mean((y - preds) ** 2))

        self.fitted_at = datetime.now(timezone.utc).isoformat()
        self.metrics_ = {
            "train_mae": mae,
            "train_rmse": float(np.sqrt(mse)),
            "train_samples": float(len(y)),
            "feature_count": float(len(self._feature_columns)),
        }
        return dict(self.metrics_)

    def _fallback_fit(self, rows: list[dict[str, Any]], target_key: str = "return_1") -> dict[str, float]:
        values = [_safe_float(r.get(target_key, 0.0)) for r in rows]
        if values:
            self.mean_return = sum(values) / len(values)
            variance = sum((v - self.mean_return) ** 2 for v in values) / len(values)
            self.std_return = max(1e-6, variance**0.5)
        else:
            self.mean_return = 0.0
            self.std_return = 1.0
        self.fitted_at = datetime.now(timezone.utc).isoformat()
        self.metrics_ = {"train_mae_proxy": abs(self.mean_return), "train_samples": float(len(values))}
        return dict(self.metrics_)

    def predict(self, rows: list[dict[str, Any]]) -> list[StandardizedPrediction]:
        if self._model is None:
            return self._fallback_predict(rows)

        X = np.array([_extract_features(r) for r in rows], dtype=np.float32)
        valid_mask = np.isfinite(X).all(axis=1)

        predictions = []
        for i, row in enumerate(rows):
            if valid_mask[i]:
                pred = float(self._model.predict(X[i : i + 1])[0])
            else:
                pred = self.mean_return if hasattr(self, 'mean_return') else 0.0

            abs_pred = abs(pred)
            confidence = clip01(min(1.0, abs_pred * 50.0))

            predictions.append(
                StandardizedPrediction(
                    expected_return=float(pred),
                    direction_probability_up=clip01(0.5 + pred * 50.0),
                    direction_probability_down=clip01(0.5 - pred * 50.0),
                    confidence=confidence,
                    prediction_horizon=self.prediction_horizon,
                    model_name=self.model_name,
                    model_version=self.model_version,
                )
            )

        if not valid_mask.any():
            return self._fallback_predict(rows)

        return predictions

    def _fallback_predict(self, rows: list[dict[str, Any]]) -> list[StandardizedPrediction]:
        mean_ret = getattr(self, 'mean_return', 0.0)
        std_ret = getattr(self, 'std_return', 1.0)
        return [
            StandardizedPrediction(
                expected_return=float(mean_ret),
                direction_probability_up=clip01(0.5 + mean_ret * 50.0),
                direction_probability_down=clip01(0.5 - mean_ret * 50.0),
                confidence=clip01(abs(mean_ret) * 50.0),
                prediction_horizon=self.prediction_horizon,
                model_name=self.model_name,
                model_version=self.model_version,
            )
            for _ in rows
        ]

    def save(self, path: Path) -> None:
        save_pickle(path, self)

    @classmethod
    def load(cls, path: Path) -> "LightGBMRegressionModel":
        obj = load_pickle(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj

    def get_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "prediction_horizon": self.prediction_horizon,
            "fitted_at": self.fitted_at or "",
            "feature_columns": list(self._feature_columns),
            "metrics": self.metrics_,
        }
