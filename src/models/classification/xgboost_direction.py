from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


try:
    import numpy as np
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    np = None
    XGBClassifier = None


@dataclass
class XGBoostDirectionClassifier(BaseTradingModel):
    model_name: str = "xgboost_direction_classifier"
    model_version: str = "v2"
    prediction_horizon: str = "60m"
    fitted_at: str | None = None
    metrics_: dict[str, float] = field(default_factory=dict)

    _model: Any = field(default=None, repr=False)
    _feature_columns: tuple[str, ...] = field(default_factory=lambda: tuple(FEATURE_COLUMNS))
    _up_prob: float = 0.5
    _down_prob: float = 0.5
    _avg_abs_return: float = 0.0

    def fit(self, rows: list[dict[str, Any]], target_key: str = "return_1") -> dict[str, float]:
        if not XGBOOST_AVAILABLE or np is None:
            return self._fallback_fit(rows, target_key)

        X = np.array([_extract_features(r) for r in rows], dtype=np.float32)
        y_raw = np.array([_safe_float(r.get(target_key, 0.0)) for r in rows], dtype=np.float32)
        y = (y_raw > 0).astype(int)

        valid_mask = np.isfinite(X).all(axis=1)
        if not valid_mask.any() or len(np.unique(y[valid_mask])) < 2:
            return self._fallback_fit(rows, target_key)

        X, y = X[valid_mask], y[valid_mask]

        self._model = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0,
        )
        self._model.fit(X, y)

        proba = self._model.predict_proba(X)
        self._up_prob = float(np.mean(proba[:, 1])) if proba.shape[1] > 1 else 0.5
        self._down_prob = 1.0 - self._up_prob
        self._avg_abs_return = float(np.mean(np.abs(y_raw[valid_mask])))

        preds = self._model.predict(X)
        accuracy = float(np.mean(preds == y))

        self.fitted_at = datetime.now(timezone.utc).isoformat()
        self.metrics_ = {
            "train_accuracy": accuracy,
            "train_samples": float(len(y)),
            "up_prob": self._up_prob,
            "down_prob": self._down_prob,
        }
        return dict(self.metrics_)

    def _fallback_fit(self, rows: list[dict[str, Any]], target_key: str = "return_1") -> dict[str, float]:
        values = [_safe_float(r.get(target_key, 0.0)) for r in rows]
        if values:
            self._up_prob = sum(1 for v in values if v > 0) / len(values)
            self._down_prob = sum(1 for v in values if v < 0) / len(values)
            self._avg_abs_return = sum(abs(v) for v in values) / len(values)
        self.fitted_at = datetime.now(timezone.utc).isoformat()
        return {
            "binary_up_prob": float(self._up_prob),
            "binary_down_prob": float(self._down_prob),
            "train_samples": float(len(values)),
        }

    def predict(self, rows: list[dict[str, Any]]) -> list[StandardizedPrediction]:
        if self._model is None:
            return self._fallback_predict(rows)

        X = np.array([_extract_features(r) for r in rows], dtype=np.float32)
        valid_mask = np.isfinite(X).all(axis=1)

        predictions = []
        for i, row in enumerate(rows):
            if valid_mask[i]:
                proba = self._model.predict_proba(X[i : i + 1])[0]
                up_prob = float(proba[1]) if len(proba) > 1 else 0.5
            else:
                up_prob = self._up_prob

            down_prob = 1.0 - up_prob
            expected_return = (up_prob - down_prob) * self._avg_abs_return
            confidence = clip01(abs(up_prob - down_prob))

            predictions.append(
                StandardizedPrediction(
                    expected_return=expected_return,
                    direction_probability_up=clip01(up_prob),
                    direction_probability_down=clip01(down_prob),
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
        expected_return = (self._up_prob - self._down_prob) * self._avg_abs_return
        confidence = clip01(abs(self._up_prob - self._down_prob))
        return [
            StandardizedPrediction(
                expected_return=float(expected_return),
                direction_probability_up=clip01(self._up_prob),
                direction_probability_down=clip01(self._down_prob),
                confidence=confidence,
                prediction_horizon=self.prediction_horizon,
                model_name=self.model_name,
                model_version=self.model_version,
            )
            for _ in rows
        ]

    def predict_proba(self, rows: list[dict[str, Any]]) -> list[dict[str, float]]:
        if self._model is None:
            return [{"up": clip01(self._up_prob), "down": clip01(self._down_prob)} for _ in rows]

        X = np.array([_extract_features(r) for r in rows], dtype=np.float32)
        valid_mask = np.isfinite(X).all(axis=1)

        result = []
        for i, row in enumerate(rows):
            if valid_mask[i]:
                proba = self._model.predict_proba(X[i : i + 1])[0]
                up_prob = float(proba[1]) if len(proba) > 1 else 0.5
            else:
                up_prob = self._up_prob
            result.append({"up": clip01(up_prob), "down": clip01(1.0 - up_prob)})
        return result

    def save(self, path: Path) -> None:
        save_pickle(path, self)

    @classmethod
    def load(cls, path: Path) -> "XGBoostDirectionClassifier":
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
