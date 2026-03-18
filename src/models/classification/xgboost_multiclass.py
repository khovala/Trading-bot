from __future__ import annotations

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
class XGBoostMulticlassClassifier(BaseTradingModel):
    model_name: str = "xgboost_multiclass_classifier"
    model_version: str = "v2"
    prediction_horizon: str = "60m"
    fitted_at: str | None = None
    metrics_: dict[str, float] = field(default_factory=dict)

    _model: Any = field(default=None, repr=False)
    _feature_columns: tuple[str, ...] = field(default_factory=lambda: tuple(FEATURE_COLUMNS))
    _buy_prob: float = 0.33
    _hold_prob: float = 0.34
    _sell_prob: float = 0.33
    _avg_abs_return: float = 0.0
    _threshold: float = 0.001

    def _create_labels(self, values: list[float]) -> np.ndarray:
        labels = np.zeros(len(values), dtype=int)
        for i, v in enumerate(values):
            if v > self._threshold:
                labels[i] = 0  # BUY
            elif v < -self._threshold:
                labels[i] = 2  # SELL
            else:
                labels[i] = 1  # HOLD
        return labels

    def fit(self, rows: list[dict[str, Any]], target_key: str = "return_1") -> dict[str, float]:
        try:
            from xgboost import XGBClassifier
        except ImportError:
            return self._fallback_fit(rows, target_key)

        X = np.array([_extract_features(r) for r in rows], dtype=np.float32)
        y_raw = np.array([_safe_float(r.get(target_key, 0.0)) for r in rows], dtype=np.float32)
        y = self._create_labels(y_raw.tolist())

        valid_mask = np.isfinite(X).all(axis=1)
        unique_labels = np.unique(y[valid_mask]) if valid_mask.any() else np.array([])

        if len(unique_labels) < 2:
            return self._fallback_fit(rows, target_key)

        X_valid, y_valid = X[valid_mask], y[valid_mask]

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
            eval_metric='mlogloss',
            verbosity=0,
        )
        self._model.fit(X_valid, y_valid)

        proba = self._model.predict_proba(X_valid)
        self._buy_prob = float(np.mean(proba[:, 0])) if proba.shape[1] > 0 else 0.33
        self._hold_prob = float(np.mean(proba[:, 1])) if proba.shape[1] > 1 else 0.34
        self._sell_prob = float(np.mean(proba[:, 2])) if proba.shape[1] > 2 else 0.33
        self._avg_abs_return = float(np.mean(np.abs(y_raw[valid_mask])))

        preds = self._model.predict(X_valid)
        accuracy = float(np.mean(preds == y_valid))

        self.fitted_at = datetime.now(timezone.utc).isoformat()
        self.metrics_ = {
            "train_accuracy": accuracy,
            "train_samples": float(len(y_valid)),
            "buy_prob": self._buy_prob,
            "hold_prob": self._hold_prob,
            "sell_prob": self._sell_prob,
        }
        return dict(self.metrics_)

    def _fallback_fit(self, rows: list[dict[str, Any]], target_key: str = "return_1") -> dict[str, float]:
        values = [_safe_float(r.get(target_key, 0.0)) for r in rows]
        if values:
            buy = sum(1 for v in values if v > self._threshold)
            sell = sum(1 for v in values if v < -self._threshold)
            hold = len(values) - buy - sell
            self._buy_prob = buy / len(values)
            self._sell_prob = sell / len(values)
            self._hold_prob = hold / len(values)
            self._avg_abs_return = sum(abs(v) for v in values) / len(values)
        self.fitted_at = datetime.now(timezone.utc).isoformat()
        return {
            "buy_prob": float(self._buy_prob),
            "hold_prob": float(self._hold_prob),
            "sell_prob": float(self._sell_prob),
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
                buy_prob = float(proba[0]) if len(proba) > 0 else 0.33
                hold_prob = float(proba[1]) if len(proba) > 1 else 0.34
                sell_prob = float(proba[2]) if len(proba) > 2 else 0.33
            else:
                buy_prob, hold_prob, sell_prob = self._buy_prob, self._hold_prob, self._sell_prob

            up = clip01(buy_prob + 0.5 * hold_prob)
            down = clip01(sell_prob + 0.5 * hold_prob)
            expected_return = (buy_prob - sell_prob) * self._avg_abs_return
            confidence = clip01(max(buy_prob, hold_prob, sell_prob))

            predictions.append(
                StandardizedPrediction(
                    expected_return=expected_return,
                    direction_probability_up=up,
                    direction_probability_down=down,
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
        up = clip01(self._buy_prob + 0.5 * self._hold_prob)
        down = clip01(self._sell_prob + 0.5 * self._hold_prob)
        expected_return = (self._buy_prob - self._sell_prob) * self._avg_abs_return
        confidence = clip01(max(self._buy_prob, self._hold_prob, self._sell_prob))
        return [
            StandardizedPrediction(
                expected_return=float(expected_return),
                direction_probability_up=up,
                direction_probability_down=down,
                confidence=confidence,
                prediction_horizon=self.prediction_horizon,
                model_name=self.model_name,
                model_version=self.model_version,
            )
            for _ in rows
        ]

    def predict_proba(self, rows: list[dict[str, Any]]) -> list[dict[str, float]]:
        if self._model is None:
            return [{"BUY": self._buy_prob, "HOLD": self._hold_prob, "SELL": self._sell_prob} for _ in rows]

        X = np.array([_extract_features(r) for r in rows], dtype=np.float32)
        valid_mask = np.isfinite(X).all(axis=1)

        result = []
        for i, row in enumerate(rows):
            if valid_mask[i]:
                proba = self._model.predict_proba(X[i : i + 1])[0]
                result.append({
                    "BUY": clip01(float(proba[0])) if len(proba) > 0 else self._buy_prob,
                    "HOLD": clip01(float(proba[1])) if len(proba) > 1 else self._hold_prob,
                    "SELL": clip01(float(proba[2])) if len(proba) > 2 else self._sell_prob,
                })
            else:
                result.append({"BUY": self._buy_prob, "HOLD": self._hold_prob, "SELL": self._sell_prob})
        return result

    def save(self, path: Path) -> None:
        save_pickle(path, self)

    @classmethod
    def load(cls, path: Path) -> "XGBoostMulticlassClassifier":
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
