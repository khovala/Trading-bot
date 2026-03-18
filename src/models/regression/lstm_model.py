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


SEQUENCE_LENGTH = 20
FEATURE_COLUMNS = [
    "return_1",
    "rolling_volatility_20",
    "momentum_10",
    "rsi_14",
    "macd",
    "macd_signal",
    "atr_14",
    "zscore_20",
    "volume_ratio_20",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _extract_sequence_features(rows: list[dict[str, Any]]) -> np.ndarray:
    features = []
    for row in rows:
        feat = [_safe_float(row.get(col, 0.0)) for col in FEATURE_COLUMNS]
        features.append(feat)
    return np.array(features, dtype=np.float32)


@dataclass
class LSTMModel(BaseTradingModel):
    model_name: str = "lstm_regression"
    model_version: str = "v2"
    prediction_horizon: str = "60m"
    fitted_at: str | None = None
    metrics_: dict[str, float] = field(default_factory=dict)

    _model: Any = field(default=None, repr=False)
    _scaler_mean: np.ndarray = field(default=None, repr=False)
    _scaler_std: np.ndarray = field(default=None, repr=False)
    _sequence_length: int = field(default=SEQUENCE_LENGTH)

    def _build_model(self, input_dim: int) -> Any:
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping
        except ImportError:
            return None

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self._sequence_length, input_dim)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def fit(self, rows: list[dict[str, Any]], target_key: str = "return_1") -> dict[str, float]:
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping
        except ImportError:
            return self._fallback_fit(rows, target_key)

        if len(rows) < self._sequence_length + 10:
            return self._fallback_fit(rows, target_key)

        X_raw = _extract_sequence_features(rows)
        y = np.array([_safe_float(r.get(target_key, 0.0)) for r in rows], dtype=np.float32)

        if X_raw.ndim != 2:
            return self._fallback_fit(rows, target_key)

        valid_mask = np.isfinite(X_raw).all(axis=1) & np.isfinite(y)
        if not valid_mask[valid_mask].sum() > 100:
            return self._fallback_fit(rows, target_key)

        self._model = self._build_model(X_raw.shape[1])

        if self._model is None:
            return self._fallback_fit(rows, target_key)

        self._scaler_mean = np.nanmean(X_raw, axis=(0, 1), keepdims=True)
        self._scaler_std = np.nanstd(X_raw, axis=(0, 1), keepdims=True)
        self._scaler_std = np.where(self._scaler_std < 1e-6, 1.0, self._scaler_std)

        X_scaled = (X_raw - self._scaler_mean) / self._scaler_std

        X_seq, y_seq = [], []
        for i in range(self._sequence_length, len(X_scaled)):
            if valid_mask[i]:
                X_seq.append(X_scaled[i - self._sequence_length:i])
                y_seq.append(y[i])

        if len(X_seq) < 50:
            return self._fallback_fit(rows, target_key)

        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.float32)

        self._model = self._build_model(X_seq.shape[-1])

        if self._model is None:
            return self._fallback_fit(rows, target_key)

        try:
            from tensorflow.keras.callbacks import EarlyStopping
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            self._model.fit(
                X_seq, y_seq,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stop],
                verbose=0
            )
            preds = self._model.predict(X_seq, verbose=0).flatten()
            mae = float(np.mean(np.abs(y_seq - preds)))
        except Exception:
            return self._fallback_fit(rows, target_key)

        self.fitted_at = datetime.now(timezone.utc).isoformat()
        self.metrics_ = {
            "train_mae": mae,
            "train_samples": float(len(y_seq)),
            "sequence_length": float(self._sequence_length),
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
        if self._model is None or len(rows) < self._sequence_length:
            return self._fallback_predict(rows)

        X_raw = _extract_sequence_features(rows)
        if X_raw.shape[0] < self._sequence_length:
            return self._fallback_predict(rows)

        if self._scaler_mean is None or self._scaler_std is None:
            return self._fallback_predict(rows)

        X_scaled = (X_raw - self._scaler_mean) / self._scaler_std

        predictions = []
        for i in range(self._sequence_length - 1, len(X_scaled)):
            seq = X_scaled[i - self._sequence_length + 1:i + 1]
            seq = np.expand_dims(seq, axis=0)
            try:
                pred = float(self._model.predict(seq, verbose=0)[0, 0])
            except Exception:
                pred = getattr(self, 'mean_return', 0.0)

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

        if not predictions:
            return self._fallback_predict(rows)

        pad = [
            StandardizedPrediction(
                expected_return=getattr(self, 'mean_return', 0.0),
                direction_probability_up=0.5,
                direction_probability_down=0.5,
                confidence=0.1,
                prediction_horizon=self.prediction_horizon,
                model_name=self.model_name,
                model_version=self.model_version,
            )
        ] * (self._sequence_length - 1)
        return pad + predictions

    def _fallback_predict(self, rows: list[dict[str, Any]]) -> list[StandardizedPrediction]:
        mean_ret = getattr(self, 'mean_return', 0.0)
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
    def load(cls, path: Path) -> "LSTMModel":
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
            "sequence_length": self._sequence_length,
            "metrics": self.metrics_,
        }
