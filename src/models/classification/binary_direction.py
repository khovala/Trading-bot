from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.models.base.interface import BaseTradingModel
from src.models.base.schemas import StandardizedPrediction
from src.models.base.serialization import clip01, load_pickle, save_pickle


@dataclass
class BinaryDirectionClassifier(BaseTradingModel):
    model_name: str = "binary_direction_classifier"
    model_version: str = "v1"
    prediction_horizon: str = "60m"
    up_prob: float = 0.5
    down_prob: float = 0.5
    avg_abs_return: float = 0.0
    fitted_at: str | None = None

    def fit(self, rows: list[dict[str, Any]], target_key: str = "return_1") -> dict[str, float]:
        values = [float(r.get(target_key, 0.0)) for r in rows]
        if values:
            self.up_prob = sum(1 for v in values if v > 0) / len(values)
            self.down_prob = sum(1 for v in values if v < 0) / len(values)
            self.avg_abs_return = sum(abs(v) for v in values) / len(values)
        self.fitted_at = datetime.now(timezone.utc).isoformat()
        return {
            "binary_up_prob": float(self.up_prob),
            "binary_down_prob": float(self.down_prob),
            "train_samples": float(len(values)),
        }

    def predict(self, rows: list[dict[str, Any]]) -> list[StandardizedPrediction]:
        expected_return = (self.up_prob - self.down_prob) * self.avg_abs_return
        confidence = clip01(abs(self.up_prob - self.down_prob))
        return [
            StandardizedPrediction(
                expected_return=float(expected_return),
                direction_probability_up=clip01(self.up_prob),
                direction_probability_down=clip01(self.down_prob),
                confidence=confidence,
                prediction_horizon=self.prediction_horizon,
                model_name=self.model_name,
                model_version=self.model_version,
            )
            for _ in rows
        ]

    def predict_proba(self, rows: list[dict[str, Any]]) -> list[dict[str, float]]:
        return [{"up": clip01(self.up_prob), "down": clip01(self.down_prob)} for _ in rows]

    def save(self, path: Path) -> None:
        save_pickle(path, self)

    @classmethod
    def load(cls, path: Path) -> "BinaryDirectionClassifier":
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
        }
