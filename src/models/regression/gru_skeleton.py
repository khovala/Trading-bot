from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.models.base.interface import BaseTradingModel
from src.models.base.schemas import StandardizedPrediction
from src.models.base.serialization import clip01, load_pickle, save_pickle


@dataclass
class GRURegressionSkeleton(BaseTradingModel):
    model_name: str = "gru_regression_skeleton"
    model_version: str = "v1"
    prediction_horizon: str = "60m"
    smoothing_alpha: float = 0.2
    smoothed_return: float = 0.0
    fitted_at: str | None = None

    def fit(self, rows: list[dict[str, Any]], target_key: str = "return_1") -> dict[str, float]:
        values = [float(r.get(target_key, 0.0)) for r in rows]
        state = 0.0
        for v in values:
            state = self.smoothing_alpha * v + (1.0 - self.smoothing_alpha) * state
        self.smoothed_return = state
        self.fitted_at = datetime.now(timezone.utc).isoformat()
        return {"train_state_value": float(state), "train_samples": float(len(values))}

    def predict(self, rows: list[dict[str, Any]]) -> list[StandardizedPrediction]:
        up = clip01(0.5 + self.smoothed_return * 10.0)
        down = clip01(1.0 - up)
        return [
            StandardizedPrediction(
                expected_return=float(self.smoothed_return),
                direction_probability_up=up,
                direction_probability_down=down,
                confidence=clip01(abs(self.smoothed_return) * 20.0),
                prediction_horizon=self.prediction_horizon,
                model_name=self.model_name,
                model_version=self.model_version,
            )
            for _ in rows
        ]

    def save(self, path: Path) -> None:
        save_pickle(path, self)

    @classmethod
    def load(cls, path: Path) -> "GRURegressionSkeleton":
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
