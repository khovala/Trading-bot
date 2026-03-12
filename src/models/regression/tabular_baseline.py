from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.models.base.interface import BaseTradingModel
from src.models.base.schemas import StandardizedPrediction
from src.models.base.serialization import clip01, load_pickle, save_pickle


@dataclass
class TabularRegressionBaseline(BaseTradingModel):
    model_name: str = "tabular_regression_baseline"
    model_version: str = "v1"
    prediction_horizon: str = "60m"
    mean_return: float = 0.0
    std_return: float = 1.0
    up_prob: float = 0.5
    down_prob: float = 0.5
    fitted_at: str | None = None
    metrics_: dict[str, float] = field(default_factory=dict)

    def fit(self, rows: list[dict[str, Any]], target_key: str = "return_1") -> dict[str, float]:
        values = [float(r.get(target_key, 0.0)) for r in rows]
        if not values:
            self.mean_return = 0.0
            self.std_return = 1.0
            self.up_prob = 0.5
            self.down_prob = 0.5
        else:
            self.mean_return = sum(values) / len(values)
            variance = sum((v - self.mean_return) ** 2 for v in values) / len(values)
            self.std_return = max(1e-6, variance**0.5)
            self.up_prob = sum(1 for v in values if v > 0) / len(values)
            self.down_prob = sum(1 for v in values if v < 0) / len(values)
        self.fitted_at = datetime.now(timezone.utc).isoformat()
        self.metrics_ = {"train_mae_proxy": abs(self.mean_return), "train_samples": float(len(values))}
        return dict(self.metrics_)

    def predict(self, rows: list[dict[str, Any]]) -> list[StandardizedPrediction]:
        confidence = clip01(abs(self.mean_return) / (self.std_return + abs(self.mean_return) + 1e-9))
        return [
            StandardizedPrediction(
                expected_return=float(self.mean_return),
                direction_probability_up=clip01(self.up_prob),
                direction_probability_down=clip01(self.down_prob),
                confidence=confidence,
                prediction_horizon=self.prediction_horizon,
                model_name=self.model_name,
                model_version=self.model_version,
            )
            for _ in rows
        ]

    def save(self, path: Path) -> None:
        save_pickle(path, self)

    @classmethod
    def load(cls, path: Path) -> "TabularRegressionBaseline":
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
