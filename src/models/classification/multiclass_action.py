from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.models.base.interface import BaseTradingModel
from src.models.base.schemas import StandardizedPrediction
from src.models.base.serialization import clip01, load_pickle, save_pickle


@dataclass
class MulticlassActionClassifier(BaseTradingModel):
    model_name: str = "multiclass_action_classifier"
    model_version: str = "v1"
    prediction_horizon: str = "60m"
    buy_prob: float = 0.33
    hold_prob: float = 0.34
    sell_prob: float = 0.33
    avg_abs_return: float = 0.0
    threshold: float = 0.001
    fitted_at: str | None = None

    def fit(self, rows: list[dict[str, Any]], target_key: str = "return_1") -> dict[str, float]:
        values = [float(r.get(target_key, 0.0)) for r in rows]
        if values:
            buy = sum(1 for v in values if v > self.threshold)
            sell = sum(1 for v in values if v < -self.threshold)
            hold = len(values) - buy - sell
            self.buy_prob = buy / len(values)
            self.sell_prob = sell / len(values)
            self.hold_prob = hold / len(values)
            self.avg_abs_return = sum(abs(v) for v in values) / len(values)
        self.fitted_at = datetime.now(timezone.utc).isoformat()
        return {
            "buy_prob": float(self.buy_prob),
            "hold_prob": float(self.hold_prob),
            "sell_prob": float(self.sell_prob),
            "train_samples": float(len(values)),
        }

    def predict(self, rows: list[dict[str, Any]]) -> list[StandardizedPrediction]:
        up = clip01(self.buy_prob + 0.5 * self.hold_prob)
        down = clip01(self.sell_prob + 0.5 * self.hold_prob)
        expected_return = (self.buy_prob - self.sell_prob) * self.avg_abs_return
        confidence = clip01(max(self.buy_prob, self.hold_prob, self.sell_prob))
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
        return [
            {"BUY": self.buy_prob, "HOLD": self.hold_prob, "SELL": self.sell_prob}
            for _ in rows
        ]

    def save(self, path: Path) -> None:
        save_pickle(path, self)

    @classmethod
    def load(cls, path: Path) -> "MulticlassActionClassifier":
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
