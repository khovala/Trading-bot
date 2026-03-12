from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.models.base.interface import BaseTradingModel
from src.models.base.schemas import StandardizedPrediction
from src.models.base.serialization import clip01, load_pickle, save_pickle


@dataclass
class WeightedEnsembleModel(BaseTradingModel):
    model_name: str = "weighted_ensemble"
    model_version: str = "v1"
    prediction_horizon: str = "60m"
    weights: dict[str, float] = field(default_factory=dict)

    def fit(self, rows: list[dict[str, Any]], target_key: str = "return_1") -> dict[str, float]:
        _ = (rows, target_key)
        if not self.weights:
            self.weights = {"default": 1.0}
        cleaned = {k: max(0.0, float(v)) for k, v in self.weights.items()}
        total = sum(cleaned.values())
        if total <= 0:
            uniform = 1.0 / max(1, len(cleaned))
            self.weights = {k: uniform for k in cleaned}
        else:
            self.weights = {k: v / total for k, v in cleaned.items()}
        return {"weight_count": float(len(self.weights))}

    def combine(self, predictions_by_model: dict[str, list[StandardizedPrediction]]) -> list[StandardizedPrediction]:
        if not predictions_by_model:
            return []
        first = next(iter(predictions_by_model.values()))
        out: list[StandardizedPrediction] = []
        for i in range(len(first)):
            e = up = down = conf = total_w = 0.0
            for model_name, preds in predictions_by_model.items():
                w = float(self.weights.get(model_name, 0.0))
                if w <= 0 or i >= len(preds):
                    continue
                p = preds[i]
                e += w * p.expected_return
                up += w * p.direction_probability_up
                down += w * p.direction_probability_down
                conf += w * p.confidence
                total_w += w
            if total_w <= 0:
                total_w = 1.0
            out.append(
                StandardizedPrediction(
                    expected_return=e / total_w,
                    direction_probability_up=clip01(up / total_w),
                    direction_probability_down=clip01(down / total_w),
                    confidence=clip01(conf / total_w),
                    prediction_horizon=self.prediction_horizon,
                    model_name=self.model_name,
                    model_version=self.model_version,
                )
            )
        return out

    def predict(self, rows: list[dict[str, Any]]) -> list[StandardizedPrediction]:
        return [
            StandardizedPrediction(
                expected_return=0.0,
                direction_probability_up=0.5,
                direction_probability_down=0.5,
                confidence=0.0,
                prediction_horizon=self.prediction_horizon,
                model_name=self.model_name,
                model_version=self.model_version,
            )
            for _ in rows
        ]

    def save(self, path: Path) -> None:
        save_pickle(path, self)

    @classmethod
    def load(cls, path: Path) -> "WeightedEnsembleModel":
        obj = load_pickle(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj

    def get_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "prediction_horizon": self.prediction_horizon,
            "weights": self.weights,
        }
