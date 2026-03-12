from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.models.base.interface import BaseTradingModel
from src.models.base.schemas import StandardizedPrediction
from src.models.base.serialization import clip01, load_pickle, save_pickle


@dataclass
class NewsFeatureModel(BaseTradingModel):
    model_name: str = "news_feature_model"
    model_version: str = "v1"
    prediction_horizon: str = "60m"
    coef_sentiment: float = 0.01
    bias: float = 0.0
    fitted_at: str | None = None

    def fit(self, rows: list[dict[str, Any]], target_key: str = "return_1") -> dict[str, float]:
        xs = [float(r.get("sentiment_mean", r.get("news_sentiment_mean", 0.0))) for r in rows]
        ys = [float(r.get(target_key, r.get("news_sentiment_mean", 0.0))) for r in rows]
        if xs and ys:
            x_mean = sum(xs) / len(xs)
            y_mean = sum(ys) / len(ys)
            var = sum((x - x_mean) ** 2 for x in xs)
            cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
            self.coef_sentiment = cov / var if var > 0 else 0.0
            self.bias = y_mean - self.coef_sentiment * x_mean
        self.fitted_at = datetime.now(timezone.utc).isoformat()
        return {
            "coef_sentiment": float(self.coef_sentiment),
            "bias": float(self.bias),
            "train_samples": float(len(xs)),
        }

    def predict(self, rows: list[dict[str, Any]]) -> list[StandardizedPrediction]:
        preds: list[StandardizedPrediction] = []
        for row in rows:
            sentiment = float(row.get("sentiment_mean", row.get("news_sentiment_mean", 0.0)))
            expected_return = self.bias + self.coef_sentiment * sentiment
            up = clip01(0.5 + expected_return * 10.0)
            down = clip01(1.0 - up)
            preds.append(
                StandardizedPrediction(
                    expected_return=float(expected_return),
                    direction_probability_up=up,
                    direction_probability_down=down,
                    confidence=clip01(abs(sentiment)),
                    prediction_horizon=self.prediction_horizon,
                    model_name=self.model_name,
                    model_version=self.model_version,
                )
            )
        return preds

    def save(self, path: Path) -> None:
        save_pickle(path, self)

    @classmethod
    def load(cls, path: Path) -> "NewsFeatureModel":
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
