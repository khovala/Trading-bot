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
    uncertainty_penalty: float = 0.25
    turnover_penalty: float = 0.10
    min_weight: float = 0.02
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def _normalize_weights(self, weights: dict[str, float]) -> dict[str, float]:
        if not weights:
            return {"default": 1.0}
        cleaned = {k: max(0.0, float(v)) for k, v in weights.items()}
        total = sum(cleaned.values())
        if total <= 0:
            uniform = 1.0 / max(1, len(cleaned))
            return {k: uniform for k in cleaned}
        return {k: v / total for k, v in cleaned.items()}

    def _turnover_proxy(self, preds: list[StandardizedPrediction]) -> float:
        if len(preds) < 2:
            return 0.0
        diffs = [
            abs(float(preds[i].expected_return) - float(preds[i - 1].expected_return))
            for i in range(1, len(preds))
        ]
        return sum(diffs) / len(diffs)

    def _build_dynamic_weights(self, predictions_by_model: dict[str, list[StandardizedPrediction]]) -> dict[str, float]:
        if not predictions_by_model:
            return self.weights
        base = self._normalize_weights(self.weights)
        raw_scores: dict[str, float] = {}
        stats: dict[str, Any] = {}
        for model_name, preds in predictions_by_model.items():
            if not preds:
                continue
            avg_conf = sum(float(p.confidence) for p in preds) / len(preds)
            avg_abs_ret = sum(abs(float(p.expected_return)) for p in preds) / len(preds)
            turnover = self._turnover_proxy(preds)
            uncertainty = max(0.0, 1.0 - avg_conf)
            score = float(base.get(model_name, self.min_weight))
            score *= max(1e-6, 1.0 - self.uncertainty_penalty * uncertainty)
            score *= max(1e-6, 1.0 - self.turnover_penalty * min(1.0, turnover * 100.0))
            score *= 1.0 + min(1.0, avg_abs_ret * 100.0) * 0.05
            raw_scores[model_name] = max(self.min_weight, score)
            stats[model_name] = {
                "avg_confidence": float(avg_conf),
                "avg_abs_expected_return": float(avg_abs_ret),
                "turnover_proxy": float(turnover),
                "uncertainty_proxy": float(uncertainty),
                "raw_score": float(raw_scores[model_name]),
            }
        normalized = self._normalize_weights(raw_scores or base)
        self.diagnostics = {
            "base_weights": base,
            "dynamic_scores": raw_scores,
            "model_stats": stats,
        }
        return normalized

    def fit(
        self,
        rows: list[dict[str, Any]],
        target_key: str = "return_1",
        *,
        predictions_by_model: dict[str, list[StandardizedPrediction]] | None = None,
    ) -> dict[str, float]:
        _ = (rows, target_key)
        self.weights = self._normalize_weights(self.weights or {"default": 1.0})
        if predictions_by_model:
            self.weights = self._build_dynamic_weights(predictions_by_model)
        return {
            "weight_count": float(len(self.weights)),
            "avg_weight": float(sum(self.weights.values()) / max(1, len(self.weights))),
        }

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
            "uncertainty_penalty": self.uncertainty_penalty,
            "turnover_penalty": self.turnover_penalty,
            "diagnostics": self.diagnostics,
        }
