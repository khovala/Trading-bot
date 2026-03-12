from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.models.base.schemas import StandardizedPrediction


class BaseTradingModel(ABC):
    @abstractmethod
    def fit(self, rows: list[dict[str, Any]], target_key: str = "return_1") -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, rows: list[dict[str, Any]]) -> list[StandardizedPrediction]:
        raise NotImplementedError

    def predict_proba(self, rows: list[dict[str, Any]]) -> list[dict[str, float]]:
        raise NotImplementedError("This model does not support predict_proba")

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseTradingModel":
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        raise NotImplementedError
