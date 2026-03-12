from __future__ import annotations

from pydantic import BaseModel, Field


class StandardizedPrediction(BaseModel):
    expected_return: float
    direction_probability_up: float = Field(ge=0.0, le=1.0)
    direction_probability_down: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    prediction_horizon: str
    model_name: str
    model_version: str
