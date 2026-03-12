from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.domain.enums import TradeAction


class Prediction(BaseModel):
    expected_return: float
    direction_probability_up: float = Field(ge=0.0, le=1.0)
    direction_probability_down: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    prediction_horizon: str
    model_name: str
    model_version: str


class ModelOutput(BaseModel):
    ticker: str
    ts: datetime
    prediction: Prediction
    extra: dict[str, Any] = Field(default_factory=dict)


class OrchestratorDecision(BaseModel):
    decision_id: str = Field(default_factory=lambda: str(uuid4()))
    ticker: str
    ts: datetime
    action: TradeAction
    confidence: float = Field(ge=0.0, le=1.0)
    contributing_models: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    explanation: dict[str, Any] = Field(default_factory=dict)


class StageResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str | None = None
    stage_name: str
    success: bool
    started_at: datetime
    finished_at: datetime
    metrics: dict[str, float] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)


class RawNewsItem(BaseModel):
    source: str
    published_at: datetime
    title: str
    text: str
    url: str
    language: str = "ru"


class TickerMappingResult(BaseModel):
    ticker: str | None = None
    issuer_name: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    method: str = "unmapped"
