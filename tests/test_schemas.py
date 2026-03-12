from __future__ import annotations

from datetime import datetime, timezone

from src.domain.enums import TradeAction
from src.domain.schemas import OrchestratorDecision, Prediction


def test_prediction_bounds() -> None:
    prediction = Prediction(
        expected_return=0.01,
        direction_probability_up=0.7,
        direction_probability_down=0.2,
        confidence=0.8,
        prediction_horizon="60m",
        model_name="baseline",
        model_version="v1",
    )
    assert prediction.confidence == 0.8


def test_orchestrator_decision_has_audit_id() -> None:
    decision = OrchestratorDecision(
        ticker="SBER",
        ts=datetime.now(timezone.utc),
        action=TradeAction.HOLD,
        confidence=0.6,
    )
    assert decision.decision_id
