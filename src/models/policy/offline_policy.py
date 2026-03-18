from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.models.base.serialization import clip01, load_pickle, save_pickle


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


@dataclass
class OfflinePolicyLayer:
    """
    Lightweight offline policy layer that approximates RL-style utility
    optimization using configurable penalties and confidence gating.
    """

    policy_name: str = "offline_policy_layer"
    policy_version: str = "v1"
    risk_aversion: float = 3.0
    turnover_penalty: float = 0.25
    drawdown_penalty: float = 0.20
    uncertainty_penalty: float = 0.35
    max_position: float = 1.0
    min_confidence: float = 0.50
    signal_deadband: float = 0.0005
    max_turnover_step: float = 0.20
    signal_to_position_scale: float = 500.0

    fitted_at: str | None = None
    train_samples: int = 0
    avg_abs_target: float = 0.0
    avg_volatility: float = 0.0
    avg_confidence: float = 0.0

    def fit(self, rows: list[dict[str, Any]], target_key: str = "return_1") -> dict[str, float]:
        self.train_samples = len(rows)
        targets = [_safe_float(r.get(target_key, 0.0)) for r in rows]
        vols = [_safe_float(r.get("rolling_volatility_20", 0.0)) for r in rows]
        confs = [_safe_float(r.get("confidence_proxy", 0.5), 0.5) for r in rows]
        self.avg_abs_target = (sum(abs(v) for v in targets) / len(targets)) if targets else 0.0
        self.avg_volatility = (sum(vols) / len(vols)) if vols else 0.0
        self.avg_confidence = (sum(confs) / len(confs)) if confs else 0.0
        self.fitted_at = datetime.now(timezone.utc).isoformat()
        return {
            "train_samples": float(self.train_samples),
            "avg_abs_target": float(self.avg_abs_target),
            "avg_volatility": float(self.avg_volatility),
            "avg_confidence": float(self.avg_confidence),
        }

    def _utility(
        self,
        *,
        signal: float,
        confidence: float,
        uncertainty: float,
        turnover: float,
        drawdown_proxy: float,
    ) -> float:
        reward = signal * confidence
        risk_cost = self.risk_aversion * drawdown_proxy + self.turnover_penalty * turnover
        uncertainty_cost = self.uncertainty_penalty * uncertainty
        return reward - risk_cost - self.drawdown_penalty * drawdown_proxy - uncertainty_cost

    def decide_batch(self, rows: list[dict[str, Any]]) -> list[dict[str, float]]:
        decisions: list[dict[str, float]] = []
        prev_position = 0.0
        for row in rows:
            signal = _safe_float(row.get("expected_return", row.get("momentum_10", 0.0)))
            if abs(signal) < self.signal_deadband:
                signal = 0.0
            
            mean_rev_signal = _safe_float(row.get("mean_reversion_direction", 0.0))
            if mean_rev_signal != 0.0 and _safe_float(row.get("mean_reversion_signal", 0.0)) > 0.0:
                signal = mean_rev_signal * 0.5 + signal * 0.5
            
            confidence = clip01(_safe_float(row.get("confidence", row.get("confidence_proxy", self.avg_confidence or 0.5))))
            uncertainty = 1.0 - confidence
            drawdown_proxy = max(0.0, _safe_float(row.get("rolling_volatility_20", self.avg_volatility)))
            raw_target = signal * self.signal_to_position_scale
            if confidence < self.min_confidence:
                raw_target = 0.0
            target_position = max(-self.max_position, min(self.max_position, raw_target))
            # Smooth position trajectory to control turnover spikes.
            delta = target_position - prev_position
            if abs(delta) > self.max_turnover_step:
                target_position = prev_position + (self.max_turnover_step if delta > 0 else -self.max_turnover_step)
            turnover = abs(target_position - prev_position)
            utility = self._utility(
                signal=signal,
                confidence=confidence,
                uncertainty=uncertainty,
                turnover=turnover,
                drawdown_proxy=drawdown_proxy,
            )
            decisions.append(
                {
                    "target_position": float(target_position),
                    "expected_utility": float(utility),
                    "turnover_proxy": float(turnover),
                    "uncertainty_proxy": float(uncertainty),
                    "confidence": float(confidence),
                }
            )
            prev_position = target_position
        return decisions

    def save(self, path: Path) -> None:
        save_pickle(path, self)

    @classmethod
    def load(cls, path: Path) -> "OfflinePolicyLayer":
        obj = load_pickle(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj

    def get_metadata(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "risk_aversion": self.risk_aversion,
            "turnover_penalty": self.turnover_penalty,
            "drawdown_penalty": self.drawdown_penalty,
            "uncertainty_penalty": self.uncertainty_penalty,
            "max_position": self.max_position,
            "min_confidence": self.min_confidence,
            "signal_deadband": self.signal_deadband,
            "max_turnover_step": self.max_turnover_step,
            "signal_to_position_scale": self.signal_to_position_scale,
            "fitted_at": self.fitted_at or "",
            "train_samples": self.train_samples,
        }
