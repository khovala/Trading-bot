from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(slots=True)
class TelegramNotifier:
    bot_token: str
    chat_id: str
    enabled: bool = False

    def send(self, message: str) -> None:
        if not self.enabled or not self.bot_token or not self.chat_id:
            return
        payload = urlencode({"chat_id": self.chat_id, "text": message}).encode("utf-8")
        req = Request(f"https://api.telegram.org/bot{self.bot_token}/sendMessage", data=payload, method="POST")
        try:
            with urlopen(req, timeout=5):  # nosec B310
                pass
        except Exception:
            return

    def alert_critical_drawdown(self, drawdown: float) -> None:
        self.send(f"ALERT: critical drawdown detected ({drawdown:.2%})")

    def alert_sudden_price_drop(self, ticker: str, pct: float) -> None:
        self.send(f"ALERT: sudden price drop {ticker} ({pct:.2%})")

    def alert_stale_data(self, stale_sources: Iterable[str]) -> None:
        self.send("ALERT: stale data for " + ", ".join(stale_sources))

    def alert_model_degradation(self, metric_name: str, value: float) -> None:
        self.send(f"ALERT: model degradation {metric_name}={value:.4f}")

    def alert_broker_failure(self, reason: str) -> None:
        self.send(f"ALERT: broker/API failure: {reason}")

    def alert_severe_negative_news(self, ticker: str, score: float) -> None:
        self.send(f"ALERT: severe negative news for {ticker}, score={score:.3f}")
