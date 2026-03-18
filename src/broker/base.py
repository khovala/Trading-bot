from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

from src.broker.tinkoff.client import (
    Order,
    OrderDirection,
    OrderType,
    TinkoffClient,
    create_tinkoff_client,
)
from src.config.settings import get_settings
from src.domain.enums import RunMode
from src.monitoring.metrics import ORDER_FAILURE_TOTAL, ORDER_SUBMISSION_TOTAL

if TYPE_CHECKING:
    from src.strategies.final_strategy import TradingSignal

logger = structlog.get_logger(__name__)


class SignalAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


@dataclass
class ExecutionResult:
    success: bool
    order_id: str | None = None
    message: str = ""
    executed_price: Decimal | None = None
    executed_lots: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class BrokerAdapter:
    broker: str = "tinkoff"
    _client: TinkoffClient | None = None

    @property
    def client(self) -> TinkoffClient:
        if self._client is None:
            self._client = create_tinkoff_client(sandbox=get_settings().trading_mode == RunMode.SANDBOX)
        return self._client

    async def connect(self) -> None:
        await self.client.connect()
        logger.info("broker_connected", broker=self.broker)

    async def disconnect(self) -> None:
        if self._client:
            await self._client.disconnect()
            self._client = None
        logger.info("broker_disconnected", broker=self.broker)

    async def execute_signal(self, signal: "TradingSignal", dry_run: bool = True) -> ExecutionResult:
        settings = get_settings()
        
        if dry_run or settings.trading_mode == RunMode.SANDBOX:
            logger.info("dry_run_order", signal=signal)
            return ExecutionResult(
                success=True,
                order_id=f"dry_run_{datetime.now().timestamp()}",
                message="Dry run - no actual order placed",
            )

        action = signal.action
        figi = signal.figi
        quantity = signal.quantity

        if action == SignalAction.HOLD:
            return ExecutionResult(success=True, message="Hold - no action")

        try:
            direction = OrderDirection.BUY if action == SignalAction.BUY else OrderDirection.SELL
            order_type = OrderType.BEST_PRICE if signal.price is None else OrderType.LIMIT
            price = signal.price

            order = await self.client.place_order(
                figi=figi,
                direction=direction,
                quantity=quantity,
                order_type=order_type,
                price=price,
            )

            ORDER_SUBMISSION_TOTAL.labels(status="success").inc()
            logger.info(
                "order_placed",
                order_id=order.order_id,
                figi=figi,
                direction=direction.value,
                quantity=quantity,
                price=str(price) if price else "MARKET",
            )

            return ExecutionResult(
                success=True,
                order_id=order.order_id,
                message=f"Order placed: {order.status.value}",
                executed_lots=order.lots_executed,
            )

        except Exception as e:
            ORDER_FAILURE_TOTAL.labels(reason=str(type(e).__name__)).inc()
            logger.error("order_failed", figi=figi, error=str(e))
            return ExecutionResult(
                success=False,
                message=f"Order failed: {str(e)}",
            )

    async def get_portfolio(self) -> dict[str, Any]:
        try:
            portfolio = await self.client.get_portfolio()
            return {
                "positions": portfolio.positions,
                "accounts": portfolio.accounts,
            }
        except Exception as e:
            logger.error("portfolio_fetch_failed", error=str(e))
            return {"positions": {}, "accounts": []}

    async def get_open_orders(self) -> list[Order]:
        try:
            return await self.client.get_orders()
        except Exception as e:
            logger.error("orders_fetch_failed", error=str(e))
            return []

    async def cancel_order(self, order_id: str) -> bool:
        try:
            return await self.client.cancel_order(order_id)
        except Exception as e:
            logger.error("order_cancel_failed", order_id=order_id, error=str(e))
            return False

    async def get_candles(self, figi: str, hours_back: int = 24) -> list[dict[str, Any]]:
        from src.broker.tinkoff.client import get_latest_candles

        try:
            candles = await get_latest_candles(figi, hours_back=hours_back)
            return [
                {
                    "open": float(c.open),
                    "high": float(c.high),
                    "low": float(c.low),
                    "close": float(c.close),
                    "volume": float(c.volume),
                    "timestamp": c.timestamp.isoformat(),
                }
                for c in candles
            ]
        except Exception as e:
            logger.error("candles_fetch_failed", figi=figi, error=str(e))
            return []

    def __enter__(self) -> "BrokerAdapter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass
