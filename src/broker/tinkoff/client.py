from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

import aiohttp

from src.config.settings import get_settings


class OrderType(str, Enum):
    MARKET = "Market"
    LIMIT = "Limit"
    BEST_PRICE = "BestPrice"


class OrderDirection(str, Enum):
    BUY = "Buy"
    SELL = "Sell"


class OrderStatus(str, Enum):
    NEW = "New"
    PARTIAL_FILL = "PartialFill"
    FILLED = "Filled"
    CANCELLED = "Cancelled"
    REJECTED = "Rejected"


@dataclass
class MarketCandle:
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    timestamp: datetime


@dataclass
class Order:
    order_id: str
    figi: str
    direction: OrderDirection
    order_type: OrderType
    status: OrderStatus
    lots_requested: int
    lots_executed: int
    price: Decimal | None
    created_at: datetime


@dataclass
class Portfolio:
    accounts: list[dict[str, Any]] = field(default_factory=list)
    positions: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class TinkoffConfig:
    token: str
    account_id: str | None = None
    sandbox: bool = True
    base_url: str = "https://sandbox-invest-public-api.tinkoff.ru"
    max_retries: int = 3
    retry_delay: float = 1.0

    def __post_init__(self):
        if not self.sandbox:
            self.base_url = "https://invest-public-api.tinkoff.ru"


class TinkoffClient:
    def __init__(self, config: TinkoffConfig):
        self.config = config
        self._session: aiohttp.ClientSession | None = None

    async def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.config.base_url}{path}"
        headers = await self._get_headers()

        for attempt in range(self.config.max_retries):
            try:
                async with self._session.request(
                    method, url, headers=headers, params=params, json=json_data
                ) as response:
                    data = await response.json()
                    if response.status == 200:
                        return data
                    elif response.status == 429:
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                        continue
                    else:
                        raise TinkoffAPIError(
                            f"API error {response.status}: {data}",
                            status_code=response.status,
                        )
            except aiohttp.ClientError as e:
                if attempt == self.config.max_retries - 1:
                    raise TinkoffAPIError(f"Request failed: {e}")
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        raise TinkoffAPIError("Max retries exceeded")

    async def connect(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def disconnect(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> "TinkoffClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

    async def get_accounts(self) -> list[dict[str, Any]]:
        result = await self._request("GET", "/v1/accounts")
        return result.get("accounts", [])

    async def get_portfolio(self, account_id: str | None = None) -> Portfolio:
        acc_id = account_id or self.config.account_id
        if not acc_id:
            accounts = await self.get_accounts()
            if accounts:
                acc_id = accounts[0]["id"]
            else:
                return Portfolio()

        result = await self._request("GET", f"/v1/portfolio", params={"brokerAccountId": acc_id})
        positions = {}
        for pos in result.get("positions", []):
            figi = pos.get("figi", "")
            positions[figi] = pos

        return Portfolio(accounts=await self.get_accounts(), positions=positions)

    async def get_candles(
        self,
        figi: str,
        from_time: datetime,
        to_time: datetime,
        interval: str = "1min",
    ) -> list[MarketCandle]:
        result = await self._request(
            "GET",
            f"/v1/candles",
            params={
                "figi": figi,
                "from": from_time.isoformat(),
                "to": to_time.isoformat(),
                "interval": interval,
            },
        )

        candles = []
        for c in result.get("candles", []):
            candles.append(
                MarketCandle(
                    open=Decimal(str(c.get("o", 0))),
                    high=Decimal(str(c.get("h", 0))),
                    low=Decimal(str(c.get("l", 0))),
                    close=Decimal(str(c.get("c", 0))),
                    volume=Decimal(str(c.get("v", 0))),
                    timestamp=datetime.fromisoformat(c.get("time", "").replace("Z", "+00:00")),
                )
            )
        return candles

    async def place_order(
        self,
        figi: str,
        direction: OrderDirection,
        quantity: int,
        order_type: OrderType = OrderType.BEST_PRICE,
        price: Decimal | None = None,
        account_id: str | None = None,
    ) -> Order:
        acc_id = account_id or self.config.account_id
        json_data: dict[str, Any] = {
            "figi": figi,
            "direction": direction.value,
            "quantity": quantity,
            "orderType": order_type.value,
        }
        if price:
            json_data["price"] = {"value": str(price), "currency": "RUB"}

        result = await self._request(
            "POST",
            "/v1/orders/limit-order",
            params={"brokerAccountId": acc_id} if acc_id else None,
            json_data=json_data,
        )

        return Order(
            order_id=result.get("orderId", ""),
            figi=figi,
            direction=direction,
            order_type=order_type,
            status=OrderStatus(result.get("status", "New")),
            lots_requested=quantity,
            lots_executed=result.get("executedLots", 0),
            price=price,
            created_at=datetime.now(timezone.utc),
        )

    async def cancel_order(self, order_id: str, account_id: str | None = None) -> bool:
        result = await self._request(
            "POST",
            f"/v1/orders/{order_id}/cancel",
            params={"brokerAccountId": account_id} if account_id else None,
        )
        return result.get("status", "") == "Ok"

    async def get_orders(self, account_id: str | None = None) -> list[Order]:
        acc_id = account_id or self.config.account_id
        result = await self._request(
            "GET",
            "/v1/orders",
            params={"brokerAccountId": acc_id} if acc_id else None,
        )

        orders = []
        for o in result.get("orders", []):
            orders.append(
                Order(
                    order_id=o.get("orderId", ""),
                    figi=o.get("figi", ""),
                    direction=OrderDirection(o.get("direction", "Buy")),
                    order_type=OrderType(o.get("orderType", "BestPrice")),
                    status=OrderStatus(o.get("status", "New")),
                    lots_requested=o.get("requestedLots", 0),
                    lots_executed=o.get("executedLots", 0),
                    price=Decimal(str(o.get("price", {}).get("value", 0))),
                    created_at=datetime.fromisoformat(
                        o.get("createdAt", "").replace("Z", "+00:00")
                    ),
                )
            )
        return orders

    async def get_instrument_by_ticker(self, ticker: str) -> dict[str, Any] | None:
        result = await self._request("GET", "/v1/securities", params={"ticker": ticker})
        instruments = result.get("securities", [])
        return instruments[0] if instruments else None


class TinkoffAPIError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def create_tinkoff_client(sandbox: bool = True) -> TinkoffClient:
    settings = get_settings()
    token = settings.tinkoff.tinkoff_token
    if not token:
        raise ValueError("TINKOFF_TOKEN not configured")

    config = TinkoffConfig(
        token=token,
        account_id=settings.tinkoff.tinkoff_account_id,
        sandbox=sandbox,
    )
    return TinkoffClient(config)


async def get_latest_candles(
    figi: str,
    interval: str = "1min",
    hours_back: int = 24,
) -> list[MarketCandle]:
    async with create_tinkoff_client() as client:
        to_time = datetime.now(timezone.utc)
        from_time = to_time - timedelta(hours=hours_back)
        return await client.get_candles(figi, from_time, to_time, interval)
