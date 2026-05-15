"""Broker abstractions."""

from __future__ import annotations

from decimal import Decimal

from qts.core.order import Fill, Order
from qts.core.portfolio import Position


class BaseBroker:
    """Broker contract."""

    async def connect(self) -> None:
        raise NotImplementedError

    async def disconnect(self) -> None:
        raise NotImplementedError

    async def get_positions(self) -> list[Position]:
        raise NotImplementedError

    async def place_order(self, order: Order) -> Fill:
        raise NotImplementedError

    async def cancel_order(self, order_id: str) -> None:
        raise NotImplementedError

    async def get_account_value(self) -> Decimal:
        raise NotImplementedError
