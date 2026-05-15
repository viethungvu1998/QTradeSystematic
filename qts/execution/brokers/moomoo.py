"""Moomoo broker adapter."""

from __future__ import annotations

from decimal import Decimal

from qts.core.errors import BrokerError
from qts.core.order import Fill, Order, OrderSide
from qts.core.portfolio import Position
from qts.core.registry import Registry
from qts.execution.base import BaseBroker


@Registry.register_broker("moomoo")
class MoomooBroker(BaseBroker):
    """Fixture-friendly Moomoo adapter."""

    def __init__(self, client=None) -> None:
        self.client = client
        self.connected = False

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    async def get_positions(self) -> list[Position]:
        if self.client is None:
            return []
        return self.client.get_positions()

    async def place_order(self, order: Order) -> Fill:
        if self.client is None:
            return Fill(
                order_id=order.client_order_id or "paper-moomoo",
                instrument=order.instrument,
                side=order.side,
                quantity=order.quantity,
                price=order.limit_price or Decimal("0"),
            )
        response = self.client.place_order(order)
        if not response.get("success", True):
            raise BrokerError(response.get("message", "Moomoo order failed"), order)
        return Fill(
            order_id=str(response["order_id"]),
            instrument=order.instrument,
            side=order.side,
            quantity=Decimal(str(response["quantity"])),
            price=Decimal(str(response["price"])),
        )

    async def cancel_order(self, order_id: str) -> None:
        if self.client is not None:
            self.client.cancel_order(order_id)

    async def get_account_value(self) -> Decimal:
        if self.client is None:
            return Decimal("0")
        return Decimal(str(self.client.get_account_value()))
