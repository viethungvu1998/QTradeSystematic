"""Order router."""

from __future__ import annotations

import asyncio

from qts.core.instrument import AssetType
from qts.core.order import Fill, Order
from qts.execution.base import BaseBroker


class OrderRouter:
    """Routes orders by asset type."""

    def __init__(self, brokers: dict[AssetType, BaseBroker]) -> None:
        self.brokers = brokers

    async def execute(self, orders: list[Order]) -> list[Fill]:
        return await asyncio.gather(*(self._dispatch(order) for order in orders))

    async def _dispatch(self, order: Order) -> Fill:
        broker = self.brokers[order.instrument.asset_type]
        return await broker.place_order(order)
