"""Typed async event bus."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from qts.core.instrument import Instrument
from qts.core.order import OrderStatus


@dataclass(frozen=True, slots=True)
class Tick:
    """Tick event payload."""

    instrument: Instrument
    price: Decimal
    timestamp: datetime
    volume: Decimal = Decimal("0")


@dataclass(frozen=True, slots=True)
class OrderUpdate:
    """Order update payload."""

    order_id: str
    instrument: Instrument
    status: OrderStatus
    filled_quantity: Decimal = Decimal("0")


EventHandler = Callable[[Any], Awaitable[None]]


class EventBus:
    """Simple async publish/subscribe bus."""

    def __init__(self) -> None:
        self._handlers: dict[type[Any], list[EventHandler]] = defaultdict(list)

    def subscribe(self, event_type: type[Any], handler: EventHandler) -> None:
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: type[Any], handler: EventHandler) -> None:
        handlers = self._handlers[event_type]
        if handler in handlers:
            handlers.remove(handler)

    async def emit(self, event: Any) -> None:
        handlers = list(self._handlers[type(event)])
        if handlers:
            await asyncio.gather(*(handler(event) for handler in handlers))
