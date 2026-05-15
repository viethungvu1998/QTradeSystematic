"""Order and fill models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum

from qts.core.instrument import Instrument


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(StrEnum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(slots=True)
class Order:
    """Order instruction."""

    instrument: Instrument
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    status: OrderStatus = OrderStatus.PENDING
    client_order_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True, slots=True)
class Fill:
    """Executed order details."""

    order_id: str
    instrument: Instrument
    side: OrderSide
    quantity: Decimal
    price: Decimal
    commission: Decimal = Decimal("0")
    filled_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
