"""Portfolio models."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from qts.core.instrument import Instrument


@dataclass(slots=True)
class Position:
    """Current holding in a single instrument."""

    instrument: Instrument
    quantity: Decimal
    market_price: Decimal
    average_cost: Decimal = Decimal("0")

    @property
    def market_value(self) -> Decimal:
        return self.quantity * self.market_price


@dataclass(slots=True)
class Portfolio:
    """Aggregate portfolio state."""

    positions: list[Position] = field(default_factory=list)
    cash: Decimal = Decimal("0")

    @property
    def total_value(self) -> Decimal:
        return self.cash + sum((position.market_value for position in self.positions), Decimal("0"))
