"""Commission models."""

from __future__ import annotations

from decimal import Decimal

from qts.core.registry import Registry


@Registry.register_commission_model("percentage")
class PercentageCommission:
    """Percentage commission model."""

    def __init__(self, rate: Decimal = Decimal("0.001")) -> None:
        self.rate = rate

    def calculate(self, notional: Decimal) -> Decimal:
        return notional * self.rate


@Registry.register_commission_model("per_trade")
class PerTradeCommission:
    """Flat commission per fill."""

    def __init__(self, amount: Decimal = Decimal("1")) -> None:
        self.amount = amount

    def calculate(self, notional: Decimal) -> Decimal:
        return self.amount
