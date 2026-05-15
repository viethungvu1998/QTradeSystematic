"""Slippage models."""

from __future__ import annotations

from decimal import Decimal

from qts.core.registry import Registry


@Registry.register_slippage_model("fixed")
class FixedSlippage:
    """Applies a constant slippage rate."""

    def __init__(self, rate: Decimal = Decimal("0.0005")) -> None:
        self.rate = rate

    def apply(self, price: Decimal, atr: Decimal | None = None) -> Decimal:
        return price * (Decimal("1") + self.rate)


@Registry.register_slippage_model("volatility_scaled")
class VolatilityScaledSlippage:
    """Scales slippage by ATR."""

    def __init__(self, base_rate: Decimal = Decimal("0.0002")) -> None:
        self.base_rate = base_rate

    def apply(self, price: Decimal, atr: Decimal | None = None) -> Decimal:
        if atr is None or price == 0:
            return price * (Decimal("1") + self.base_rate)
        multiplier = Decimal("1") + (atr / price)
        return price * (Decimal("1") + self.base_rate * multiplier)
