"""Fill models."""

from __future__ import annotations

from decimal import Decimal

from qts.core.registry import Registry


@Registry.register_fill_model("immediate")
class ImmediateFill:
    """Uses the current close."""

    def get_fill_price(self, current_open: Decimal, current_close: Decimal, next_open: Decimal | None) -> Decimal:
        return current_close


@Registry.register_fill_model("next_open")
class NextOpenFill:
    """Uses the next bar open when available."""

    def get_fill_price(self, current_open: Decimal, current_close: Decimal, next_open: Decimal | None) -> Decimal:
        return next_open if next_open is not None else current_close


@Registry.register_fill_model("vwap")
class VWAPFill:
    """Approximates VWAP with the average of open and close."""

    def get_fill_price(self, current_open: Decimal, current_close: Decimal, next_open: Decimal | None) -> Decimal:
        return (current_open + current_close) / Decimal("2")
