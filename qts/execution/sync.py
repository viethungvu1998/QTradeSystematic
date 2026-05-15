"""Position synchronisation."""

from __future__ import annotations

from decimal import Decimal

from qts.core.instrument import Instrument
from qts.core.order import Order, OrderSide, OrderType
from qts.core.portfolio import Position


class PositionSync:
    """Converts target weights into delta orders."""

    def compute_deltas(
        self,
        target_weights: dict[str, Decimal],
        current_positions: list[Position],
        instruments: dict[str, Instrument],
        latest_prices: dict[str, Decimal],
        account_value: Decimal,
    ) -> list[Order]:
        current_by_symbol = {position.instrument.symbol: position for position in current_positions}
        orders: list[Order] = []
        for symbol, target_weight in target_weights.items():
            instrument = instruments[symbol]
            price = latest_prices[symbol]
            target_quantity = (account_value * target_weight) / price if price else Decimal("0")
            current_quantity = current_by_symbol.get(symbol, Position(instrument, Decimal("0"), price)).quantity
            delta = target_quantity - current_quantity
            if delta == 0:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(
                Order(
                    instrument=instrument,
                    side=side,
                    order_type=OrderType.MARKET,
                    quantity=abs(delta),
                )
            )
        return orders
