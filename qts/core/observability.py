"""Observability domain models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from qts.core.order import OrderSide
from qts.core.portfolio import Portfolio


@dataclass(frozen=True, slots=True)
class ClosedTrade:
    """Completed round-trip trade."""

    ticker: str
    entry_time: datetime
    exit_time: datetime
    start_price: Decimal
    end_price: Decimal
    quantity: Decimal
    profit_pct: Decimal
    fee: Decimal
    side: OrderSide


@dataclass(frozen=True, slots=True)
class TokenSnapshot:
    """Held token details inside a portfolio snapshot."""

    token: str
    quantity: Decimal
    avg_buy_price: Decimal
    current_price: Decimal


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """Point-in-time portfolio state."""

    timestamp: datetime
    tokens: tuple[TokenSnapshot, ...]
    equity: Decimal


def snapshot_portfolio(portfolio: Portfolio, ts: datetime) -> PortfolioSnapshot:
    """Return an immutable snapshot of the current portfolio state."""
    tokens = tuple(
        TokenSnapshot(
            token=position.instrument.symbol,
            quantity=position.quantity,
            avg_buy_price=position.average_cost,
            current_price=position.market_price,
        )
        for position in portfolio.positions
        if position.quantity != Decimal("0")
    )
    return PortfolioSnapshot(timestamp=ts, tokens=tokens, equity=portfolio.total_value)
