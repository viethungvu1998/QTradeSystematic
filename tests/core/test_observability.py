from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from qts.core.instrument import AssetType, Instrument
from qts.core.observability import ClosedTrade, TokenSnapshot, snapshot_portfolio
from qts.core.order import OrderSide
from qts.core.portfolio import Portfolio, Position


def test_closed_trade_construction_and_profit_signs():
    entry_time = datetime(2024, 1, 1, tzinfo=UTC)
    exit_time = datetime(2024, 1, 2, tzinfo=UTC)

    long_trade = ClosedTrade(
        ticker="AAPL",
        entry_time=entry_time,
        exit_time=exit_time,
        start_price=Decimal("100"),
        end_price=Decimal("110"),
        quantity=Decimal("2"),
        profit_pct=Decimal("0.10"),
        fee=Decimal("1.25"),
        side=OrderSide.BUY,
    )
    short_trade = ClosedTrade(
        ticker="AAPL",
        entry_time=entry_time,
        exit_time=exit_time,
        start_price=Decimal("100"),
        end_price=Decimal("90"),
        quantity=Decimal("2"),
        profit_pct=Decimal("-0.10"),
        fee=Decimal("1.25"),
        side=OrderSide.SELL,
    )

    assert long_trade.quantity == Decimal("2")
    assert long_trade.profit_pct == (
        long_trade.end_price - long_trade.start_price
    ) / long_trade.start_price
    assert short_trade.profit_pct == (
        short_trade.end_price - short_trade.start_price
    ) / short_trade.start_price
    assert long_trade.side is OrderSide.BUY
    assert short_trade.side is OrderSide.SELL


def test_snapshot_portfolio_returns_tokens_and_equity():
    aapl = Instrument("AAPL", AssetType.STOCK, "NASDAQ", "USD")
    msft = Instrument("MSFT", AssetType.STOCK, "NASDAQ", "USD")
    cash = Decimal("10")
    portfolio = Portfolio(
        positions=[
            Position(aapl, Decimal("2"), Decimal("50"), average_cost=Decimal("45")),
            Position(msft, Decimal("0"), Decimal("20"), average_cost=Decimal("18")),
        ],
        cash=cash,
    )
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)

    snapshot = snapshot_portfolio(portfolio, timestamp)

    assert snapshot.timestamp == timestamp
    assert snapshot.tokens == (
        TokenSnapshot(
            token="AAPL",
            quantity=Decimal("2"),
            avg_buy_price=Decimal("45"),
            current_price=Decimal("50"),
        ),
    )
    assert snapshot.equity == Decimal("110")
