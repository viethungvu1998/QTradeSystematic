from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from qts.core.events import EventBus, Tick
from qts.core.instrument import AssetType, Instrument
from qts.core.order import Fill, Order, OrderSide, OrderType
from qts.core.portfolio import Portfolio, Position
from qts.core.registry import Registry


def test_instrument_model():
    stock = Instrument("AAPL", AssetType.STOCK, "NASDAQ", "USD")
    crypto = Instrument("BTC/USDT", AssetType.CRYPTO, "BINANCE", "USDT")
    assert stock.asset_type is AssetType.STOCK
    assert crypto.asset_type is AssetType.CRYPTO
    assert len(list(AssetType)) == 2


def test_order_and_fill_models():
    instrument = Instrument("AAPL", AssetType.STOCK, "NASDAQ", "USD")
    order_types = list(OrderType)
    assert len(order_types) == 4
    order = Order(instrument, OrderSide.BUY, OrderType.LIMIT, Decimal("10"), limit_price=Decimal("100"))
    fill = Fill("1", instrument, OrderSide.BUY, Decimal("10"), Decimal("100"))
    assert isinstance(order.quantity, Decimal)
    assert isinstance(fill.price, Decimal)


def test_portfolio_total_value():
    instrument = Instrument("AAPL", AssetType.STOCK, "NASDAQ", "USD")
    portfolio = Portfolio(
        positions=[Position(instrument, Decimal("2"), Decimal("50"))],
        cash=Decimal("10"),
    )
    assert portfolio.total_value == Decimal("110")


def test_registry_resolution_and_missing_key():
    @Registry.register_feature("stub_feature")
    class StubFeature:
        pass

    assert Registry.get_feature("stub_feature") is StubFeature
    with pytest.raises(Exception, match="missing_feature"):
        Registry.get_feature("missing_feature")


@pytest.mark.asyncio
async def test_event_bus_subscribe_unsubscribe():
    bus = EventBus()
    instrument = Instrument("AAPL", AssetType.STOCK, "NASDAQ", "USD")
    events = []

    async def handler(event):
        events.append(event)

    tick = Tick(instrument, Decimal("100"), datetime.now(timezone.utc))
    bus.subscribe(Tick, handler)
    await bus.emit(tick)
    bus.unsubscribe(Tick, handler)
    await bus.emit(tick)
    assert len(events) == 1
