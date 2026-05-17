from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import polars as pl
import pytest

from qts.core.events import EventBus, Tick
from qts.core.instrument import AssetType, Instrument
from qts.core.order import Fill, Order, OrderSide, OrderType
from qts.core.portfolio import Portfolio, Position
from qts.core.registry import Registry
from qts.research.strategies.base import BaseStrategy


def test_instrument_model():
    stock = Instrument("AAPL", AssetType.STOCK, "NASDAQ", "USD")
    vn_stock = Instrument("VN:VNM", AssetType.VN_STOCK, "HOSE", "VND")
    commodity = Instrument("CMX:CL", AssetType.COMMODITY, "NYMEX", "USD")
    crypto = Instrument("BTC/USDT", AssetType.CRYPTO, "BINANCE", "USDT")
    assert stock.asset_type is AssetType.STOCK
    assert vn_stock.asset_type is AssetType.VN_STOCK
    assert commodity.asset_type is AssetType.COMMODITY
    assert crypto.asset_type is AssetType.CRYPTO
    assert AssetType.from_symbol("PERP:ETH/USDT") is AssetType.CRYPTO_FUTURES
    assert AssetType.from_symbol("VNF:VN30F2503") is AssetType.VN_FUTURES
    assert AssetType.from_symbol("VNW:CVNM2403") is AssetType.VN_WARRANT
    assert AssetType.from_symbol("BTC/USDT") is AssetType.CRYPTO
    assert AssetType.from_symbol("VN:VNM") is AssetType.VN_STOCK
    assert AssetType.from_symbol("CMX:CL") is AssetType.COMMODITY
    assert AssetType.from_symbol("AAPL") is AssetType.STOCK
    assert len(list(AssetType)) == 7


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


class DummyStrategy(BaseStrategy):
    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        return self.empty_signal_frame()


def test_base_strategy_empty_signal_frame_schema():
    frame = DummyStrategy.empty_signal_frame()
    assert frame.columns == ["date", "symbol", "signal", "weight"]
    assert frame.schema == {
        "date": pl.Date,
        "symbol": pl.String,
        "signal": pl.Int32,
        "weight": pl.Float64,
    }


def test_base_strategy_validate_rejects_invalid_values():
    strategy = DummyStrategy()
    invalid_signal = pl.DataFrame(
        [{"date": datetime(2024, 1, 1).date(), "symbol": "AAPL", "signal": 2, "weight": 0.5}],
        schema={"date": pl.Date, "symbol": pl.String, "signal": pl.Int32, "weight": pl.Float64},
    )
    with pytest.raises(ValueError, match="signal"):
        strategy.validate_signal_frame(invalid_signal)

    invalid_weight = pl.DataFrame([{"date": datetime(2024, 1, 1).date(), "symbol": "AAPL", "signal": 1, "weight": 1.5}])
    with pytest.raises(ValueError, match="weight"):
        strategy.validate_signal_frame(invalid_weight)


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
