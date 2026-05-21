from __future__ import annotations

from decimal import Decimal

import pytest

from qts.core.instrument import AssetType, Instrument
from qts.core.order import Fill, Order, OrderSide, OrderType
from qts.core.portfolio import Position
from qts.execution.base import BaseBroker
from qts.execution.router import OrderRouter
from qts.execution.sync import PositionSync
from qts.orchestration.tasks.execution_tasks import sync_positions


class MockBroker(BaseBroker):
    def __init__(self) -> None:
        self.orders = []

    async def connect(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    async def get_positions(self):
        return []

    async def place_order(self, order: Order) -> Fill:
        self.orders.append(order)
        return Fill("1", order.instrument, order.side, order.quantity, Decimal("10"))

    async def cancel_order(self, order_id: str) -> None:
        return None

    async def get_account_value(self) -> Decimal:
        return Decimal("1000")


@pytest.mark.asyncio
async def test_order_router_dispatches_by_asset_type():
    stock = MockBroker()
    crypto = MockBroker()
    router = OrderRouter({AssetType.STOCK: stock, AssetType.CRYPTO: crypto})
    orders = [
        Order(
            Instrument("AAPL", AssetType.STOCK, "NASDAQ", "USD"),
            OrderSide.BUY,
            OrderType.MARKET,
            Decimal("1"),
        ),
        Order(
            Instrument("BTC/USDT", AssetType.CRYPTO, "BINANCE", "USDT"),
            OrderSide.SELL,
            OrderType.MARKET,
            Decimal("1"),
        ),
    ]
    await router.execute(orders)
    assert len(stock.orders) == 1
    assert len(crypto.orders) == 1


def test_position_sync_generates_delta_orders():
    instrument = Instrument("AAPL", AssetType.STOCK, "NASDAQ", "USD")
    position = Position(instrument, Decimal("5"), Decimal("10"))
    orders = PositionSync().compute_deltas(
        target_weights={"AAPL": Decimal("0.1")},
        current_positions=[position],
        instruments={"AAPL": instrument},
        latest_prices={"AAPL": Decimal("10")},
        account_value=Decimal("1000"),
    )
    assert orders[0].side is OrderSide.BUY


@pytest.mark.asyncio
async def test_sync_positions_creates_instruments_for_new_targets(stock_ohlcv):
    broker = MockBroker()
    orders, snapshot = await sync_positions(
        config=None,
        syncer=PositionSync(),
        brokers={AssetType.STOCK: broker},
        target_weights={"AAPL": Decimal("0.1")},
        data=stock_ohlcv,
    )
    assert len(orders) == 1
    assert orders[0].instrument.symbol == "AAPL"
    assert orders[0].instrument.currency == "USD"
    assert snapshot.equity == Decimal("0")


@pytest.mark.asyncio
async def test_sync_positions_creates_quote_currency_for_crypto_futures(crypto_futures_ohlcv):
    broker = MockBroker()
    orders, _ = await sync_positions(
        config=None,
        syncer=PositionSync(),
        brokers={AssetType.CRYPTO_FUTURES: broker},
        target_weights={"PERP:BTC/USDT": Decimal("0.1")},
        data=crypto_futures_ohlcv,
    )
    assert len(orders) == 1
    assert orders[0].instrument.asset_type is AssetType.CRYPTO_FUTURES
    assert orders[0].instrument.currency == "USDT"
