"""F-32 paper verify: BinanceBroker against the Binance testnet.

Requires testnet API keys generated at https://testnet.binance.vision
(GitHub login → "Generate HMAC_SHA256 Key").
Store them in .env as:
    BINANCE_TESTNET_API_KEY=<key>
    BINANCE_TESTNET_SECRET_KEY=<secret>

NOTE: BINANCE_DEMO_TRADING_* keys from binance.com paper trading do NOT work
here — those are for a different environment. The testnet.binance.vision keys
are separate and free to generate.

Run with: pytest -m paper
"""

from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path

import pytest

from qts.core.instrument import AssetType, Instrument
from qts.core.order import Order, OrderSide, OrderType
from qts.execution.brokers.binance import BinanceBroker

_ENV_FILE = Path(__file__).parent.parent.parent / ".env"
_TESTNET_URL = "https://testnet.binance.vision"


def _load_env() -> None:
    """Load .env if dotenv is available; silently skip if not."""
    try:
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv(_ENV_FILE)
    except ImportError:
        pass


def _testnet_broker() -> BinanceBroker | None:
    """Return a BinanceBroker pointed at testnet, or None if keys are missing."""
    _load_env()
    api_key = os.environ.get("BINANCE_TESTNET_API_KEY")
    secret = os.environ.get("BINANCE_TESTNET_SECRET_KEY")
    if not api_key or not secret:
        return None
    from binance.spot import Spot  # noqa: PLC0415

    client = Spot(api_key=api_key, api_secret=secret, base_url=_TESTNET_URL)
    return BinanceBroker(client=client)


_SKIP_NO_TESTNET_KEYS = pytest.mark.skipif(
    _testnet_broker() is None,
    reason=(
        "Testnet keys not found. Generate at https://testnet.binance.vision "
        "and set BINANCE_TESTNET_API_KEY + BINANCE_TESTNET_SECRET_KEY in .env"
    ),
)


@pytest.fixture(scope="module")
def broker() -> BinanceBroker:
    b = _testnet_broker()
    if b is None:
        pytest.skip("Testnet keys not available — see module docstring")
    return b


@pytest.mark.paper
@_SKIP_NO_TESTNET_KEYS
async def test_binance_broker_connect(broker: BinanceBroker) -> None:
    """F-32: connect() succeeds and sets connected=True."""
    await broker.connect()
    assert broker.connected


@pytest.mark.paper
@_SKIP_NO_TESTNET_KEYS
async def test_binance_broker_get_account_value(broker: BinanceBroker) -> None:
    """F-32: get_account_value() returns a non-negative Decimal."""
    await broker.connect()
    value = await broker.get_account_value()
    assert isinstance(value, Decimal)
    assert value >= 0


@pytest.mark.paper
@_SKIP_NO_TESTNET_KEYS
async def test_binance_broker_get_positions(broker: BinanceBroker) -> None:
    """F-32: get_positions() returns a list; each Position has Decimal fields."""
    from qts.core.portfolio import Position

    await broker.connect()
    positions = await broker.get_positions()

    assert isinstance(positions, list)
    for pos in positions:
        assert isinstance(pos, Position)
        assert isinstance(pos.quantity, Decimal)
        assert isinstance(pos.market_price, Decimal)


@pytest.mark.paper
@_SKIP_NO_TESTNET_KEYS
async def test_binance_broker_place_limit_order(broker: BinanceBroker) -> None:
    """F-32: place_order(BTC/USDT, qty=0.001, LIMIT) → Fill with Decimal fields."""
    from binance.spot import Spot  # noqa: PLC0415

    await broker.connect()

    # Get current market price to set a realistic limit
    spot = Spot(base_url=_TESTNET_URL)
    ticker = spot.ticker_price("BTCUSDT")
    market_price = Decimal(ticker["price"])
    # Place 1 % below market so it rests in the book (won't fill immediately)
    limit_price = (market_price * Decimal("0.99")).quantize(Decimal("0.01"))

    instrument = Instrument("BTC/USDT", AssetType.CRYPTO, "BINANCE", "USDT")
    order = Order(
        instrument=instrument,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("0.001"),
        limit_price=limit_price,
    )
    fill = await broker.place_order(order)

    assert fill.instrument.symbol == "BTC/USDT"
    assert isinstance(fill.quantity, Decimal)
    assert isinstance(fill.price, Decimal)
    assert isinstance(fill.commission, Decimal)
    assert fill.order_id != ""

    # Clean up — cancel the resting order
    await broker.cancel_order(fill.order_id)
