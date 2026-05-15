"""F-11 paper verify: BinanceDataSource against the Binance testnet.

Requires no credentials — klines is a public endpoint on testnet.binance.vision.
Run with: pytest -m paper
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from qts.data._schemas import OHLCV_COLUMNS
from qts.data.sources.binance import BinanceDataSource


@pytest.fixture(scope="module")
def testnet_source() -> BinanceDataSource:
    """Unauthenticated source pointing at testnet (klines is public)."""
    from binance.spot import Spot

    client = Spot(base_url="https://testnet.binance.vision")
    return BinanceDataSource(client=client)


@pytest.mark.paper
async def test_binance_data_source_ohlcv_schema(testnet_source: BinanceDataSource) -> None:
    """F-11: fetch daily OHLCV for BTC/USDT; assert schema and non-empty result."""
    end = date.today()
    start = end - timedelta(days=30)

    df = await testnet_source.get_ohlcv("BTC/USDT", start, end, "1d")

    assert df.columns == OHLCV_COLUMNS
    assert df.height > 0
    assert df["symbol"][0] == "BTC/USDT"


@pytest.mark.paper
async def test_binance_data_source_ohlcv_date_filter(testnet_source: BinanceDataSource) -> None:
    """F-11: date range filter returns only rows within [start, end]."""
    end = date.today()
    start = end - timedelta(days=7)

    df = await testnet_source.get_ohlcv("BTC/USDT", start, end, "1d")

    assert df.height > 0
    assert df["date"].min() >= start
    assert df["date"].max() <= end


@pytest.mark.paper
async def test_binance_data_source_ohlcv_types(testnet_source: BinanceDataSource) -> None:
    """F-11: numeric columns are float, date column is date, symbol is string."""
    import polars as pl

    end = date.today()
    start = end - timedelta(days=7)

    df = await testnet_source.get_ohlcv("BTC/USDT", start, end, "1d")

    assert df.schema["date"] == pl.Date
    assert df.schema["symbol"] == pl.Utf8
    for col in ("open", "high", "low", "close", "volume"):
        assert df.schema[col] == pl.Float64
    assert (df["close"] > 0).all()
    assert (df["volume"] > 0).all()
