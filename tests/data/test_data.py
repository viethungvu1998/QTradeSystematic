from __future__ import annotations

from datetime import date

import polars as pl

from qts.data.bundles.local import LocalBundleAdapter
from qts.data.manager import DataManager
from qts.data.sources.binance import BinanceDataSource
from qts.data.sources.fmp import FMPDataSource
from qts.data.sources.yahoo import YahooDataSource
from qts.data.storage.duckdb import DuckDBStorage
from qts.data.storage.parquet import ParquetStorage


def test_parquet_storage_roundtrip(tmp_path, stock_ohlcv):
    storage = ParquetStorage(tmp_path)
    storage.write("ohlcv", stock_ohlcv)
    loaded = storage.read("ohlcv")
    assert loaded.columns == ["date", "symbol", "open", "high", "low", "close", "volume"]


def test_duckdb_storage_append_deduplicates(stock_ohlcv):
    storage = DuckDBStorage()
    storage.write("prices", stock_ohlcv.head(5))
    storage.append("prices", stock_ohlcv.head(5))
    assert storage.read("prices").height == 5


async def test_data_manager_stock_and_crypto(tmp_path, stock_ohlcv, crypto_ohlcv):
    duck = DuckDBStorage()
    cache = ParquetStorage(tmp_path / "cache")
    bundle = LocalBundleAdapter(tmp_path / "bundle")
    manager = DataManager(
        stock_source=FMPDataSource(ohlcv_payloads={"AAPL": stock_ohlcv}),
        crypto_source=BinanceDataSource(ohlcv_payloads={"BTC/USDT": crypto_ohlcv}),
        storage=duck,
        cache=cache,
        bundle_adapter=bundle,
    )
    data = await manager.get_ohlcv(["AAPL", "BTC/USDT"], date(2024, 1, 1), date(2024, 3, 20))
    assert {"stock_prices", "crypto_prices"} <= set(duck.list_keys())
    assert bundle.exists("qts-stock-bundle")
    cached = await manager.get_ohlcv(["AAPL"], date(2024, 1, 1), date(2024, 3, 20))
    assert cached.height > 0
    assert data.columns == ["date", "symbol", "open", "high", "low", "close", "volume"]
