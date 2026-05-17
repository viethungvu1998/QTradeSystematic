from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from qts.core.events import Tick
from qts.data._schemas import DataType
from qts.data.base import BaseDataSource
from qts.data.bundles.local import LocalBundleAdapter
from qts.data.manager import DataManager
from qts.data.sources.binance import BinanceDataSource, BinanceFuturesDataSource
from qts.data.sources.dnse import DNSEDataSource
from qts.data.sources.fmp import FMPDataSource
from qts.data.sources.vnstock import VnstockDataSource, VnstockFuturesDataSource
from qts.data.sources.yahoo import YahooDataSource
from qts.core.errors import DataSourceError, DataSourceWarning
from qts.data.storage.duckdb import DuckDBStorage
from qts.data.storage.parquet import ParquetStorage


class MockSource(BaseDataSource):
    CAPABILITIES = frozenset({DataType.OHLCV})

    async def get_ohlcv(self, symbol, start, end, interval):
        raise NotImplementedError

    async def get_fundamentals(self, symbol):
        raise NotImplementedError

    async def stream_ticks(self, symbols: list[str]):
        raise NotImplementedError
        yield Tick  # pragma: no cover


class CountingSource(BaseDataSource):
    CAPABILITIES = frozenset({DataType.OHLCV})

    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame
        self.calls = 0

    async def fetch(self, data_type: DataType, symbol: str, **kwargs) -> pl.DataFrame:
        self.calls += 1
        return self.frame.filter(pl.col("symbol") == symbol)

    async def get_ohlcv(self, symbol, start, end, interval):
        raise NotImplementedError

    async def get_fundamentals(self, symbol):
        raise NotImplementedError

    async def stream_ticks(self, symbols: list[str]):
        raise NotImplementedError
        yield Tick  # pragma: no cover


def test_data_type_and_capabilities():
    assert list(DataType) == [
        DataType.OHLCV,
        DataType.FUNDAMENTALS,
        DataType.OPTIONS_CHAIN,
        DataType.FUNDING_RATES,
        DataType.OPEN_INTEREST,
        DataType.FUTURES_OHLCV,
    ]
    assert DataType.OHLCV in MockSource.CAPABILITIES
    assert DataType.FUNDING_RATES not in MockSource.CAPABILITIES
    with pytest.raises(NotImplementedError):
        import asyncio

        asyncio.run(MockSource().fetch(DataType.OHLCV, "AAPL"))


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


@pytest.mark.asyncio
async def test_data_manager_generic_get_uses_duckdb_first(tmp_path, stock_ohlcv):
    duck = DuckDBStorage()
    cache = ParquetStorage(tmp_path / "cache")
    source = CountingSource(stock_ohlcv)
    manager = DataManager(
        stock_source=source,
        crypto_source=None,
        storage=duck,
        cache=cache,
    )
    first = await manager.get(DataType.OHLCV, ["AAPL"], start=date(2024, 1, 1), end=date(2024, 3, 20))
    second = await manager.get(DataType.OHLCV, ["AAPL"], start=date(2024, 1, 1), end=date(2024, 3, 20))
    funding = await manager.get(DataType.FUNDING_RATES, ["AAPL"], start=date(2024, 1, 1), end=date(2024, 3, 20))
    assert source.calls == 1
    assert duck.read("stock_prices").height == first.height
    assert second.height == first.height
    assert funding.is_empty()


@pytest.mark.asyncio
async def test_data_manager_crypto_futures_routing(tmp_path, crypto_futures_ohlcv):
    duck = DuckDBStorage()
    manager = DataManager(
        stock_source=None,
        crypto_source=None,
        crypto_futures_source=BinanceFuturesDataSource(ohlcv_payloads={"PERP:BTC/USDT": crypto_futures_ohlcv}),
        storage=duck,
        cache=ParquetStorage(tmp_path / "cache"),
    )
    data = await manager.get_futures_ohlcv(["PERP:BTC/USDT"], date(2024, 1, 1), date(2024, 3, 20))
    assert data.height > 0
    assert "futures_prices" in duck.list_keys()
    assert "crypto_prices" not in duck.list_keys()
    assert data.columns == ["date", "symbol", "open", "high", "low", "close", "volume"]
    assert data["symbol"][0] == "PERP:BTC/USDT"


@pytest.mark.asyncio
async def test_data_manager_vn_stock_routing(tmp_path, vn_stock_ohlcv):
    duck = DuckDBStorage()
    manager = DataManager(
        stock_source=None,
        vn_stock_source=DNSEDataSource(ohlcv_payloads={"VN:VNM": vn_stock_ohlcv}),
        crypto_source=None,
        storage=duck,
        cache=ParquetStorage(tmp_path / "cache"),
        bundle_adapter=LocalBundleAdapter(tmp_path / "bundle"),
    )
    data = await manager.get(DataType.OHLCV, ["VN:VNM"], start=date(2024, 1, 1), end=date(2024, 3, 20))
    assert data.height > 0
    assert "vn_stock_prices" in duck.list_keys()
    assert not (tmp_path / "bundle" / "qts-stock-bundle").exists()


@pytest.mark.asyncio
async def test_vnstock_equity_fixture_mode(tmp_path, vn_stock_ohlcv):
    source = VnstockDataSource(ohlcv_payloads={"VN:VNM": vn_stock_ohlcv})
    result = await source.get_ohlcv("VN:VNM", date(2024, 1, 1), date(2024, 3, 20), "1d")
    assert result.columns == ["date", "symbol", "open", "high", "low", "close", "volume"]
    assert result.height > 0
    assert result["symbol"][0] == "VN:VNM"


@pytest.mark.asyncio
async def test_vnstock_equity_unknown_symbol_raises(vn_stock_ohlcv):
    source = VnstockDataSource(ohlcv_payloads={"VN:VNM": vn_stock_ohlcv})
    with pytest.raises(DataSourceError):
        await source.get_ohlcv("VN:UNKNOWN", date(2024, 1, 1), date(2024, 3, 20), "1d")


@pytest.mark.asyncio
async def test_vnstock_equity_fundamentals_fixture(vn_stock_ohlcv):
    fundamentals = pl.DataFrame({"symbol": ["VN:VNM"], "pe": [15.2], "pb": [2.1]})
    source = VnstockDataSource(fundamentals_payloads={"VN:VNM": fundamentals})
    result = await source.get_fundamentals("VN:VNM")
    assert "pe" in result.columns


@pytest.mark.asyncio
async def test_vnstock_futures_fixture_mode(tmp_path, vn_futures_ohlcv):
    source = VnstockFuturesDataSource(ohlcv_payloads={"VNF:VN30F2503": vn_futures_ohlcv})
    result = await source.get_ohlcv("VNF:VN30F2503", date(2024, 1, 1), date(2024, 3, 20), "1d")
    assert result.columns == ["date", "symbol", "open", "high", "low", "close", "volume"]
    assert result.height > 0
    assert result["symbol"][0] == "VNF:VN30F2503"


@pytest.mark.asyncio
async def test_data_manager_vn_futures_routing(tmp_path, vn_futures_ohlcv):
    duck = DuckDBStorage()
    manager = DataManager(
        stock_source=None,
        crypto_source=None,
        vn_futures_source=VnstockFuturesDataSource(
            ohlcv_payloads={"VNF:VN30F2503": vn_futures_ohlcv}
        ),
        storage=duck,
        cache=ParquetStorage(tmp_path / "cache"),
    )
    data = await manager.get_vn_futures_ohlcv(
        ["VNF:VN30F2503"], date(2024, 1, 1), date(2024, 3, 20)
    )
    assert data.height > 0
    assert "vn_futures_prices" in duck.list_keys()
    assert "vn_stock_prices" not in duck.list_keys()
    assert data.columns == ["date", "symbol", "open", "high", "low", "close", "volume"]
    assert data["symbol"][0] == "VNF:VN30F2503"


@pytest.mark.asyncio
async def test_vnstock_registered_in_registry():
    from qts.core.registry import Registry

    assert Registry.get_data_source("vnstock") is VnstockDataSource
    assert Registry.get_data_source("vnstock_futures") is VnstockFuturesDataSource


# --- DNSE tests ---

@pytest.mark.asyncio
async def test_dnse_equity_fixture_mode(vn_stock_ohlcv):
    source = DNSEDataSource(ohlcv_payloads={"VN:VNM": vn_stock_ohlcv})
    result = await source.get_ohlcv("VN:VNM", date(2024, 1, 1), date(2024, 3, 20), "1d")
    assert result.columns == ["date", "symbol", "open", "high", "low", "close", "volume"]
    assert result.height > 0
    assert result["symbol"][0] == "VN:VNM"


@pytest.mark.asyncio
async def test_dnse_futures_fixture_mode(vn_futures_ohlcv):
    source = DNSEDataSource(ohlcv_payloads={"VNF:VN30F2503": vn_futures_ohlcv})
    result = await source.fetch(
        DataType.FUTURES_OHLCV, "VNF:VN30F2503",
        start=date(2024, 1, 1), end=date(2024, 3, 20), interval="1d",
    )
    assert result.height > 0
    assert result["symbol"][0] == "VNF:VN30F1M"


@pytest.mark.asyncio
async def test_dnse_futures_warns_when_history_starts_after_request(vn_futures_ohlcv):
    source = DNSEDataSource(ohlcv_payloads={"VNF:VN30F2503": vn_futures_ohlcv})
    with pytest.warns(
        DataSourceWarning,
        match=r"starts on 2024-01-01, later than requested start 2023-01-01",
    ):
        result = await source.fetch(
            DataType.FUTURES_OHLCV,
            "VNF:VN30F2503",
            start=date(2023, 1, 1),
            end=date(2024, 3, 20),
            interval="1d",
        )
    assert result.height > 0


@pytest.mark.asyncio
async def test_dnse_warrant_fixture_mode(vn_warrant_ohlcv):
    source = DNSEDataSource(ohlcv_payloads={"VNW:CVNM2403": vn_warrant_ohlcv})
    result = await source.get_ohlcv("VNW:CVNM2403", date(2024, 1, 1), date(2024, 3, 20), "1d")
    assert result.height > 0
    assert result["symbol"][0] == "VNW:CVNM2403"


@pytest.mark.asyncio
async def test_dnse_warrant_underlying_fixture_mode(vn_warrant_ohlcv):
    source = DNSEDataSource(ohlcv_payloads={"VNW:CVNM2403": vn_warrant_ohlcv})
    result = await source.get_ohlcv("VNW:VNM", date(2024, 1, 1), date(2024, 3, 20), "1d")
    assert result.height > 0
    assert set(result["symbol"].unique().to_list()) == {"VNW:CVNM2403"}


@pytest.mark.asyncio
async def test_dnse_unknown_symbol_raises(vn_stock_ohlcv):
    source = DNSEDataSource(ohlcv_payloads={"VN:VNM": vn_stock_ohlcv})
    with pytest.raises(DataSourceError):
        await source.get_ohlcv("VN:UNKNOWN", date(2024, 1, 1), date(2024, 3, 20), "1d")


@pytest.mark.asyncio
async def test_data_manager_vn_warrant_routing(tmp_path, vn_warrant_ohlcv):
    duck = DuckDBStorage()
    manager = DataManager(
        stock_source=None,
        crypto_source=None,
        vn_warrant_source=DNSEDataSource(ohlcv_payloads={"VNW:CVNM2403": vn_warrant_ohlcv}),
        storage=duck,
        cache=ParquetStorage(tmp_path / "cache"),
    )
    data = await manager.get(
        DataType.OHLCV, ["VNW:CVNM2403"], start=date(2024, 1, 1), end=date(2024, 3, 20)
    )
    assert data.height > 0
    assert "vn_warrant_prices" in duck.list_keys()
    assert "vn_stock_prices" not in duck.list_keys()
    assert data["symbol"][0] == "VNW:CVNM2403"


@pytest.mark.asyncio
async def test_data_manager_expands_vn_warrant_basket_symbols(tmp_path, vn_warrant_ohlcv):
    duck = DuckDBStorage()
    manager = DataManager(
        stock_source=None,
        crypto_source=None,
        vn_warrant_source=DNSEDataSource(ohlcv_payloads={"VNW:CVNM2403": vn_warrant_ohlcv}),
        storage=duck,
        cache=ParquetStorage(tmp_path / "cache"),
    )
    data = await manager.get(
        DataType.OHLCV, ["VNW:VNM"], start=date(2024, 1, 1), end=date(2024, 3, 20)
    )
    assert data.height > 0
    assert set(data["symbol"].unique().to_list()) == {"VNW:CVNM2403"}
    stored = duck.query("SELECT DISTINCT symbol FROM vn_warrant_prices ORDER BY symbol")
    assert stored["symbol"].to_list() == ["VNW:CVNM2403"]


@pytest.mark.asyncio
async def test_dnse_registered_in_registry():
    from qts.core.registry import Registry

    assert Registry.get_data_source("dnse") is DNSEDataSource
    assert DNSEDataSource.CAPABILITIES == frozenset({DataType.OHLCV, DataType.FUTURES_OHLCV})
