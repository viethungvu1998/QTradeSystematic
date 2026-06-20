from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from qts.core.errors import DataSourceError, DataSourceWarning
from qts.core.events import Tick
from qts.data._schemas import FUTURES_INTRADAY_OHLCV_COLUMNS, OHLCV_COLUMNS, DataType
from qts.data.base import BaseDataSource
from qts.data.bundles.local import LocalBundleAdapter
from qts.data.manager import DataManager
from qts.data.sources.binance import BinanceDataSource, BinanceFuturesDataSource
from qts.data.sources.dnse import DNSEDataSource
from qts.data.sources.fmp import FMPDataSource
from qts.data.sources.vnstock import (
    _KBS_ETF_LIST,
    _FUNDAMENTALS_SCHEMA,
    VnstockDataSource,
    VnstockFuturesDataSource,
    _KBSClient,
)
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
        self.requests: list[dict] = []

    async def fetch(self, data_type: DataType, symbol: str, **kwargs) -> pl.DataFrame:
        self.calls += 1
        self.requests.append(dict(kwargs))
        result = self.frame.filter(pl.col("symbol") == symbol)
        if "date" in result.columns:
            start = kwargs.get("start")
            end = kwargs.get("end")
            if start is not None:
                result = result.filter(pl.col("date") >= start)
            if end is not None:
                result = result.filter(pl.col("date") <= end)
        return result

    async def get_ohlcv(self, symbol, start, end, interval):
        raise NotImplementedError

    async def get_fundamentals(self, symbol):
        raise NotImplementedError

    async def stream_ticks(self, symbols: list[str]):
        raise NotImplementedError
        yield Tick  # pragma: no cover


def _vn_futures_intraday_fixture(symbol: str = "VNF:VN30F1M") -> pl.DataFrame:
    rows = []
    intervals = ("15m", "30m", "1h")
    for interval in intervals:
        for index in range(2):
            bar_time = datetime(2024, 1, 2, 9, 0) + timedelta(minutes=15 * index)
            price = 1_250.0 + index
            rows.append(
                {
                    "bar_time": bar_time,
                    "date": bar_time.date(),
                    "symbol": symbol,
                    "interval": interval,
                    "open": price,
                    "high": price + 1.0,
                    "low": price - 1.0,
                    "close": price + 0.5,
                    "volume": 1_000.0 + index,
                }
            )
    return pl.DataFrame(rows)


def _fundamentals_frame(
    *,
    symbol: str = "VN:VNM",
    report_type: str = "KQKD",
    period: str = "2025",
    fiscal_year: int = 2025,
    quarter: int | None = None,
    report_date: date = date(2026, 2, 27),
    item_en: str = "Revenue",
    value: float = 123.0,
) -> pl.DataFrame:
    return pl.DataFrame(
        [{
            "symbol": symbol,
            "report_type": report_type,
            "period": period,
            "fiscal_year": fiscal_year,
            "quarter": quarter,
            "report_date": report_date,
            "item_en": item_en,
            "value": value,
        }],
        schema=_FUNDAMENTALS_SCHEMA,
    )


class FakeKBSFundamentalsClient:
    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame
        self.calls = 0

    def get_all_financials(
        self,
        symbol: str,
        termtype: int = 1,
        pages: int = 3,
    ) -> pl.DataFrame:
        self.calls += 1
        return self.frame


class FakeVCIClient:
    def __init__(
        self,
        *,
        metadata: dict[str, dict[str, str]],
        statement_rows: dict[str, list[dict]],
        ratio_rows: list[dict],
    ) -> None:
        self.metadata = metadata
        self.statement_rows = statement_rows
        self.ratio_rows = ratio_rows
        self.statement_requests: list[tuple[str, int, int]] = []
        self.ratio_requests: list[tuple[int, int]] = []
        self.metadata_calls = 0

    def get_statement_metadata(self, symbol: str) -> dict[str, dict[str, str]]:
        self.metadata_calls += 1
        return self.metadata

    def get_statement_rows(
        self,
        symbol: str,
        section: str,
        termtype: int,
        limit: int,
    ) -> list[dict]:
        self.statement_requests.append((section, termtype, limit))
        return self.statement_rows.get(section, [])

    def get_ratio_rows(self, symbol: str, termtype: int, limit: int) -> list[dict]:
        self.ratio_requests.append((termtype, limit))
        return self.ratio_rows


def _vci_metadata_fixture() -> dict[str, dict[str, str]]:
    return {
        "BALANCE_SHEET": {"bsa1": "Current assets"},
        "INCOME_STATEMENT": {"isa1": "Sales"},
        "CASH_FLOW": {"cfa1": "Net profit/(loss) before tax"},
    }


def _vci_annual_statement_rows() -> dict[str, list[dict]]:
    return {
        "BALANCE_SHEET": [{
            "yearReport": 2025,
            "lengthReport": 5,
            "publicDate": "2026-02-27T00:00:00",
            "bsa1": 1000,
            "organCode": "VNM",
            "ticker": "VNM",
        }],
        "INCOME_STATEMENT": [{
            "yearReport": 2025,
            "lengthReport": 5,
            "createDate": "2026-01-30T12:00:00",
            "isa1": 2000,
        }],
        "CASH_FLOW": [{
            "yearReport": 2025,
            "lengthReport": 5,
            "updateDate": "2026-02-15T09:30:00",
            "cfa1": 3000,
        }],
    }


def _vci_quarterly_statement_rows() -> dict[str, list[dict]]:
    return {
        "BALANCE_SHEET": [{
            "yearReport": 2025,
            "lengthReport": 4,
            "publicDate": "2026-01-30T00:00:00",
            "bsa1": 1000,
        }],
        "INCOME_STATEMENT": [{
            "yearReport": 2025,
            "lengthReport": 4,
            "publicDate": "2026-01-30T00:00:00",
            "isa1": 2000,
        }],
        "CASH_FLOW": [{
            "yearReport": 2025,
            "lengthReport": 4,
            "publicDate": "2026-01-30T00:00:00",
            "cfa1": 3000,
        }],
    }


def _vci_annual_ratio_rows() -> list[dict]:
    return [{
        "year": "2025",
        "yearReport": 2025,
        "quarter": 5,
        "ratioYearId": 123,
        "ratioTTMId": None,
        "organCode": "VNM",
        "pe": 15.2,
        "pb": 2.1,
    }]


def _vci_quarterly_ratio_rows() -> list[dict]:
    return [
        {
            "year": "2025",
            "yearReport": 2025,
            "quarter": 4,
            "ratioYearId": None,
            "ratioTTMId": None,
            "organCode": "VNM",
            "pe": 15.2,
            "pb": 2.1,
        },
        {
            "year": "2025",
            "yearReport": 2025,
            "quarter": 5,
            "ratioYearId": 123,
            "ratioTTMId": None,
            "organCode": "VNM",
            "pe": 16.2,
        },
    ]


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


def test_data_manager_upsert_bars_replaces_identity_and_sorts():
    duck = DuckDBStorage()
    manager = DataManager(stock_source=None, crypto_source=None, storage=duck)
    table = "vn_futures_intraday_prices"
    identity = ["symbol", "interval", "bar_time"]
    sort_by = ["bar_time"]

    first = _vn_futures_intraday_fixture().filter(pl.col("interval") == "15m").sort(
        "bar_time",
        descending=True,
    )
    second = pl.DataFrame(
        [
            {
                "bar_time": datetime(2024, 1, 2, 9, 15),
                "date": date(2024, 1, 2),
                "symbol": "VNF:VN30F1M",
                "interval": "15m",
                "open": 1_300.0,
                "high": 1_301.0,
                "low": 1_299.0,
                "close": 1_300.5,
                "volume": 2_000.0,
            },
            {
                "bar_time": datetime(2024, 1, 2, 9, 30),
                "date": date(2024, 1, 2),
                "symbol": "VNF:VN30F1M",
                "interval": "15m",
                "open": 1_301.0,
                "high": 1_302.0,
                "low": 1_300.0,
                "close": 1_301.5,
                "volume": 2_100.0,
            },
            {
                "bar_time": datetime(2024, 1, 2, 9, 30),
                "date": date(2024, 1, 2),
                "symbol": "VNF:VN30F1M",
                "interval": "15m",
                "open": 1_302.0,
                "high": 1_303.0,
                "low": 1_301.0,
                "close": 1_302.5,
                "volume": 2_200.0,
            },
        ]
    )

    first_result = manager.upsert_bars(table, first, sort_by=sort_by, identity=identity)
    second_result = manager.upsert_bars(table, second, sort_by=sort_by, identity=identity)

    stored = duck.read(table)
    duplicates = duck.query(
        """
        select symbol, interval, bar_time, count(*) as row_count
        from vn_futures_intraday_prices
        group by 1, 2, 3
        having count(*) > 1
        """
    )

    assert first_result["bar_time"].to_list() == sorted(first_result["bar_time"].to_list())
    assert second_result.height == 3
    assert stored["bar_time"].to_list() == sorted(stored["bar_time"].to_list())
    assert duplicates.is_empty()
    assert stored.filter(pl.col("bar_time") == datetime(2024, 1, 2, 9, 15))[
        "close"
    ].item() == 1_300.5
    assert stored.filter(pl.col("bar_time") == datetime(2024, 1, 2, 9, 30))[
        "close"
    ].item() == 1_302.5


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
async def test_data_manager_generic_get_reuses_cached_or_stored_data(tmp_path, stock_ohlcv):
    duck = DuckDBStorage()
    cache = ParquetStorage(tmp_path / "cache")
    source = CountingSource(stock_ohlcv)
    manager = DataManager(
        stock_source=source,
        crypto_source=None,
        storage=duck,
        cache=cache,
    )
    first = await manager.get(
        DataType.OHLCV,
        ["AAPL"],
        start=date(2024, 1, 1),
        end=date(2024, 3, 20),
    )
    second = await manager.get(
        DataType.OHLCV,
        ["AAPL"],
        start=date(2024, 1, 1),
        end=date(2024, 3, 20),
    )
    funding = await manager.get(
        DataType.FUNDING_RATES,
        ["AAPL"],
        start=date(2024, 1, 1),
        end=date(2024, 3, 20),
    )
    assert source.calls == 1
    assert duck.read("stock_prices").height == first.height
    assert second.height == first.height
    assert funding.is_empty()


@pytest.mark.asyncio
async def test_data_manager_uses_local_cache_before_duckdb(tmp_path, stock_ohlcv):
    duck = DuckDBStorage()
    cache = ParquetStorage(tmp_path / "cache")
    source = CountingSource(stock_ohlcv)
    manager = DataManager(
        stock_source=source,
        crypto_source=None,
        storage=duck,
        cache=cache,
    )
    request = {
        "start": date(2024, 1, 1),
        "end": date(2024, 1, 15),
        "interval": "1d",
    }
    cached_frame = stock_ohlcv.filter(pl.col("date").is_between(request["start"], request["end"]))
    cache.write(manager._cache_key(DataType.OHLCV, "AAPL", **request), cached_frame)

    result = await manager.get(DataType.OHLCV, ["AAPL"], **request)

    assert source.calls == 0
    assert result.height == cached_frame.height
    assert duck.read("stock_prices").height == cached_frame.height


@pytest.mark.asyncio
async def test_data_manager_uses_complete_duckdb_range_without_source(tmp_path, stock_ohlcv):
    duck = DuckDBStorage()
    cache = ParquetStorage(tmp_path / "cache")
    source = CountingSource(stock_ohlcv)
    duck.write(
        "stock_prices",
        stock_ohlcv.filter(pl.col("date").is_between(date(2024, 1, 1), date(2024, 1, 15))),
    )
    manager = DataManager(
        stock_source=source,
        crypto_source=None,
        storage=duck,
        cache=cache,
    )

    result = await manager.get(
        DataType.OHLCV,
        ["AAPL"],
        start=date(2024, 1, 1),
        end=date(2024, 1, 15),
        interval="1d",
    )

    assert source.calls == 0
    assert result.height == 15


@pytest.mark.asyncio
async def test_data_manager_fetches_only_trailing_boundary_gap(tmp_path, stock_ohlcv):
    duck = DuckDBStorage()
    source = CountingSource(stock_ohlcv)
    duck.write(
        "stock_prices",
        stock_ohlcv.filter(pl.col("date").is_between(date(2024, 1, 1), date(2024, 1, 10))),
    )
    manager = DataManager(
        stock_source=source,
        crypto_source=None,
        storage=duck,
        cache=ParquetStorage(tmp_path / "cache"),
    )

    result = await manager.get(
        DataType.OHLCV,
        ["AAPL"],
        start=date(2024, 1, 1),
        end=date(2024, 1, 15),
        interval="1d",
    )

    assert result.height == 15
    assert source.requests == [
        {"start": date(2024, 1, 11), "end": date(2024, 1, 15), "interval": "1d"}
    ]


@pytest.mark.asyncio
async def test_data_manager_fetches_only_leading_boundary_gap(tmp_path, stock_ohlcv):
    duck = DuckDBStorage()
    source = CountingSource(stock_ohlcv)
    duck.write(
        "stock_prices",
        stock_ohlcv.filter(pl.col("date").is_between(date(2024, 1, 6), date(2024, 1, 15))),
    )
    manager = DataManager(
        stock_source=source,
        crypto_source=None,
        storage=duck,
        cache=ParquetStorage(tmp_path / "cache"),
    )

    result = await manager.get(
        DataType.OHLCV,
        ["AAPL"],
        start=date(2024, 1, 1),
        end=date(2024, 1, 15),
        interval="1d",
    )

    assert result.height == 15
    assert source.requests == [
        {"start": date(2024, 1, 1), "end": date(2024, 1, 5), "interval": "1d"}
    ]


@pytest.mark.asyncio
async def test_data_manager_fetches_full_range_when_duckdb_empty(tmp_path, stock_ohlcv):
    duck = DuckDBStorage()
    source = CountingSource(stock_ohlcv)
    manager = DataManager(
        stock_source=source,
        crypto_source=None,
        storage=duck,
        cache=ParquetStorage(tmp_path / "cache"),
    )

    result = await manager.get(
        DataType.OHLCV,
        ["AAPL"],
        start=date(2024, 1, 1),
        end=date(2024, 1, 15),
        interval="1d",
    )

    assert result.height == 15
    assert source.requests == [
        {"start": date(2024, 1, 1), "end": date(2024, 1, 15), "interval": "1d"}
    ]


@pytest.mark.asyncio
async def test_data_manager_crypto_futures_routing(tmp_path, crypto_futures_ohlcv):
    duck = DuckDBStorage()
    manager = DataManager(
        stock_source=None,
        crypto_source=None,
        crypto_futures_source=BinanceFuturesDataSource(
            ohlcv_payloads={"PERP:BTC/USDT": crypto_futures_ohlcv}
        ),
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
    data = await manager.get(
        DataType.OHLCV,
        ["VN:VNM"],
        start=date(2024, 1, 1),
        end=date(2024, 3, 20),
    )
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


def test_vnstock_rejects_unknown_fundamentals_source():
    with pytest.raises(ValueError):
        VnstockDataSource(fundamentals_source="unknown")


@pytest.mark.asyncio
async def test_vnstock_kbs_fundamentals_dispatch_and_schema_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("qts.data.sources.vnstock.cache_dir", lambda: tmp_path)
    frame = _fundamentals_frame()
    client = FakeKBSFundamentalsClient(frame)
    source = VnstockDataSource(client=client)

    result = await source.get_fundamentals("VN:VNM", force_refresh=True)

    assert client.calls == 1
    assert result.equals(frame)


@pytest.mark.asyncio
async def test_vnstock_vci_annual_fundamentals_match_current_schema(tmp_path, monkeypatch):
    monkeypatch.setattr("qts.data.sources.vnstock.cache_dir", lambda: tmp_path)
    source = VnstockDataSource(
        fundamentals_source="vci",
        vci_client=FakeVCIClient(
            metadata=_vci_metadata_fixture(),
            statement_rows=_vci_annual_statement_rows(),
            ratio_rows=_vci_annual_ratio_rows(),
        ),
    )

    result = await source.get_fundamentals(
        "VN:VNM",
        termtype=1,
        pages=1,
        force_refresh=True,
    )

    assert result.columns == list(_FUNDAMENTALS_SCHEMA.keys())
    assert result.dtypes == pl.DataFrame(schema=_FUNDAMENTALS_SCHEMA).dtypes
    assert set(result["report_type"].unique().to_list()) == {"CDKT", "CSTC", "KQKD", "LCTT"}
    assert result["quarter"].null_count() == result.height
    assert result["period"].unique().to_list() == ["2025"]
    assert result["report_date"].null_count() == 0
    assert set(result["item_en"].unique().to_list()) == {
        "Current assets",
        "Net profit/(loss) before tax",
        "Sales",
        "pb",
        "pe",
    }


@pytest.mark.asyncio
async def test_vnstock_vci_quarterly_fundamentals_match_current_schema(tmp_path, monkeypatch):
    monkeypatch.setattr("qts.data.sources.vnstock.cache_dir", lambda: tmp_path)
    source = VnstockDataSource(
        fundamentals_source="vci",
        vci_client=FakeVCIClient(
            metadata=_vci_metadata_fixture(),
            statement_rows=_vci_quarterly_statement_rows(),
            ratio_rows=_vci_quarterly_ratio_rows(),
        ),
    )

    result = await source.get_fundamentals(
        "VN:VNM",
        termtype=2,
        pages=1,
        force_refresh=True,
    )

    assert result.columns == list(_FUNDAMENTALS_SCHEMA.keys())
    assert result.dtypes == pl.DataFrame(schema=_FUNDAMENTALS_SCHEMA).dtypes
    assert set(result["report_type"].unique().to_list()) == {"CDKT", "CSTC", "KQKD", "LCTT"}
    assert result["quarter"].null_count() == 0
    assert result["quarter"].unique().to_list() == [4]
    assert result["period"].unique().to_list() == ["2025-Q4"]
    assert result["report_date"].null_count() == 0
    assert set(result["item_en"].unique().to_list()) == {
        "Current assets",
        "Net profit/(loss) before tax",
        "Sales",
        "pb",
        "pe",
    }


@pytest.mark.asyncio
async def test_vnstock_shared_fundamentals_cache_is_reused_across_sources(tmp_path, monkeypatch):
    monkeypatch.setattr("qts.data.sources.vnstock.cache_dir", lambda: tmp_path)
    cached = _fundamentals_frame(value=111.0)
    kbs_client = FakeKBSFundamentalsClient(cached)
    vci_client = FakeVCIClient(
        metadata=_vci_metadata_fixture(),
        statement_rows=_vci_annual_statement_rows(),
        ratio_rows=_vci_annual_ratio_rows(),
    )

    kbs_source = VnstockDataSource(client=kbs_client)
    await kbs_source.get_fundamentals("VN:VNM", force_refresh=True)

    vci_source = VnstockDataSource(fundamentals_source="vci", vci_client=vci_client)
    result = await vci_source.get_fundamentals("VN:VNM", force_refresh=False)

    assert result.equals(cached)
    assert vci_client.metadata_calls == 0
    assert vci_client.statement_requests == []
    assert vci_client.ratio_requests == []


@pytest.mark.asyncio
async def test_vnstock_force_refresh_overwrites_shared_cache_with_selected_provider(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr("qts.data.sources.vnstock.cache_dir", lambda: tmp_path)
    cached = _fundamentals_frame(value=111.0)
    refreshed_client = FakeVCIClient(
        metadata=_vci_metadata_fixture(),
        statement_rows=_vci_annual_statement_rows(),
        ratio_rows=_vci_annual_ratio_rows(),
    )

    kbs_source = VnstockDataSource(client=FakeKBSFundamentalsClient(cached))
    await kbs_source.get_fundamentals("VN:VNM", force_refresh=True)

    vci_source = VnstockDataSource(fundamentals_source="vci", vci_client=refreshed_client)
    refreshed = await vci_source.get_fundamentals("VN:VNM", termtype=1, force_refresh=True)

    cached_again = await kbs_source.get_fundamentals("VN:VNM", force_refresh=False)

    assert refreshed_client.metadata_calls == 1
    assert refreshed_client.statement_requests
    assert refreshed_client.ratio_requests == [(1, 12)]
    assert cached_again.equals(refreshed)


def test_kbs_client_get_etfs_normalizes_listing_payload():
    class FakeResponse:
        def __init__(self, payload: object) -> None:
            self.payload = payload

        def raise_for_status(self) -> None:
            pass

        def json(self) -> object:
            return self.payload

    class FakeHttp:
        def __init__(self, payload: object) -> None:
            self.payload = payload
            self.urls: list[str] = []

        def get(self, url: str):
            self.urls.append(url)
            return FakeResponse(self.payload)

    http = FakeHttp(
        {
            "status": 200,
            "data": ["e1vfvn30", "VN:FUEVFVND", {"symbol": "fuessv30"}, "E1VFVN30"],
        }
    )
    client = object.__new__(_KBSClient)
    client._http = http

    symbols = client.get_etfs()

    assert symbols == ["VN:E1VFVN30", "VN:FUESSV30", "VN:FUEVFVND"]
    assert http.urls == [_KBS_ETF_LIST]


@pytest.mark.asyncio
async def test_vnstock_futures_fixture_mode(tmp_path, vn_futures_ohlcv):
    source = VnstockFuturesDataSource(ohlcv_payloads={"VNF:VN30F2503": vn_futures_ohlcv})
    result = await source.get_ohlcv("VNF:VN30F2503", date(2024, 1, 1), date(2024, 3, 20), "1d")
    assert result.columns == ["date", "symbol", "open", "high", "low", "close", "volume"]
    assert result.height > 0
    assert result["symbol"][0] == "VNF:VN30F2503"


@pytest.mark.asyncio
async def test_vnstock_futures_rolls_vn30f1m_to_kbs_front_month_code():
    class FakeKBSClient:
        def __init__(self) -> None:
            self.requests: list[dict] = []

        def get_ohlcv(self, **kwargs):
            self.requests.append(kwargs)
            return [
                {
                    "t": "2026-05-29T09:00:00",
                    "o": 2000,
                    "h": 2001,
                    "l": 1999,
                    "c": 2000.5,
                    "v": 1000,
                }
            ]

    client = FakeKBSClient()
    source = VnstockFuturesDataSource(client=client)

    result = await source.get_ohlcv(
        "VNF:VN30F1M",
        date(2026, 5, 29),
        date(2026, 5, 29),
        "15m",
    )

    assert client.requests[0]["symbol"] == "41I1G6000"
    assert result.columns == FUTURES_INTRADAY_OHLCV_COLUMNS
    assert result["symbol"].to_list() == ["VNF:VN30F1M"]


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
async def test_vnstock_futures_intraday_fixture_preserves_bar_identity():
    source = VnstockFuturesDataSource(
        ohlcv_payloads={"VNF:VN30F1M": _vn_futures_intraday_fixture()}
    )

    result = await source.fetch(
        DataType.FUTURES_OHLCV,
        "VNF:VN30F1M",
        start=date(2024, 1, 2),
        end=date(2024, 1, 2),
        interval="15m",
    )

    assert result.columns == FUTURES_INTRADAY_OHLCV_COLUMNS
    assert result.height == 2
    assert result["symbol"].unique().to_list() == ["VNF:VN30F1M"]
    assert result["interval"].unique().to_list() == ["15m"]
    assert result["bar_time"].n_unique() == 2


@pytest.mark.asyncio
async def test_data_manager_vn_futures_intraday_stores_multiple_intervals_without_collision(
    tmp_path,
):
    duck = DuckDBStorage()
    manager = DataManager(
        stock_source=None,
        crypto_source=None,
        vn_futures_source=VnstockFuturesDataSource(
            ohlcv_payloads={"VNF:VN30F1M": _vn_futures_intraday_fixture()}
        ),
        storage=duck,
        cache=ParquetStorage(tmp_path / "cache"),
    )

    for interval in ("15m", "30m", "1h"):
        data = await manager.get_vn_futures_ohlcv(
            ["VNF:VN30F1M"],
            date(2024, 1, 2),
            date(2024, 1, 2),
            interval=interval,
        )
        assert data.columns == FUTURES_INTRADAY_OHLCV_COLUMNS
        assert data["interval"].unique().to_list() == [interval]

    stored = duck.read("vn_futures_intraday_prices")
    assert stored.height == 6
    assert set(stored["interval"].unique().to_list()) == {"15m", "30m", "1h"}
    assert "vn_futures_prices" not in duck.list_keys()


@pytest.mark.asyncio
async def test_data_manager_vn_futures_intraday_deduplicates_repeated_runs(tmp_path):
    duck = DuckDBStorage()
    manager = DataManager(
        stock_source=None,
        crypto_source=None,
        vn_futures_source=VnstockFuturesDataSource(
            ohlcv_payloads={"VNF:VN30F1M": _vn_futures_intraday_fixture()}
        ),
        storage=duck,
        cache=ParquetStorage(tmp_path / "cache"),
    )

    request = {
        "start": date(2024, 1, 2),
        "end": date(2024, 1, 2),
        "interval": "15m",
    }
    first = await manager.get_vn_futures_ohlcv(["VNF:VN30F1M"], **request)
    second = await manager.get_vn_futures_ohlcv(["VNF:VN30F1M"], **request)

    assert first.height == 2
    assert second.height == 2
    assert duck.read("vn_futures_intraday_prices").height == 2


@pytest.mark.asyncio
async def test_data_manager_vn_futures_daily_keeps_legacy_table_and_schema(
    tmp_path,
    vn_futures_ohlcv,
):
    duck = DuckDBStorage()
    manager = DataManager(
        stock_source=None,
        crypto_source=None,
        vn_futures_source=VnstockFuturesDataSource(
            ohlcv_payloads={"VNF:VN30F1M": vn_futures_ohlcv}
        ),
        storage=duck,
        cache=ParquetStorage(tmp_path / "cache"),
    )

    data = await manager.get_vn_futures_ohlcv(
        ["VNF:VN30F1M"],
        date(2024, 1, 1),
        date(2024, 3, 20),
    )

    assert data.columns == OHLCV_COLUMNS
    assert "vn_futures_prices" in duck.list_keys()
    assert "vn_futures_intraday_prices" not in duck.list_keys()


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
