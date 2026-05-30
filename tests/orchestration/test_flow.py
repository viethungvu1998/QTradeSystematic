from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace

import polars as pl

from qts.config.builder import Config
from qts.core.instrument import AssetType
from qts.data._schemas import DataType
from qts.data.sources.binance import BinanceDataSource, BinanceFuturesDataSource
from qts.data.sources.dnse import DNSEDataSource
from qts.data.sources.fmp import FMPDataSource
from qts.orchestration.flow import qts_flow
from qts.orchestration.flows.data_fetch_flow import data_fetch_flow
from qts.orchestration.runtime import resolved_brokers
from qts.orchestration.tasks.data_tasks import (
    download_futures_ohlcv,
    download_ohlcv,
    download_vn_futures_intraday_ohlcv,
)


async def test_qts_flow_research_and_live(
    monkeypatch,
    config_dir,
    stock_ohlcv,
    crypto_ohlcv,
    tmp_path,
):
    original_build = Config.build
    monkeypatch.setenv("QTS_ROOT", str(tmp_path / "qts_root"))

    def build_with_fixtures(path):
        resolved = original_build(path)
        resolved.stock_source = FMPDataSource(ohlcv_payloads={"AAPL": stock_ohlcv})
        resolved.vn_stock_source = DNSEDataSource()
        resolved.crypto_source = BinanceDataSource(ohlcv_payloads={"BTC/USDT": crypto_ohlcv})
        return resolved

    monkeypatch.setattr(Config, "build", staticmethod(build_with_fixtures))
    research_result = await qts_flow(str(config_dir / "research.yaml"))
    live_result = await qts_flow(str(config_dir / "live.yaml"))
    assert research_result.metrics["sharpe"] == research_result.metrics["sharpe"]
    assert "fills" in live_result
    assert isinstance(live_result["orders"], list)


async def test_data_fetch_flow_idempotent(
    monkeypatch,
    config_dir,
    stock_ohlcv,
    crypto_ohlcv,
    tmp_path,
):
    original_build = Config.build
    monkeypatch.setenv("QTS_ROOT", str(tmp_path / "qts_root"))

    def build_with_fixtures(path):
        resolved = original_build(path)
        resolved.stock_source = FMPDataSource(ohlcv_payloads={"AAPL": stock_ohlcv})
        resolved.crypto_source = BinanceDataSource(ohlcv_payloads={"BTC/USDT": crypto_ohlcv})
        return resolved

    monkeypatch.setattr(Config, "build", staticmethod(build_with_fixtures))
    await data_fetch_flow(str(config_dir / "research.yaml"), ["stock"], ["ohlcv"])
    await data_fetch_flow(str(config_dir / "research.yaml"), ["stock"], ["ohlcv"])


async def test_qts_flow_research_includes_crypto_futures_signals(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("QTS_ROOT", str(tmp_path / "qts_root"))
    rows = []
    for index in range(365):
        current = date(2023, 1, 1) + timedelta(days=index)
        price = 43_000.0 + index
        rows.append(
            {
                "date": current,
                "symbol": "PERP:BTC/USDT",
                "open": price,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price + 0.5,
                "volume": 1_000.0 + index,
            }
        )
    crypto_futures_ohlcv = pl.DataFrame(rows)
    research_path = tmp_path / "crypto_futures_research.yaml"
    research_path.write_text(
        """
workflow: research
asset_types: [crypto_futures]
universe:
  crypto_futures: [PERP:BTC/USDT]
start_date: "2023-01-01"
end_date: "2023-12-31"
initial_capital: 100000
data_sources:
  crypto_futures: binance_futures
storage: duckdb
features:
  technical: true
  fundamental: false
  onchain: false
  forward_returns:
    periods: [1]
strategy:
  type: factor
  params: {}
backtest_engine: vectorbt
"""
    )
    original_build = Config.build

    def build_with_fixtures(path):
        resolved = original_build(path)
        resolved.crypto_futures_source = BinanceFuturesDataSource(
            ohlcv_payloads={"PERP:BTC/USDT": crypto_futures_ohlcv}
        )
        return resolved

    monkeypatch.setattr(Config, "build", staticmethod(build_with_fixtures))
    result = await qts_flow(str(research_path))

    assert result.signals.height > 0
    assert "PERP:BTC/USDT" in result.signals["symbol"].to_list()


async def test_qts_flow_research_uses_walk_forward_signals(monkeypatch, config_dir, tmp_path):
    monkeypatch.setenv("QTS_ROOT", str(tmp_path / "qts_root"))
    rows = []
    for index in range(365):
        current = date(2023, 1, 1) + timedelta(days=index)
        price = 100.0 + index
        rows.append(
            {
                "date": current,
                "symbol": "AAPL",
                "open": price,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price + 0.5,
                "volume": 1_000.0 + index,
            }
        )
    stock_ohlcv = pl.DataFrame(rows)

    research_path = config_dir / "stock_research.yaml"
    research_path.write_text(
        """
workflow: research
asset_types: [stock]
universe:
  stock: [AAPL]
start_date: "2023-01-01"
end_date: "2023-12-31"
initial_capital: 100000
data_sources:
  stock: fmp
storage: duckdb
features:
  technical: true
  fundamental: false
  onchain: false
  forward_returns:
    periods: [1]
strategy:
  type: factor
  params: {}
backtest_engine: vectorbt
"""
    )

    original_build = Config.build

    def build_with_fixtures(path):
        resolved = original_build(path)
        resolved.stock_source = FMPDataSource(ohlcv_payloads={"AAPL": stock_ohlcv})
        return resolved

    monkeypatch.setattr(Config, "build", staticmethod(build_with_fixtures))
    result = await qts_flow(str(research_path))

    assert set(result.metrics) == {"sharpe", "sortino", "cagr", "max_drawdown", "win_rate"}
    assert all(signal_date.day == 1 for signal_date in result.signals["date"].to_list())
    assert result.signals["date"].n_unique() == result.signals.height


async def test_download_tasks_include_vn_warrants_and_vn_futures():
    calls = []

    class FakeManager:
        async def get(self, data_type, symbols, **kwargs):
            calls.append((data_type, list(symbols), kwargs))
            return pl.DataFrame()

    config = SimpleNamespace(
        universe=SimpleNamespace(
            stock=[],
            vn_stock=[],
            vn_warrant=["VNM"],
            vn_futures=["VNF:VN30F2503"],
            crypto=[],
            crypto_futures=[],
        ),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 20),
    )

    await download_ohlcv(config, FakeManager())
    await download_futures_ohlcv(config, FakeManager())

    assert calls == [
        (
            DataType.OHLCV,
            ["VNW:VNM"],
            {"start": date(2024, 1, 1), "end": date(2024, 3, 20)},
        ),
        (
            DataType.FUTURES_OHLCV,
            ["VNF:VN30F1M"],
            {"start": date(2024, 1, 1), "end": date(2024, 3, 20)},
        ),
    ]


async def test_download_vn_futures_intraday_ohlcv_fetches_required_intervals():
    calls = []

    class FakeManager:
        async def get(self, data_type, symbols, **kwargs):
            calls.append((data_type, list(symbols), kwargs))
            return pl.DataFrame(
                {
                    "bar_time": [datetime(2024, 1, 2, 9, 0)],
                    "date": [date(2024, 1, 2)],
                    "symbol": [symbols[0]],
                    "interval": [kwargs["interval"]],
                    "open": [1_250.0],
                    "high": [1_251.0],
                    "low": [1_249.0],
                    "close": [1_250.5],
                    "volume": [1_000.0],
                }
            )

    config = SimpleNamespace(
        universe=SimpleNamespace(
            stock=[],
            vn_stock=[],
            vn_warrant=[],
            vn_futures=["VNF:VN30F1M"],
            crypto=[],
            crypto_futures=[],
        ),
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 2),
    )

    result = await download_vn_futures_intraday_ohlcv(config, FakeManager())

    assert result.height == 3
    assert calls == [
        (
            DataType.FUTURES_OHLCV,
            ["VNF:VN30F1M"],
            {"start": date(2024, 1, 2), "end": date(2024, 1, 2), "interval": "1h"},
        ),
        (
            DataType.FUTURES_OHLCV,
            ["VNF:VN30F1M"],
            {"start": date(2024, 1, 2), "end": date(2024, 1, 2), "interval": "15m"},
        ),
        (
            DataType.FUTURES_OHLCV,
            ["VNF:VN30F1M"],
            {"start": date(2024, 1, 2), "end": date(2024, 1, 2), "interval": "30m"},
        ),
    ]


async def test_qts_flow_passes_vn_sources_to_data_manager(monkeypatch):
    captured = {}
    sentinel_warrant = object()
    sentinel_futures = object()

    class FakePipeline:
        def requires_fundamentals(self) -> bool:
            return False

        def with_fundamentals(self, fundamentals):
            return self

    config = SimpleNamespace(
        asset_types=["vn_warrant", "vn_futures"],
        workflow="research",
        features=SimpleNamespace(fundamental=False),
    )
    resolved = SimpleNamespace(
        raw=config,
        stock_source=None,
        vn_stock_source=None,
        vn_warrant_source=sentinel_warrant,
        vn_futures_source=sentinel_futures,
        crypto_source=None,
        crypto_futures_source=None,
        storage=object(),
        cache=object(),
        bundle_adapter=object(),
        feature_pipeline=FakePipeline(),
        strategy=object(),
        engine=object(),
    )
    resolved.with_fundamentals = lambda fundamentals: resolved.feature_pipeline

    class FakeManager:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    async def fake_download_ohlcv(config, manager):
        return pl.DataFrame()

    async def fake_download_futures_ohlcv(config, manager):
        return pl.DataFrame()

    def fake_build_features(config, pipeline, ohlcv):
        return pl.DataFrame()

    def fake_run_backtest(config, engine, strategy, featured, **kwargs):
        return SimpleNamespace(metrics={"sharpe": 0.0}, signals=pl.DataFrame())

    monkeypatch.setattr(Config, "build", staticmethod(lambda path: resolved))
    monkeypatch.setattr("qts.orchestration.runtime.DataManager", FakeManager)
    monkeypatch.setattr("qts.orchestration.flow.download_ohlcv", fake_download_ohlcv)
    monkeypatch.setattr(
        "qts.orchestration.flow.download_futures_ohlcv",
        fake_download_futures_ohlcv,
    )
    monkeypatch.setattr("qts.orchestration.flow.build_features", fake_build_features)
    monkeypatch.setattr("qts.orchestration.flow.run_backtest", fake_run_backtest)

    await qts_flow("unused.yaml")

    assert captured["vn_warrant_source"] is sentinel_warrant
    assert captured["vn_futures_source"] is sentinel_futures


async def test_data_fetch_flow_routes_vn_symbols(monkeypatch):
    captured = {"init": None, "gets": []}
    config = SimpleNamespace(
        universe=SimpleNamespace(
            stock=[],
            vn_stock=[],
            vn_warrant=["VNM"],
            vn_futures=["VNF:VN30F2503"],
            crypto=[],
            crypto_futures=[],
        ),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 20),
    )
    resolved = SimpleNamespace(
        raw=config,
        stock_source=None,
        vn_stock_source=None,
        vn_warrant_source="vn-warrant-source",
        vn_futures_source="vn-futures-source",
        crypto_source=None,
        crypto_futures_source=None,
        storage=object(),
        cache=object(),
    )

    class FakeManager:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        async def get(self, data_type, symbols, **kwargs):
            captured["gets"].append((data_type, list(symbols), kwargs))
            return pl.DataFrame()

    monkeypatch.setattr(Config, "build", staticmethod(lambda path: resolved))
    monkeypatch.setattr("qts.orchestration.runtime.DataManager", FakeManager)

    await data_fetch_flow("unused.yaml", ["vn_warrant", "vn_futures"], ["ohlcv", "futures_ohlcv"])

    assert captured["init"]["vn_warrant_source"] == "vn-warrant-source"
    assert captured["init"]["vn_futures_source"] == "vn-futures-source"
    assert captured["gets"] == [
        (
            DataType.OHLCV,
            ["VNW:VNM", "VNF:VN30F1M"],
            {"start": date(2024, 1, 1), "end": date(2024, 3, 20)},
        ),
        (
            DataType.FUTURES_OHLCV,
            ["VNW:VNM", "VNF:VN30F1M"],
            {"start": date(2024, 1, 1), "end": date(2024, 3, 20)},
        ),
    ]


def test_resolved_brokers_collects_extended_asset_types():
    resolved = SimpleNamespace(
        stock_broker="stock-broker",
        vn_stock_broker=None,
        vn_warrant_broker="vn-warrant-broker",
        vn_futures_broker="vn-futures-broker",
        crypto_broker=None,
        crypto_futures_broker="crypto-futures-broker",
    )
    assert resolved_brokers(resolved) == {
        AssetType.STOCK: "stock-broker",
        AssetType.VN_WARRANT: "vn-warrant-broker",
        AssetType.VN_FUTURES: "vn-futures-broker",
        AssetType.CRYPTO_FUTURES: "crypto-futures-broker",
    }
