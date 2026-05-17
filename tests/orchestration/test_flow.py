from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from qts.config.builder import Config
from qts.data.sources.binance import BinanceDataSource, BinanceFuturesDataSource
from qts.data.sources.dnse import DNSEDataSource
from qts.data.sources.fmp import FMPDataSource
from qts.orchestration.flows.data_fetch_flow import data_fetch_flow
from qts.orchestration.flow import qts_flow


async def test_qts_flow_research_and_live(monkeypatch, config_dir, stock_ohlcv, crypto_ohlcv, tmp_path):
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


async def test_data_fetch_flow_idempotent(monkeypatch, config_dir, stock_ohlcv, crypto_ohlcv, tmp_path):
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
