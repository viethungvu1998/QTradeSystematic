from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest


def _ohlcv_frame(symbol: str, start: date, days: int, base_price: float) -> pl.DataFrame:
    rows = []
    for index in range(days):
        current = start + timedelta(days=index)
        price = base_price + index
        rows.append(
            {
                "date": current,
                "symbol": symbol,
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price + 0.5,
                "volume": 1_000 + index * 10,
            }
        )
    return pl.DataFrame(rows)


@pytest.fixture
def stock_ohlcv() -> pl.DataFrame:
    return _ohlcv_frame("AAPL", date(2024, 1, 1), 80, 100)


@pytest.fixture
def crypto_ohlcv() -> pl.DataFrame:
    return _ohlcv_frame("BTC/USDT", date(2024, 1, 1), 80, 30_000)


@pytest.fixture
def paired_ohlcv() -> pl.DataFrame:
    left = _ohlcv_frame("AAA", date(2024, 1, 1), 80, 100)
    right = _ohlcv_frame("BBB", date(2024, 1, 1), 80, 99).with_columns(
        (pl.col("close") + pl.Series([(i % 5) - 2 for i in range(80)])).alias("close")
    )
    return pl.concat([left, right], how="vertical")


@pytest.fixture
def fundamentals_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["AAPL"],
            "pe_ratio": [21.5],
            "ev_ebitda": [15.2],
        }
    )


@pytest.fixture
def onchain_frame() -> pl.DataFrame:
    start = date(2024, 1, 1)
    return pl.DataFrame(
        {
            "date": [start + timedelta(days=index) for index in range(80)],
            "symbol": ["BTC/USDT"] * 80,
            "nvt_ratio": [10 + index * 0.1 for index in range(80)],
            "active_addresses": [1000 + index for index in range(80)],
        }
    )


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    research = """
workflow: research
asset_types: [stock, crypto]
universe:
  stock: [AAPL]
  crypto: [BTC/USDT]
start_date: "2024-01-01"
end_date: "2024-03-20"
initial_capital: 100000
data_sources:
  stock: fmp
  crypto: binance
storage: duckdb
features:
  technical: true
  fundamental: false
  onchain: false
  forward_returns:
    periods: [1, 5]
strategy:
  type: factor
  params: {}
backtest_engine: fast
"""
    validation = """
workflow: validation
asset_types: [stock]
universe:
  stock: [AAPL]
start_date: "2024-01-01"
end_date: "2024-03-20"
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
backtest_engine: normal
fill_model: next_open
slippage_model: volatility_scaled
commission:
  model: percentage
  rate: 0.001
calendar: nyse
promotion_gate:
  max_sharpe_degradation: 0.3
"""
    live = """
workflow: live
asset_types: [stock, crypto]
universe:
  stock: [AAPL]
  crypto: [BTC/USDT]
data_sources:
  stock: fmp
  crypto: binance
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
backtest_engine: fast
fill_model: next_open
slippage_model: volatility_scaled
commission:
  model: percentage
  rate: 0.001
brokers:
  stock: moomoo
  crypto: binance
schedule:
  stock: "0 16 * * 1-5"
  crypto: "0 */4 * * *"
"""
    (tmp_path / "research.yaml").write_text(research)
    (tmp_path / "validation.yaml").write_text(validation)
    (tmp_path / "live.yaml").write_text(live)
    return tmp_path
