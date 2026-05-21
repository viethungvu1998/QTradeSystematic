from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import polars as pl

from qts.research.backtest._runner import run_backtest_frame
from qts.research.backtest.base import BacktestConfig, UniverseConfig


class FailingStrategy:
    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        raise AssertionError("prebuilt signals should be used")


def test_run_backtest_frame_emits_observability_on_signal_transition():
    data = _ohlcv()
    signals = pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 3)],
            "symbol": ["AAPL", "AAPL"],
            "signal": [1, 0],
            "weight": [1.0, 0.0],
        }
    )

    result = run_backtest_frame(
        "zipline",
        FailingStrategy(),
        data,
        _config(),
        prebuilt_signals=signals,
    )

    assert result.trade_log.height == 1
    trade = result.trade_log.row(0, named=True)
    assert trade["ticker"] == "AAPL"
    assert trade["start_price"] == 100.0
    assert trade["end_price"] == 120.0
    assert trade["profit_pct"] == 0.2
    assert result.portfolio_snapshots.height == 4
    assert result.portfolio_snapshots["tokens"][0][0]["token"] == "AAPL"


def _ohlcv() -> pl.DataFrame:
    rows = []
    for index, close in enumerate([100.0, 110.0, 120.0, 130.0]):
        rows.append(
            {
                "date": date(2024, 1, 1) + timedelta(days=index),
                "symbol": "AAPL",
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1_000.0,
            }
        )
    return pl.DataFrame(rows)


def _config() -> BacktestConfig:
    return BacktestConfig(
        workflow="research",
        asset_types=["stock"],
        universe=UniverseConfig(stock=["AAPL"]),
        initial_capital=Decimal("100000"),
    )
