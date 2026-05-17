"""Shared fixtures for research module tests."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def history_pd(rng):
    """80-day synthetic price history for 3 symbols A/B/C as pandas DataFrame."""
    symbols = ["A", "B", "C"]
    base = date(2023, 1, 2)
    rows = []
    for sym in symbols:
        price = 100.0
        for i in range(80):
            price *= 1 + rng.normal(0, 0.01)
            rows.append({
                "date": base + timedelta(days=i),
                "symbol": sym,
                "close": price,
                "volume": float(rng.integers(1_000_000, 10_000_000)),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def predictions():
    """pd.Series of scores indexed by symbol."""
    return pd.Series({"A": 0.5, "B": -0.2, "C": 0.1})


@pytest.fixture
def ohlcv_pl(rng):
    """60-day OHLCV Polars frame for 3 symbols."""
    symbols = ["A", "B", "C"]
    base = date(2023, 1, 2)
    rows = []
    for sym in symbols:
        close = 100.0
        for i in range(60):
            close *= 1 + rng.normal(0, 0.01)
            rows.append({
                "date": base + timedelta(days=i),
                "symbol": sym,
                "open": close * 0.999,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": float(rng.integers(1_000_000, 5_000_000)),
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


@pytest.fixture
def backtest_result_fixture(rng):
    """BacktestResult with ~2 years of synthetic daily returns."""
    from qts.research.backtest.base import BacktestResult

    n = 504
    base = date(2022, 1, 3)
    dates = [base + timedelta(days=i) for i in range(n)]
    rets = rng.normal(0.0003, 0.01, n).tolist()
    equity = [100_000.0]
    for r in rets:
        equity.append(equity[-1] * (1 + r))
    returns_df = pl.DataFrame({"date": dates, "portfolio_return": rets}).with_columns(
        pl.col("date").cast(pl.Date)
    )
    equity_df = pl.DataFrame({"date": dates, "equity": equity[1:]}).with_columns(
        pl.col("date").cast(pl.Date)
    )
    signals_df = pl.DataFrame(
        schema={"date": pl.Date, "symbol": pl.String, "signal": pl.Int32, "weight": pl.Float64}
    )
    return BacktestResult(
        engine_name="test",
        metrics={"sharpe": 1.2, "sortino": 1.5, "cagr": 0.12, "max_drawdown": -0.08, "win_rate": 0.54},
        returns=returns_df,
        equity_curve=equity_df,
        signals=signals_df,
    )
