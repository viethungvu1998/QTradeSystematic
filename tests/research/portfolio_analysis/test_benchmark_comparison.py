"""Tests for portfolio_analysis.benchmark_comparison."""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from qts.research.portfolio_analysis.benchmark_comparison import compare_to_benchmark


def _returns_df(n: int, value: float, col: str = "portfolio_return", start_offset: int = 0) -> pl.DataFrame:
    base = date(2023, 1, 2)
    dates = [base + timedelta(days=i + start_offset) for i in range(n)]
    return pl.DataFrame({"date": dates, col: [value] * n}).with_columns(pl.col("date").cast(pl.Date))


def test_identical_series_beta_and_alpha(rng):
    vals = rng.normal(0.001, 0.01, 252).tolist()
    base = date(2023, 1, 2)
    dates = [base + timedelta(days=i) for i in range(252)]
    s = pl.DataFrame({"date": dates, "portfolio_return": vals}).with_columns(pl.col("date").cast(pl.Date))
    result = compare_to_benchmark(s, s)
    assert abs(result["beta"] - 1.0) < 1e-6
    assert abs(result["alpha"]) < 1e-3  # near zero annualised
    # tracking_error == 0 when series are identical → IR is nan
    assert result["tracking_error"] == pytest.approx(0.0, abs=1e-9)


def test_positive_active_return():
    strat = _returns_df(100, 0.0011)
    bench = _returns_df(100, 0.0010)
    result = compare_to_benchmark(strat, bench)
    assert result["information_ratio"] > 0


def test_few_overlapping_dates_returns_nan():
    strat = _returns_df(2, 0.001)
    bench = _returns_df(2, 0.001)
    result = compare_to_benchmark(strat, bench)
    assert math.isnan(result["alpha"])
    assert math.isnan(result["beta"])
    assert math.isnan(result["information_ratio"])


def test_non_overlapping_dates_returns_nan():
    strat = _returns_df(50, 0.001, start_offset=0)
    bench = _returns_df(50, 0.001, start_offset=100)
    result = compare_to_benchmark(strat, bench)
    assert all(math.isnan(v) for v in result.values())


def test_result_keys():
    strat = _returns_df(50, 0.001)
    bench = _returns_df(50, 0.001)
    result = compare_to_benchmark(strat, bench)
    assert set(result.keys()) == {"alpha", "beta", "information_ratio", "tracking_error", "correlation"}
