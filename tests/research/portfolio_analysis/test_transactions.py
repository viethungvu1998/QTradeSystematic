"""Tests for portfolio_analysis.transactions."""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from qts.research.portfolio_analysis.transactions import (
    compute_avg_holding_period,
    compute_turnover,
)


def _make_signals(n_dates: int, n_symbols: int, signal: int = 1, weight: float = 0.25) -> pl.DataFrame:
    base = date(2023, 1, 2)
    rows = []
    for i in range(n_dates):
        for j in range(n_symbols):
            rows.append({
                "date": base + timedelta(days=i),
                "symbol": f"S{j}",
                "signal": signal,
                "weight": weight,
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def test_compute_turnover_all_zero_signals():
    signals = _make_signals(10, 5, signal=0, weight=0.0)
    result = compute_turnover(signals)
    assert (result["turnover"] == 0.0).all()


def test_compute_turnover_static_positions():
    signals = _make_signals(10, 3, signal=1, weight=0.33)
    result = compute_turnover(signals)
    # Day 1 has turnover (entering from flat); days 2+ are static → turnover = 0
    day1 = result["date"].min()
    after_day1 = result.filter(pl.col("date") > day1)
    assert (after_day1["turnover"] <= 1e-10).all()


def test_compute_turnover_empty():
    empty = pl.DataFrame(schema={"date": pl.Date, "symbol": pl.String, "signal": pl.Int32, "weight": pl.Float64})
    result = compute_turnover(empty)
    assert result.is_empty()
    assert set(result.columns) == {"date", "turnover"}


def test_compute_turnover_schema(rng):
    signals = _make_signals(5, 2)
    result = compute_turnover(signals)
    assert "date" in result.columns
    assert "turnover" in result.columns


def test_compute_avg_holding_period_empty():
    empty = pl.DataFrame(schema={"date": pl.Date, "symbol": pl.String, "signal": pl.Int32, "weight": pl.Float64})
    result = compute_avg_holding_period(empty)
    assert math.isnan(result)


def test_compute_avg_holding_period_zero_turnover():
    # All-zero signals → avg daily turnover == 0 → holding period == inf
    signals = _make_signals(10, 3, signal=0, weight=0.0)
    result = compute_avg_holding_period(signals)
    assert math.isinf(result)


def test_compute_avg_holding_period_changing(rng):
    base = date(2023, 1, 2)
    rows = []
    for i in range(20):
        sig = 1 if i % 2 == 0 else -1
        rows.append({"date": base + timedelta(days=i), "symbol": "A", "signal": sig, "weight": 0.5})
    signals = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))
    result = compute_avg_holding_period(signals)
    assert math.isfinite(result) and result > 0
