"""Tests for portfolio_analysis.factor."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

scipy = pytest.importorskip("scipy")

from qts.research.portfolio_analysis.factor import compute_factor_quantile_returns, compute_ic


@pytest.fixture
def factor_df(rng):
    """5 symbols × 20 dates panel with factor and forward return columns."""
    symbols = ["A", "B", "C", "D", "E"]
    base = date(2023, 1, 2)
    rows = []
    for sym in symbols:
        for i in range(20):
            fwd = float(rng.normal(0, 0.01)) if i < 18 else None
            rows.append({
                "date": base + timedelta(days=i),
                "symbol": sym,
                "f1": float(rng.normal(0, 1)),
                "f2": float(rng.normal(0, 1)),
                "f3": float(rng.normal(0, 1)),
                "fwd_ret": fwd,
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def test_compute_ic_spearman_columns(factor_df):
    result = compute_ic(factor_df, ["f1", "f2", "f3"], "fwd_ret", method="spearman")
    assert "date" in result.columns
    for col in ["f1", "f2", "f3"]:
        assert col in result.columns


def test_compute_ic_spearman_values_bounded(factor_df):
    result = compute_ic(factor_df, ["f1", "f2", "f3"], "fwd_ret", method="spearman")
    for col in ["f1", "f2", "f3"]:
        vals = result[col].drop_nulls()
        assert (vals >= -1.0).all() and (vals <= 1.0).all()


def test_compute_ic_pearson_columns(factor_df):
    result = compute_ic(factor_df, ["f1", "f2", "f3"], "fwd_ret", method="pearson")
    assert "date" in result.columns


def test_compute_ic_skips_sparse_dates(factor_df):
    # Dates with < 3 non-null observations should be skipped
    result = compute_ic(factor_df, ["f1"], "fwd_ret", method="spearman")
    # The last 2 dates have null fwd_ret for all 5 symbols → skipped
    assert result.height <= 18


def test_compute_factor_quantile_returns_quantile_values(factor_df):
    result = compute_factor_quantile_returns(factor_df, "f1", "fwd_ret", n_quantiles=5)
    assert "quantile" in result.columns
    assert "mean_return" in result.columns
    unique_q = set(result["quantile"].unique().to_list())
    # qcut with string labels cast to Int32 produces 0-based ordinals
    assert unique_q.issubset({0, 1, 2, 3, 4})


def test_compute_factor_quantile_returns_no_out_of_range(factor_df):
    result = compute_factor_quantile_returns(factor_df, "f1", "fwd_ret", n_quantiles=5)
    assert result.filter(~pl.col("quantile").is_in([0, 1, 2, 3, 4])).is_empty()


def test_compute_ic_empty_df():
    empty = pl.DataFrame(schema={"date": pl.Date, "symbol": pl.String,
                                  "f1": pl.Float64, "fwd_ret": pl.Float64})
    result = compute_ic(empty, ["f1"], "fwd_ret")
    assert result.is_empty()
    assert "date" in result.columns
    assert "f1" in result.columns
