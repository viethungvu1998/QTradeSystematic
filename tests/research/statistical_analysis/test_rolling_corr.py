"""Tests for statistical_analysis.rolling_corr."""

from __future__ import annotations

import polars as pl
import pytest

from qts.research.statistical_analysis.rolling_corr import (
    average_pairwise_correlation,
    rolling_correlation_matrix,
)


def test_rolling_correlation_matrix_entry_count(ohlcv_pl):
    result = rolling_correlation_matrix(ohlcv_pl, window=20)
    # log_ret drops 1st date per symbol → 59 dates; 59 - 20 + 1 = 40 windows
    assert len(result) == 40


def test_rolling_correlation_matrix_structure(ohlcv_pl):
    result = rolling_correlation_matrix(ohlcv_pl, window=20)
    for date_str, corr_df in result.items():
        assert "symbol" in corr_df.columns
        symbol_cols = [c for c in corr_df.columns if c != "symbol"]
        assert len(symbol_cols) >= 1


def test_rolling_correlation_matrix_diagonal_ones(ohlcv_pl):
    result = rolling_correlation_matrix(ohlcv_pl, window=20)
    for date_str, corr_df in list(result.items())[:5]:
        symbols = [c for c in corr_df.columns if c != "symbol"]
        for sym in symbols:
            row = corr_df.filter(pl.col("symbol") == sym)
            if row.height > 0:
                diag_val = row[sym][0]
                assert abs(diag_val - 1.0) < 1e-9, f"Diagonal != 1 for {sym} on {date_str}"


def test_rolling_correlation_matrix_too_few_rows(ohlcv_pl):
    short = ohlcv_pl.head(5 * 3)  # 5 dates × 3 symbols
    result = rolling_correlation_matrix(short, window=20)
    assert result == {}


def test_rolling_correlation_matrix_empty():
    empty = pl.DataFrame(schema={"date": pl.Date, "symbol": pl.String, "close": pl.Float64})
    result = rolling_correlation_matrix(empty, window=10)
    assert result == {}


def test_average_pairwise_correlation_schema(ohlcv_pl):
    result = average_pairwise_correlation(ohlcv_pl, window=20)
    assert "date" in result.columns
    assert "avg_correlation" in result.columns


def test_average_pairwise_correlation_values_bounded(ohlcv_pl):
    result = average_pairwise_correlation(ohlcv_pl, window=20)
    vals = result["avg_correlation"].drop_nulls()
    assert (vals >= 0.0).all() and (vals <= 1.0).all()


def test_average_pairwise_correlation_length_matches(ohlcv_pl):
    matrices = rolling_correlation_matrix(ohlcv_pl, window=20)
    avg = average_pairwise_correlation(ohlcv_pl, window=20)
    assert avg.height == len(matrices)  # one row per valid window
