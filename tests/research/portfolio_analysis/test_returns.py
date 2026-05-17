"""Tests for portfolio_analysis.returns."""

from __future__ import annotations

import pytest

from qts.research.portfolio_analysis.returns import (
    compute_drawdown_series,
    compute_monthly_returns,
    create_returns_tearsheet,
)


def test_compute_monthly_returns_row_count(backtest_result_fixture):
    result = compute_monthly_returns(backtest_result_fixture)
    assert set(result.columns) >= {"year", "month", "monthly_return"}
    # 504 consecutive calendar days starting 2022-01-03 ≈ 16-17 months
    assert result.height >= 14


def test_compute_monthly_returns_columns(backtest_result_fixture):
    result = compute_monthly_returns(backtest_result_fixture)
    assert "year" in result.columns
    assert "month" in result.columns
    assert "monthly_return" in result.columns


def test_compute_drawdown_series_non_positive(backtest_result_fixture):
    result = compute_drawdown_series(backtest_result_fixture)
    assert (result["drawdown"] <= 0.0).all()


def test_compute_drawdown_series_has_negative(backtest_result_fixture):
    result = compute_drawdown_series(backtest_result_fixture)
    assert result["drawdown"].min() < 0.0


def test_compute_drawdown_series_length(backtest_result_fixture):
    result = compute_drawdown_series(backtest_result_fixture)
    assert result.height == backtest_result_fixture.equity_curve.height


def test_create_returns_tearsheet_keys(backtest_result_fixture):
    sheet = create_returns_tearsheet(backtest_result_fixture)
    assert set(sheet.keys()) == {"metrics", "monthly_returns", "drawdown_series", "annual_returns"}


def test_create_returns_tearsheet_annual_rows(backtest_result_fixture):
    sheet = create_returns_tearsheet(backtest_result_fixture)
    # 504 calendar days starting 2022-01-03 → spans at least 2 calendar years
    assert sheet["annual_returns"].height >= 2


def test_create_returns_tearsheet_metrics(backtest_result_fixture):
    sheet = create_returns_tearsheet(backtest_result_fixture)
    assert sheet["metrics"] == backtest_result_fixture.metrics
