"""Returns-based tearsheet from BacktestResult."""

from __future__ import annotations

import math
from datetime import date

import polars as pl

from qts.research.backtest.base import BacktestResult


def compute_monthly_returns(result: BacktestResult) -> pl.DataFrame:
    """Aggregate daily portfolio returns into a (year, month, return) table."""
    daily = result.returns.with_columns(
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
    )
    return (
        daily.group_by(["year", "month"])
        .agg(
            ((1 + pl.col("portfolio_return")).product() - 1).alias("monthly_return")
        )
        .sort(["year", "month"])
    )


def compute_drawdown_series(result: BacktestResult) -> pl.DataFrame:
    """Return a (date, drawdown) frame where drawdown is the % decline from prior peak."""
    equity = result.equity_curve["equity"].to_list()
    dates = result.equity_curve["date"].to_list()
    peak = equity[0] if equity else 0.0
    drawdowns = []
    for val in equity:
        if val > peak:
            peak = val
        drawdowns.append((val / peak - 1) if peak > 0 else 0.0)
    return pl.DataFrame({"date": dates, "drawdown": drawdowns})


def create_returns_tearsheet(result: BacktestResult) -> dict:
    """Summarise BacktestResult into a flat metrics dict plus monthly returns and drawdowns.

    Returns
    -------
    dict with keys:
        metrics        — copy of result.metrics
        monthly_returns — pl.DataFrame (year, month, monthly_return)
        drawdown_series — pl.DataFrame (date, drawdown)
        annual_returns  — pl.DataFrame (year, annual_return)
    """
    monthly = compute_monthly_returns(result)
    drawdown = compute_drawdown_series(result)

    annual = (
        monthly.group_by("year")
        .agg(
            ((1 + pl.col("monthly_return")).product() - 1).alias("annual_return")
        )
        .sort("year")
    )

    return {
        "metrics": result.metrics,
        "monthly_returns": monthly,
        "drawdown_series": drawdown,
        "annual_returns": annual,
    }
