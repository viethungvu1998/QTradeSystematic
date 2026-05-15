"""Shared backtest runner helpers."""

from __future__ import annotations

from decimal import Decimal

import polars as pl

from qts.research.backtest.base import BacktestConfig, BacktestResult
from qts.research.backtest.metrics import cagr, max_drawdown, sharpe_ratio, sortino_ratio, win_rate
from qts.research.strategies.base import BaseStrategy


def run_backtest_frame(
    engine_name: str,
    strategy,
    data: pl.DataFrame,
    config: BacktestConfig,
) -> BacktestResult:
    signals = strategy.generate_signals(data).sort(["symbol", "date"])
    joined = (
        data.sort(["symbol", "date"])
        .join(signals, on=["date", "symbol"], how="left")
        .with_columns(
            pl.col("signal").fill_null(0),
            pl.col("weight").fill_null(0.0),
            (pl.col("close").pct_change().over("symbol")).fill_null(0.0).alias("asset_return"),
        )
        .with_columns((pl.col("asset_return") * pl.col("signal") * pl.col("weight")).alias("strategy_return"))
    )
    daily = (
        joined.group_by("date")
        .agg(pl.col("strategy_return").mean().alias("portfolio_return"))
        .sort("date")
    )
    capital = float(config.initial_capital or Decimal("100000"))
    equity_values = []
    running = capital
    for value in daily["portfolio_return"].to_list():
        running *= 1 + float(value)
        equity_values.append(running)
    daily = daily.with_columns(pl.Series("equity", equity_values))
    returns_list = [float(value) for value in daily["portfolio_return"].to_list()]
    equity_list = [capital, *equity_values]
    metrics = {
        "sharpe": sharpe_ratio(returns_list),
        "sortino": sortino_ratio(returns_list),
        "cagr": cagr(equity_list),
        "max_drawdown": max_drawdown(equity_list),
        "win_rate": win_rate(returns_list),
    }
    return BacktestResult(
        engine_name=engine_name,
        metrics=metrics,
        returns=daily.select("date", "portfolio_return"),
        equity_curve=daily.select("date", "equity"),
        signals=signals,
    )
