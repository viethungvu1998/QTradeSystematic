"""Shared backtest runner helpers."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import polars as pl

from qts.research.backtest.base import BacktestConfig, BacktestResult
from qts.research.backtest.metrics import build_metrics
from qts.research.strategies.base import BaseStrategy


def _rebalance_dates(dates: list[date], frequency: str | int) -> list[date]:
    interval = _parse_session_interval(frequency)
    if interval is not None:
        return dates[::interval]
    normalized = str(frequency).strip().lower()
    if normalized == "daily":
        return dates
    if normalized == "weekly":
        keys = [item.isocalendar()[:2] for item in dates]
    elif normalized == "monthly":
        keys = [(item.year, item.month) for item in dates]
    else:
        raise ValueError(f"Unsupported rebalance frequency: {frequency}")
    selected: list[date] = []
    previous_key: tuple[int, int] | None = None
    for item, key in zip(dates, keys, strict=True):
        if key != previous_key:
            selected.append(item)
            previous_key = key
    return selected


def _parse_session_interval(frequency: str | int) -> int | None:
    if not isinstance(frequency, int) or isinstance(frequency, bool):
        return None
    if frequency < 1:
        raise ValueError(f"Unsupported rebalance frequency: {frequency}")
    return frequency


def walk_forward_signals(
    pipeline,
    strategy,
    ohlcv: pl.DataFrame,
    config: BacktestConfig,
) -> pl.DataFrame:
    all_dates = sorted(ohlcv["date"].unique().to_list())
    rebalance_dates = _rebalance_dates(all_dates, config.rebalance_frequency)
    date_index = {item: index for index, item in enumerate(all_dates)}
    rows: list[pl.DataFrame] = []
    for rebalance_date in rebalance_dates:
        index = date_index[rebalance_date]
        window_start = all_dates[max(0, index - config.train_window)]
        train_slice = ohlcv.filter(
            (pl.col("date") >= window_start) & (pl.col("date") < rebalance_date)
        )
        if train_slice.height < 2:
            continue
        featured = pipeline.fit_transform(train_slice)
        if featured.is_empty():
            continue
        last_date = featured["date"].max()
        signals_for_date = (
            strategy.generate_signals(featured)
            .filter(pl.col("date") == last_date)
            .with_columns(pl.lit(rebalance_date).alias("date"))
        )
        if not signals_for_date.is_empty():
            rows.append(signals_for_date)
    if not rows:
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "symbol": pl.String,
                "signal": pl.Int32,
                "weight": pl.Float64,
            }
        )
    return pl.concat(rows, how="vertical")


def run_backtest_frame(
    engine_name: str,
    strategy: BaseStrategy,
    data: pl.DataFrame,
    config: BacktestConfig,
    prebuilt_signals: pl.DataFrame | None = None,
) -> BacktestResult:
    if prebuilt_signals is None:
        signals = strategy.generate_signals(data).sort(["symbol", "date"])
        signals_for_join = signals
    else:
        signals = prebuilt_signals.sort(["symbol", "date"])
        all_dates_df = data.select("date").unique().sort("date")
        all_symbols_df = data.select("symbol").unique()
        full_grid = all_dates_df.join(all_symbols_df, how="cross")
        signals_for_join = (
            full_grid.join(signals, on=["date", "symbol"], how="left")
            .sort(["symbol", "date"])
            .with_columns(
                pl.col("signal").forward_fill().over("symbol").fill_null(0),
                pl.col("weight").forward_fill().over("symbol").fill_null(0.0),
            )
        )
    joined = (
        data.sort(["symbol", "date"])
        .join(signals_for_join, on=["date", "symbol"], how="left")
        .with_columns(
            pl.col("signal").fill_null(0),
            pl.col("weight").fill_null(0.0),
            (pl.col("close").pct_change().over("symbol")).fill_null(0.0).alias("asset_return"),
        )
        .with_columns(
            (pl.col("asset_return") * pl.col("signal") * pl.col("weight")).alias("strategy_return")
        )
    )
    daily = (
        joined.group_by("date")
        .agg(pl.col("strategy_return").sum().alias("portfolio_return"))
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
    metrics = build_metrics(returns_list, equity_list)
    return BacktestResult(
        engine_name=engine_name,
        metrics=metrics,
        returns=daily.select("date", "portfolio_return"),
        equity_curve=daily.select("date", "equity"),
        signals=signals,
    )
