"""Turnover and holding period analysis from signal frames."""

from __future__ import annotations

import polars as pl


def compute_turnover(signals: pl.DataFrame) -> pl.DataFrame:
    """Compute daily one-way turnover from a SignalFrame.

    Turnover = 0.5 * sum(|w_t - w_{t-1}|) across symbols, where w is the signed weight
    (signal * weight).

    Returns
    -------
    pl.DataFrame with columns [date, turnover].
    """
    if signals.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "turnover": pl.Float64})

    full_grid = (
        signals.select("date").unique()
        .join(signals.select("symbol").unique(), how="cross")
    )
    filled = (
        full_grid
        .join(signals, on=["date", "symbol"], how="left")
        .sort(["symbol", "date"])
        .with_columns(
            pl.col("signal").fill_null(0),
            pl.col("weight").fill_null(0.0),
        )
        .with_columns(
            (pl.col("signal").cast(pl.Float64) * pl.col("weight")).alias("pos_weight")
        )
        .with_columns(
            pl.col("pos_weight").shift(1).over("symbol").alias("prev_weight")
        )
        .with_columns(
            (pl.col("pos_weight") - pl.col("prev_weight").fill_null(0.0)).abs().alias("delta")
        )
    )
    return (
        filled.group_by("date")
        .agg((0.5 * pl.col("delta").sum()).alias("turnover"))
        .sort("date")
    )


def compute_avg_holding_period(signals: pl.DataFrame) -> float:
    """Estimate average holding period in trading days from signal positions.

    Uses the heuristic: avg_holding = 1 / (one_way_turnover_per_day) when turnover > 0.
    """
    turnover_df = compute_turnover(signals)
    if turnover_df.is_empty():
        return float("nan")
    avg_daily = turnover_df["turnover"].mean()
    if avg_daily is None or avg_daily <= 0:
        return float("inf")
    return 1.0 / float(avg_daily)
