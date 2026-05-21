"""Universe screener transform — filters symbols by volume and momentum."""

from __future__ import annotations

import polars as pl

from qts.core.registry import Registry


@Registry.register_transform("universe_screener")
def universe_screener(
    df: pl.DataFrame,
    *,
    lookback_days: int = 730,
    volume_top_n: int = 1000,
    momentum_top_n: int = 75,
    min_avg_volume: float = 100_000,
    min_avg_price: float = 4.0,
    min_last_price: float = 5.0,
    volatility_filter: bool = True,
    max_volatility: float = 0.25,
    symbol_column: str = "symbol",
    date_column: str = "date",
    close_column: str = "close",
    volume_column: str = "volume",
) -> pl.DataFrame:
    """Two-stage funnel: top N symbols by dollar volume, then top N by momentum."""
    avg_stats = (
        df.sort([symbol_column, date_column])
        .group_by(symbol_column)
        .agg(
            [
                pl.col(volume_column).mean().alias("avg_volume"),
                pl.col(close_column).mean().alias("avg_price"),
                pl.col(close_column).last().alias("last_price"),
                pl.col(close_column).std().alias("std_price"),
            ]
        )
        .with_columns(
            (pl.col("avg_volume") * pl.col("avg_price")).alias("dollar_volume")
        )
    )

    avg_stats = avg_stats.filter(
        (pl.col("avg_volume") >= min_avg_volume)
        & (pl.col("avg_price") >= min_avg_price)
        & (pl.col("last_price") >= min_last_price)
    )

    if volatility_filter:
        avg_stats = avg_stats.with_columns(
            (pl.col("std_price") / pl.col("avg_price")).alias("cv")
        ).filter(pl.col("cv") <= max_volatility)

    top_volume = avg_stats.sort("dollar_volume", descending=True).head(volume_top_n)

    momentum = (
        df.filter(pl.col(symbol_column).is_in(top_volume[symbol_column]))
        .sort([symbol_column, date_column])
        .group_by(symbol_column)
        .agg(
            [
                pl.col(close_column).first().alias("first_price"),
                pl.col(close_column).last().alias("last_price2"),
            ]
        )
        .with_columns(
            (pl.col("last_price2") / pl.col("first_price") - 1).alias("momentum")
        )
        .sort("momentum", descending=True)
        .head(momentum_top_n)
    )

    surviving_symbols = momentum[symbol_column]
    return df.filter(pl.col(symbol_column).is_in(surviving_symbols))
