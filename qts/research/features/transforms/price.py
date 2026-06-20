"""Price-derived feature transforms."""

from __future__ import annotations

import polars as pl

from qts.core.registry import Registry


@Registry.register_transform("average_price")
def average_price_transform(
    df: pl.DataFrame,
    *,
    high_column: str = "high",
    low_column: str = "low",
    close_column: str = "close",
    output_column: str = "avg_price",
) -> pl.DataFrame:
    """Append the average price ``(high + low + close) / 3``."""
    return df.with_columns(
        (
            (pl.col(high_column) + pl.col(low_column) + pl.col(close_column))
            / 3.0
        ).alias(output_column)
    )


@Registry.register_transform("log_price")
def log_price_transform(
    df: pl.DataFrame,
    *,
    price_column: str = "avg_price",
    output_column: str = "log_price",
) -> pl.DataFrame:
    """Append the natural log of a positive price column."""
    return df.with_columns(
        pl.when(pl.col(price_column) > 0)
        .then(pl.col(price_column).log())
        .otherwise(None)
        .alias(output_column)
    )
