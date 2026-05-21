"""QS-Momentum cross-sectional score transform."""

from __future__ import annotations

import polars as pl

from qts.core.registry import Registry


@Registry.register_transform("qsmom")
def qsmom_transform(
    df: pl.DataFrame,
    *,
    roc_fast_period: int = 21,
    roc_slow_period: int = 252,
    returns_period: int = 126,
    symbol_column: str = "symbol",
    date_column: str = "date",
    close_column: str = "close",
) -> pl.DataFrame:
    """Compute the QS-Momentum composite score."""
    col_name = (
        f"{close_column}_qsmom_{roc_fast_period}_{roc_slow_period}_{returns_period}"
    )

    return (
        df.sort([symbol_column, date_column])
        .with_columns(
            [
                (
                    pl.col(close_column)
                    / pl.col(close_column).shift(roc_fast_period).over(symbol_column)
                    - 1
                ).alias("_qsmom_roc_fast"),
                (
                    pl.col(close_column)
                    / pl.col(close_column).shift(roc_slow_period).over(symbol_column)
                    - 1
                ).alias("_qsmom_roc_slow"),
                (
                    pl.col(close_column)
                    / pl.col(close_column).shift(returns_period).over(symbol_column)
                    - 1
                ).alias("_qsmom_returns"),
            ]
        )
        .with_columns(
            pl.when(pl.col("_qsmom_roc_fast") > 0)
            .then(pl.col("_qsmom_roc_slow") + pl.col("_qsmom_returns"))
            .otherwise(pl.lit(None))
            .alias(col_name)
        )
        .drop(["_qsmom_roc_fast", "_qsmom_roc_slow", "_qsmom_returns"])
    )
