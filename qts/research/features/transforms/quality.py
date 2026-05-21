"""Price-quality preprocessor — wraps existing anomaly detection utilities."""

from __future__ import annotations

import polars as pl

from qts.core.registry import Registry


@Registry.register_transform("price_preprocessor")
def price_preprocessor(
    df: pl.DataFrame,
    *,
    min_trading_days: int = 504,
    remove_large_gaps: bool = True,
    remove_low_volume: bool = True,
    symbol_column: str = "symbol",
    date_column: str = "date",
) -> pl.DataFrame:
    """Drop symbols that fail minimum history or data-quality checks."""
    try:
        from qts.data.preprocessor import (  # type: ignore[import]
            flag_anomalies,
            remove_flagged_symbols,
        )

        flagged = flag_anomalies(
            df,
            remove_large_gaps=remove_large_gaps,
            remove_low_volume=remove_low_volume,
        )
        return remove_flagged_symbols(flagged)
    except ImportError:
        pass

    counts = (
        df.group_by(symbol_column)
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") >= min_trading_days)
    )
    return df.filter(pl.col(symbol_column).is_in(counts[symbol_column]))
