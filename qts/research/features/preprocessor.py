"""OHLCV preprocessing helpers."""

from __future__ import annotations

import polars as pl


OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
EPSILON = 1e-10


def preprocess_ohlcv(df: pl.DataFrame, min_trading_days: int = 252) -> pl.DataFrame:
    """Normalize OHLCV inputs before feature computation."""

    filled_columns = [
        pl.col(column).forward_fill().over("symbol").backward_fill().over("symbol").alias(column)
        for column in OHLCV_COLUMNS
    ]
    bounded_columns = [
        pl.when(pl.col(column) <= 0)
        .then(pl.lit(EPSILON))
        .otherwise(pl.col(column))
        .alias(column)
        for column in OHLCV_COLUMNS
    ]
    price_columns = [pl.col(column) for column in ("open", "high", "low", "close")]
    return (
        df.unique(subset=["symbol", "date"], keep="last")
        .sort(["symbol", "date"])
        .with_columns(filled_columns)
        .with_columns(bounded_columns)
        .with_columns(
            pl.max_horizontal(price_columns).alias("high"),
            pl.min_horizontal(price_columns).alias("low"),
        )
        .with_columns(
            pl.col("open").clip(pl.col("low"), pl.col("high")).alias("open"),
            pl.col("close").clip(pl.col("low"), pl.col("high")).alias("close"),
        )
        .filter(pl.len().over("symbol") >= min_trading_days)
    )


def flag_anomalies(
    df: pl.DataFrame,
    max_gap_days: int = 7,
    volatility_threshold: float = 5.0,
    min_volume_threshold: float = 1.0,
    min_notional_usd: float | None = 1_000_000,
) -> pl.DataFrame:
    """Append diagnostic flag columns to a preprocessed OHLCV frame.

    Adds boolean columns (flag_large_gap, flag_high_price, flag_low_volume,
    flag_high_volume) without removing any rows. Designed to run *after*
    preprocess_ohlcv so the frame is already clean.

    Parameters
    ----------
    df:
        Preprocessed OHLCV frame with [date, symbol, open, high, low, close, volume].
    max_gap_days:
        Calendar days between consecutive bars above which flag_large_gap is set.
    volatility_threshold:
        Std-dev multiplier for per-symbol high-price and high-volume detection.
    min_volume_threshold:
        Daily share volume below which flag_low_volume is set.
    min_notional_usd:
        If provided, also set flag_low_volume when close × volume < this value.
    """
    # Gap flag
    result = (
        df.sort(["symbol", "date"])
        .with_columns(
            pl.col("date").diff().over("symbol").dt.total_days().alias("_gap_days")
        )
        .with_columns(
            (pl.col("_gap_days") > max_gap_days).fill_null(False).alias("flag_large_gap")
        )
        .drop("_gap_days")
    )

    # Per-symbol price statistics for anomaly detection
    stats = result.group_by("symbol").agg(
        pl.col("close").mean().alias("_price_mean"),
        pl.col("close").std().alias("_price_std"),
        pl.col("volume").mean().alias("_vol_mean"),
        pl.col("volume").std().alias("_vol_std"),
    )
    result = result.join(stats, on="symbol", how="left")

    price_cols = ["open", "high", "low", "close"]
    result = result.with_columns(
        pl.any_horizontal(
            [
                pl.col(c) > pl.col("_price_mean") + volatility_threshold * pl.col("_price_std")
                for c in price_cols
            ]
        ).alias("flag_high_price"),
        (
            pl.col("volume")
            > pl.col("_vol_mean") + volatility_threshold * pl.col("_vol_std")
        ).alias("flag_high_volume"),
    )

    low_vol_expr = pl.col("volume") < min_volume_threshold
    if min_notional_usd is not None:
        low_vol_expr = low_vol_expr | (pl.col("close") * pl.col("volume") < min_notional_usd)
    result = result.with_columns(low_vol_expr.alias("flag_low_volume"))

    return result.drop(["_price_mean", "_price_std", "_vol_mean", "_vol_std"])


def remove_flagged_symbols(
    df: pl.DataFrame,
    *,
    remove_anomalies: bool = False,
    remove_large_gaps: bool = False,
    remove_low_volume: bool = False,
    low_volume_fraction_threshold: float = 0.05,
) -> pl.DataFrame:
    """Drop symbols that triggered quality flags in flag_anomalies().

    Expects the flag columns produced by flag_anomalies() to be present.
    Each removal rule operates at the symbol level (drop the whole symbol,
    not just the flagged rows).

    Parameters
    ----------
    remove_anomalies:
        Drop symbols that ever triggered flag_high_price or flag_large_gap.
    remove_large_gaps:
        Drop symbols that ever had flag_large_gap.
    remove_low_volume:
        Drop symbols where the fraction of flag_low_volume days >=
        low_volume_fraction_threshold.
    low_volume_fraction_threshold:
        Fraction threshold (0–1) for the low-volume rule.
    """
    if remove_anomalies and "flag_high_price" in df.columns:
        bad = (
            df.filter(pl.col("flag_high_price") | pl.col("flag_large_gap"))
            .select("symbol")
            .unique()
        )
        if bad.height:
            df = df.filter(~pl.col("symbol").is_in(bad["symbol"].to_list()))

    if remove_large_gaps and "flag_large_gap" in df.columns:
        bad = df.filter(pl.col("flag_large_gap")).select("symbol").unique()
        if bad.height:
            df = df.filter(~pl.col("symbol").is_in(bad["symbol"].to_list()))

    if remove_low_volume and "flag_low_volume" in df.columns:
        summary = (
            df.group_by("symbol")
            .agg(
                pl.len().alias("total"),
                pl.col("flag_low_volume").sum().alias("low_vol_days"),
            )
            .with_columns((pl.col("low_vol_days") / pl.col("total")).alias("frac"))
        )
        bad = summary.filter(pl.col("frac") >= low_volume_fraction_threshold).select("symbol")
        if bad.height:
            df = df.filter(~pl.col("symbol").is_in(bad["symbol"].to_list()))

    return df
