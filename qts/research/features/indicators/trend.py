"""Trend indicators."""

from __future__ import annotations

import polars as pl

from qts.core.registry import Registry
from qts.research.features.base import BaseFeature


def _true_range_expr() -> pl.Expr:
    previous_close = pl.col("close").shift(1).over("symbol").fill_null(pl.col("close"))
    return pl.max_horizontal(
        [
            pl.col("high") - pl.col("low"),
            (pl.col("high") - previous_close).abs(),
            (pl.col("low") - previous_close).abs(),
        ]
    )


@Registry.register_feature("macd")
class MACDFeature(BaseFeature):
    """Moving average convergence divergence."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        result = (
            df.sort(["symbol", "date"])
            .with_columns(
                pl.col("close")
                .ewm_mean(span=self.fast, adjust=False, min_samples=self.fast)
                .over("symbol")
                .alias("_ema_fast"),
                pl.col("close")
                .ewm_mean(span=self.slow, adjust=False, min_samples=self.slow)
                .over("symbol")
                .alias("_ema_slow"),
            )
            .with_columns((pl.col("_ema_fast") - pl.col("_ema_slow")).alias("macd_line"))
            .with_columns(
                pl.col("macd_line")
                .ewm_mean(span=self.signal, adjust=False, min_samples=self.signal)
                .over("symbol")
                .alias("macd_signal")
            )
            .with_columns((pl.col("macd_line") - pl.col("macd_signal")).alias("macd_hist"))
            .drop(["_ema_fast", "_ema_slow"])
        )
        return self._validate_append_only(df, result)


@Registry.register_feature("adx")
class ADXFeature(BaseFeature):
    """Average directional index."""

    def __init__(self, period: int = 14) -> None:
        self.period = period

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        alpha = 1 / self.period
        result = (
            df.sort(["symbol", "date"])
            .with_columns(
                pl.col("high").diff().over("symbol").alias("_up_move"),
                (-pl.col("low").diff().over("symbol")).alias("_down_move"),
                _true_range_expr().alias("_true_range"),
            )
            .with_columns(
                pl.when((pl.col("_up_move") > pl.col("_down_move")) & (pl.col("_up_move") > 0))
                .then(pl.col("_up_move"))
                .otherwise(0.0)
                .alias("_plus_dm"),
                pl.when((pl.col("_down_move") > pl.col("_up_move")) & (pl.col("_down_move") > 0))
                .then(pl.col("_down_move"))
                .otherwise(0.0)
                .alias("_minus_dm"),
            )
            .with_columns(
                pl.col("_true_range")
                .ewm_mean(alpha=alpha, adjust=False, min_samples=self.period)
                .over("symbol")
                .alias("_atr"),
                pl.col("_plus_dm")
                .ewm_mean(alpha=alpha, adjust=False, min_samples=self.period)
                .over("symbol")
                .alias("_plus_dm_smoothed"),
                pl.col("_minus_dm")
                .ewm_mean(alpha=alpha, adjust=False, min_samples=self.period)
                .over("symbol")
                .alias("_minus_dm_smoothed"),
            )
            .with_columns(
                ((100 * pl.col("_plus_dm_smoothed")) / (pl.col("_atr") + 1e-8)).alias("_plus_di"),
                ((100 * pl.col("_minus_dm_smoothed")) / (pl.col("_atr") + 1e-8)).alias("_minus_di"),
            )
            .with_columns(
                (
                    (100 * (pl.col("_plus_di") - pl.col("_minus_di")).abs())
                    / (pl.col("_plus_di") + pl.col("_minus_di") + 1e-8)
                ).alias("_dx")
            )
            .with_columns(
                pl.col("_dx")
                .ewm_mean(alpha=alpha, adjust=False, min_samples=self.period)
                .over("symbol")
                .alias(f"adx_{self.period}")
            )
            .drop(
                [
                    "_up_move",
                    "_down_move",
                    "_true_range",
                    "_plus_dm",
                    "_minus_dm",
                    "_atr",
                    "_plus_dm_smoothed",
                    "_minus_dm_smoothed",
                    "_plus_di",
                    "_minus_di",
                    "_dx",
                ]
            )
        )
        return self._validate_append_only(df, result)
