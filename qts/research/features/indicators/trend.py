"""Trend indicators."""

from __future__ import annotations

import polars as pl

from qts.core.registry import Registry
from qts.research.features.base import BaseFeature


def _true_range_expr(
    *,
    high_column: str = "high",
    low_column: str = "low",
    close_column: str = "close",
) -> pl.Expr:
    previous_close = (
        pl.col(close_column).shift(1).over("symbol").fill_null(pl.col(close_column))
    )
    return pl.max_horizontal(
        [
            pl.col(high_column) - pl.col(low_column),
            (pl.col(high_column) - previous_close).abs(),
            (pl.col(low_column) - previous_close).abs(),
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
    """Average directional index. Accepts a single period or a list of periods."""

    def __init__(
        self,
        period: int | list[int] = 14,
        *,
        periods: list[int] | tuple[int, ...] | None = None,
        high_column: str = "high",
        low_column: str = "low",
        close_column: str = "close",
        price_column: str | None = None,
    ) -> None:
        resolved_periods = periods if periods is not None else period
        self.periods = (
            [resolved_periods]
            if isinstance(resolved_periods, int)
            else list(resolved_periods)
        )
        self.high_column = high_column
        self.low_column = low_column
        self.close_column = price_column or close_column

    def _compute_period(self, df: pl.DataFrame, period: int) -> pl.DataFrame:
        alpha = 1 / period
        return (
            df
            .with_columns(
                pl.col(self.high_column).diff().over("symbol").alias("_up_move"),
                (-pl.col(self.low_column).diff().over("symbol")).alias("_down_move"),
                _true_range_expr(
                    high_column=self.high_column,
                    low_column=self.low_column,
                    close_column=self.close_column,
                ).alias("_true_range"),
            )
            .with_columns(
                pl.when((pl.col("_up_move") > pl.col("_down_move")) & (pl.col("_up_move") > 0))
                .then(pl.col("_up_move")).otherwise(0.0).alias("_plus_dm"),
                pl.when((pl.col("_down_move") > pl.col("_up_move")) & (pl.col("_down_move") > 0))
                .then(pl.col("_down_move")).otherwise(0.0).alias("_minus_dm"),
            )
            .with_columns(
                pl.col("_true_range").ewm_mean(alpha=alpha, adjust=False, min_samples=period).over("symbol").alias("_atr"),
                pl.col("_plus_dm").ewm_mean(alpha=alpha, adjust=False, min_samples=period).over("symbol").alias("_plus_dms"),
                pl.col("_minus_dm").ewm_mean(alpha=alpha, adjust=False, min_samples=period).over("symbol").alias("_minus_dms"),
            )
            .with_columns(
                (100 * pl.col("_plus_dms") / (pl.col("_atr") + 1e-8)).alias("_plus_di"),
                (100 * pl.col("_minus_dms") / (pl.col("_atr") + 1e-8)).alias("_minus_di"),
            )
            .with_columns(
                (100 * (pl.col("_plus_di") - pl.col("_minus_di")).abs()
                 / (pl.col("_plus_di") + pl.col("_minus_di") + 1e-8)).alias("_dx")
            )
            .with_columns(
                pl.col("_dx").ewm_mean(alpha=alpha, adjust=False, min_samples=period).over("symbol").alias(f"adx_{period}")
            )
            .drop(["_up_move", "_down_move", "_true_range", "_plus_dm", "_minus_dm",
                   "_atr", "_plus_dms", "_minus_dms", "_plus_di", "_minus_di", "_dx"])
        )

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.sort(["symbol", "date"])
        for period in self.periods:
            result = self._compute_period(result, period)
        return self._validate_append_only(df, result)
