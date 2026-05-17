"""Volatility indicators."""

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


@Registry.register_feature("atr")
class ATRFeature(BaseFeature):
    """Average true range."""

    def __init__(self, periods: list[int] | tuple[int, ...] = (14,)) -> None:
        self.periods = list(periods)

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.sort(["symbol", "date"]).with_columns(_true_range_expr().alias("_true_range"))
        for period in self.periods:
            result = result.with_columns(
                pl.col("_true_range")
                .ewm_mean(alpha=1 / period, adjust=False, min_samples=period)
                .over("symbol")
                .alias(f"atr_{period}")
            )
        result = result.drop("_true_range")
        return self._validate_append_only(df, result)


@Registry.register_feature("bollinger")
class BollingerFeature(BaseFeature):
    """Bollinger bands."""

    def __init__(self, window: int = 20, n_std: float = 2.0) -> None:
        self.window = window
        self.n_std = n_std

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        result = (
            df.sort(["symbol", "date"])
            .with_columns(
                pl.col("close")
                .rolling_mean(self.window, min_samples=self.window)
                .over("symbol")
                .alias(f"bb_mid_{self.window}"),
                pl.col("close")
                .rolling_std(self.window, min_samples=self.window, ddof=0)
                .over("symbol")
                .alias("_bb_std"),
            )
            .with_columns(
                (pl.col(f"bb_mid_{self.window}") + (self.n_std * pl.col("_bb_std"))).alias(
                    f"bb_upper_{self.window}"
                ),
                (pl.col(f"bb_mid_{self.window}") - (self.n_std * pl.col("_bb_std"))).alias(
                    f"bb_lower_{self.window}"
                ),
            )
            .drop("_bb_std")
        )
        return self._validate_append_only(df, result)


@Registry.register_feature("hist_vol")
class HistVolFeature(BaseFeature):
    """Historical volatility."""

    def __init__(self, periods: list[int] | tuple[int, ...] = (20, 60)) -> None:
        self.periods = list(periods)

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.sort(["symbol", "date"]).with_columns(
            (pl.col("close").log() - pl.col("close").shift(1).over("symbol").log()).alias(
                "_log_return"
            )
        )
        for period in self.periods:
            result = result.with_columns(
                pl.col("_log_return")
                .rolling_std(period, min_samples=period, ddof=0)
                .over("symbol")
                .alias(f"hist_vol_{period}")
            )
        result = result.drop("_log_return")
        return self._validate_append_only(df, result)
