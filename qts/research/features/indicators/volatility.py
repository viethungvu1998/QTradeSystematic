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
    """Average true range. Set normalize=True to output ATR / close instead of raw ATR."""

    def __init__(self, periods: list[int] | tuple[int, ...] = (14,), normalize: bool = False) -> None:
        self.periods = list(periods)
        self.normalize = normalize

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.sort(["symbol", "date"]).with_columns(_true_range_expr().alias("_true_range"))
        for period in self.periods:
            raw_atr = (
                pl.col("_true_range")
                .ewm_mean(alpha=1 / period, adjust=False, min_samples=period)
                .over("symbol")
            )
            col_name = f"atr_norm_{period}" if self.normalize else f"atr_{period}"
            expr = (raw_atr / (pl.col("close") + 1e-8)) if self.normalize else raw_atr
            result = result.with_columns(expr.alias(col_name))
        result = result.drop("_true_range")
        return self._validate_append_only(df, result)


@Registry.register_feature("bollinger")
class BollingerFeature(BaseFeature):
    """Bollinger bands. Accepts a single window or a list of windows.
    When pct_b=True outputs bb_pct_b_{window} (position within bands, clipped [0,1]).
    When pct_b=False (default) outputs bb_mid, bb_upper, bb_lower per window.
    """

    def __init__(self, window: int | list[int] = 20, n_std: float = 2.0, pct_b: bool = False) -> None:
        self.windows = [window] if isinstance(window, int) else list(window)
        self.n_std = n_std
        self.pct_b = pct_b

    def _compute_window(self, df: pl.DataFrame, w: int) -> pl.DataFrame:
        result = (
            df
            .with_columns(
                pl.col("close").rolling_mean(w, min_samples=w).over("symbol").alias("_bmid"),
                pl.col("close").rolling_std(w, min_samples=w, ddof=0).over("symbol").alias("_bstd"),
            )
            .with_columns(
                (pl.col("_bmid") + self.n_std * pl.col("_bstd")).alias("_bup"),
                (pl.col("_bmid") - self.n_std * pl.col("_bstd")).alias("_blo"),
            )
        )
        if self.pct_b:
            result = result.with_columns(
                ((pl.col("close") - pl.col("_blo")) / (pl.col("_bup") - pl.col("_blo") + 1e-8))
                .clip(0.0, 1.0).alias(f"bb_pct_b_{w}")
            )
        else:
            result = result.with_columns(
                pl.col("_bmid").alias(f"bb_mid_{w}"),
                pl.col("_bup").alias(f"bb_upper_{w}"),
                pl.col("_blo").alias(f"bb_lower_{w}"),
            )
        return result.drop(["_bmid", "_bstd", "_bup", "_blo"])

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.sort(["symbol", "date"])
        for w in self.windows:
            result = self._compute_window(result, w)
        return self._validate_append_only(df, result)


@Registry.register_feature("hist_vol")
class HistVolFeature(BaseFeature):
    """Historical volatility."""

    def __init__(
        self,
        periods: list[int] | tuple[int, ...] = (20, 60),
        *,
        windows: list[int] | tuple[int, ...] | None = None,
        price_column: str = "close",
    ) -> None:
        self.periods = list(windows if windows is not None else periods)
        self.price_column = price_column

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        base_return = (
            pl.col("close").log() - pl.col("close").shift(1).over("symbol").log()
            if self.price_column == "close"
            else pl.col(self.price_column) - pl.col(self.price_column).shift(1).over("symbol")
        )
        result = df.sort(["symbol", "date"]).with_columns(base_return.alias("_log_return"))
        for period in self.periods:
            result = result.with_columns(
                pl.col("_log_return")
                .rolling_std(period, min_samples=period, ddof=0)
                .over("symbol")
                .alias(f"hist_vol_{period}")
            )
        result = result.drop("_log_return")
        return self._validate_append_only(df, result)
