"""Statistical indicators."""

from __future__ import annotations

import polars as pl

from qts.core.registry import Registry
from qts.research.features.base import BaseFeature


@Registry.register_feature("zscore")
class ZScoreFeature(BaseFeature):
    """Rolling z-score of close price."""

    def __init__(self, windows: list[int] | tuple[int, ...] = (21, 63)) -> None:
        self.windows = list(windows)

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.sort(["symbol", "date"])
        for window in self.windows:
            mean_column = f"_zscore_mean_{window}"
            std_column = f"_zscore_std_{window}"
            result = (
                result.with_columns(
                    pl.col("close")
                    .rolling_mean(window, min_samples=window)
                    .over("symbol")
                    .alias(mean_column),
                    pl.col("close")
                    .rolling_std(window, min_samples=window, ddof=0)
                    .over("symbol")
                    .alias(std_column),
                )
                .with_columns(
                    ((pl.col("close") - pl.col(mean_column)) / (pl.col(std_column) + 1e-8)).alias(
                        f"zscore_{window}"
                    )
                )
                .drop([mean_column, std_column])
            )
        return self._validate_append_only(df, result)


@Registry.register_feature("zscore_ret")
class ZScoreRetFeature(BaseFeature):
    """Rolling z-score of log returns — different from ZScoreFeature which uses close price."""

    def __init__(self, windows: list[int] | tuple[int, ...] = (7, 14, 21)) -> None:
        self.windows = list(windows)

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.sort(["symbol", "date"]).with_columns(
            (pl.col("close").log() - pl.col("close").shift(1).over("symbol").log()).alias("_lr")
        )
        for w in self.windows:
            result = (
                result
                .with_columns(
                    pl.col("_lr").rolling_mean(w, min_samples=w).over("symbol").alias("_rm"),
                    pl.col("_lr").rolling_std(w, min_samples=w, ddof=0).over("symbol").alias("_rs"),
                )
                .with_columns(
                    ((pl.col("_lr") - pl.col("_rm")) / (pl.col("_rs") + 1e-8)).alias(f"zscore_ret_{w}")
                )
                .drop(["_rm", "_rs"])
            )
        return self._validate_append_only(df, result.drop("_lr"))
