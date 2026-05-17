"""Statistical indicators."""

from __future__ import annotations

import polars as pl

from qts.core.registry import Registry
from qts.research.features.base import BaseFeature


@Registry.register_feature("zscore")
class ZScoreFeature(BaseFeature):
    """Rolling z-score."""

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
