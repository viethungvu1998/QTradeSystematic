"""Volume indicators."""

from __future__ import annotations

import polars as pl

from qts.core.registry import Registry
from qts.research.features.base import BaseFeature


@Registry.register_feature("obv")
class OBVFeature(BaseFeature):
    """On-balance volume."""

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        previous_close = pl.col("close").shift(1).over("symbol").fill_null(pl.col("close"))
        result = df.sort(["symbol", "date"]).with_columns(
            pl.when(pl.col("close") > previous_close)
            .then(pl.col("volume"))
            .when(pl.col("close") < previous_close)
            .then(-pl.col("volume"))
            .otherwise(0.0)
            .cum_sum()
            .over("symbol")
            .alias("obv")
        )
        return self._validate_append_only(df, result)


@Registry.register_feature("volume_ratio")
class VolumeRatioFeature(BaseFeature):
    """Volume divided by rolling mean volume."""

    def __init__(self, window: int = 20) -> None:
        self.window = window

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.sort(["symbol", "date"]).with_columns(
            (
                pl.col("volume")
                / pl.col("volume").rolling_mean(self.window, min_samples=self.window).over("symbol")
            ).alias(f"vol_ratio_{self.window}")
        )
        return self._validate_append_only(df, result)
