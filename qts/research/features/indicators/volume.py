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
    """Volume divided by rolling mean volume. Accepts a single window or a list of windows."""

    def __init__(
        self,
        window: int | list[int] = 20,
        *,
        windows: list[int] | tuple[int, ...] | None = None,
        volume_column: str = "volume",
    ) -> None:
        resolved_windows = windows if windows is not None else window
        self.windows = (
            [resolved_windows]
            if isinstance(resolved_windows, int)
            else list(resolved_windows)
        )
        self.volume_column = volume_column

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.sort(["symbol", "date"])
        for w in self.windows:
            result = result.with_columns(
                (
                    pl.col(self.volume_column)
                    / (
                        pl.col(self.volume_column)
                        .rolling_mean(w, min_samples=w)
                        .over("symbol")
                        + 1e-8
                    )
                )
                .alias(f"vol_ratio_{w}")
            )
        return self._validate_append_only(df, result)
