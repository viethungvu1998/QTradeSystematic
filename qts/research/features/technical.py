"""Technical indicators."""

from __future__ import annotations

import polars as pl

from qts.core.registry import Registry
from qts.research.features.base import BaseFeature


@Registry.register_feature("technical")
class TechnicalFeatures(BaseFeature):
    """Technical indicators that operate on OHLCV data."""

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        original = df
        enriched = (
            df.sort(["symbol", "date"])
            .with_columns(
                [
                    (pl.col("close").diff().over("symbol")).alias("close_delta"),
                    (pl.when(pl.col("close").diff().over("symbol") > 0)
                    .then(pl.col("close").diff().over("symbol"))
                    .otherwise(0))
                    .alias("gain"),
                    (pl.when(pl.col("close").diff().over("symbol") < 0)
                    .then((-pl.col("close").diff().over("symbol")))
                    .otherwise(0))
                    .alias("loss"),
                    (pl.col("high") - pl.col("low")).alias("true_range"),
                ]
            )
            .with_columns(
                [
                    (100 - (100 / (1 + (pl.col("gain").rolling_mean(14).over("symbol") / pl.col("loss").rolling_mean(14).over("symbol").replace(0, None))))).alias("rsi_14"),
                    pl.col("close").ewm_mean(span=12).over("symbol").alias("ema_12"),
                    pl.col("close").ewm_mean(span=26).over("symbol").alias("ema_26"),
                    pl.col("true_range").rolling_mean(14).over("symbol").alias("atr_14"),
                    pl.col("close").rolling_mean(20).over("symbol").alias("bb_mid_20"),
                    pl.col("close").rolling_std(20).over("symbol").alias("bb_std_20"),
                ]
            )
            .with_columns(
                [
                    (pl.col("ema_12") - pl.col("ema_26")).alias("macd"),
                    (pl.col("bb_mid_20") + 2 * pl.col("bb_std_20")).alias("bb_upper_20"),
                    (pl.col("bb_mid_20") - 2 * pl.col("bb_std_20")).alias("bb_lower_20"),
                ]
            )
            .drop(["close_delta", "gain", "loss", "true_range", "ema_12", "ema_26", "bb_std_20"])
        )
        return self._validate_append_only(original, enriched)
