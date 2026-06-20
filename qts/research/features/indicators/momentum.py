"""Momentum indicators."""

from __future__ import annotations

import polars as pl

from qts.core.registry import Registry
from qts.research.features.base import BaseFeature


def _rsi_expr(period: int, price_column: str) -> pl.Expr:
    delta = pl.col(price_column).diff().over("symbol")
    gains = pl.when(delta > 0).then(delta).otherwise(0.0)
    losses = pl.when(delta < 0).then(-delta).otherwise(0.0)
    avg_gain = gains.ewm_mean(alpha=1 / period, adjust=False, min_samples=period).over("symbol")
    avg_loss = losses.ewm_mean(alpha=1 / period, adjust=False, min_samples=period).over("symbol")
    relative_strength = avg_gain / (avg_loss + 1e-8)
    return 100 - (100 / (1 + relative_strength))


@Registry.register_feature("rsi")
class RSIFeature(BaseFeature):
    """Relative strength index."""

    def __init__(
        self,
        periods: list[int] | tuple[int, ...] = (14,),
        price_column: str = "close",
    ) -> None:
        self.periods = list(periods)
        self.price_column = price_column

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.sort(["symbol", "date"]).with_columns(
            [
                _rsi_expr(period, self.price_column).alias(f"rsi_{period}")
                for period in self.periods
            ]
        )
        return self._validate_append_only(df, result)


@Registry.register_feature("roc")
class ROCFeature(BaseFeature):
    """Rate of change."""

    def __init__(
        self,
        periods: list[int] | tuple[int, ...] = (1, 5, 21),
        price_column: str = "close",
    ) -> None:
        self.periods = list(periods)
        self.price_column = price_column

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.sort(["symbol", "date"]).with_columns(
            [
                (
                    (
                        pl.col(self.price_column)
                        / pl.col(self.price_column).shift(period).over("symbol")
                    )
                    - 1
                ).alias(f"roc_{period}")
                for period in self.periods
            ]
        )
        return self._validate_append_only(df, result)


@Registry.register_feature("ma")
class MovingAverageFeature(BaseFeature):
    """Simple moving average of close price."""

    def __init__(
        self,
        periods: list[int] | tuple[int, ...] = (20,),
        price_column: str = "close",
    ) -> None:
        self.periods = list(periods)
        self.price_column = price_column

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.sort(["symbol", "date"]).with_columns(
            [
                pl.col(self.price_column)
                .rolling_mean(period, min_samples=period)
                .over("symbol")
                .alias(f"ma_{period}")
                for period in self.periods
            ]
        )
        return self._validate_append_only(df, result)
