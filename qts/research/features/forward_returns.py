"""Forward returns."""

from __future__ import annotations

import polars as pl

from qts.core.registry import Registry
from qts.research.features.base import BaseFeature


@Registry.register_feature("forward_returns")
class ForwardReturns(BaseFeature):
    """Target variable builder."""

    def __init__(self, periods: list[int]) -> None:
        self.periods = periods

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        original = df
        expressions = []
        for period in self.periods:
            expressions.append(
                ((pl.col("close").shift(-period).over("symbol") / pl.col("close")) - 1).alias(
                    f"forward_return_{period}"
                )
            )
        transformed = df.sort(["symbol", "date"]).with_columns(expressions)
        return self._validate_append_only(original, transformed)
