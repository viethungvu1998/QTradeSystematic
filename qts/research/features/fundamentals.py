"""Fundamental feature joins."""

from __future__ import annotations

import polars as pl

from qts.core.registry import Registry
from qts.research.features.base import BaseFeature


@Registry.register_feature("fundamental")
class FundamentalFeatures(BaseFeature):
    """Stock-only fundamental features."""

    def __init__(self, fundamentals: pl.DataFrame | None = None) -> None:
        self.fundamentals = fundamentals if fundamentals is not None else pl.DataFrame()

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        original = df
        if self.fundamentals.is_empty():
            return df
        required = {"symbol", "pe_ratio", "ev_ebitda"}
        if not required.issubset(self.fundamentals.columns):
            return df
        if not set(df["symbol"].unique().to_list()) & set(self.fundamentals["symbol"].unique().to_list()):
            return df
        transformed = df.join(self.fundamentals.select(sorted(required)), on="symbol", how="left")
        return self._validate_append_only(original, transformed)
