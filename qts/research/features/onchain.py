"""On-chain feature joins."""

from __future__ import annotations

import polars as pl

from qts.core.registry import Registry
from qts.research.features.base import BaseFeature


@Registry.register_feature("onchain")
class OnchainFeatures(BaseFeature):
    """Crypto-only on-chain features."""

    def __init__(self, onchain: pl.DataFrame | None = None) -> None:
        self.onchain = onchain if onchain is not None else pl.DataFrame()

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        original = df
        if self.onchain.is_empty():
            return df
        required = {"date", "symbol", "nvt_ratio", "active_addresses"}
        if not required.issubset(self.onchain.columns):
            return df
        if not set(df["symbol"].unique().to_list()) & set(self.onchain["symbol"].unique().to_list()):
            return df
        transformed = df.join(self.onchain.select(sorted(required)), on=["date", "symbol"], how="left")
        return self._validate_append_only(original, transformed)
