"""Base feature contract."""

from __future__ import annotations

import polars as pl


class BaseFeature:
    """Feature contract."""

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError

    def _validate_append_only(self, original: pl.DataFrame, transformed: pl.DataFrame) -> pl.DataFrame:
        missing = set(original.columns) - set(transformed.columns)
        if missing:
            raise ValueError(f"Feature dropped columns: {sorted(missing)}")
        return transformed
