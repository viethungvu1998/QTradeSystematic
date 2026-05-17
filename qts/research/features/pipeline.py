"""Feature pipeline utilities."""

from __future__ import annotations

import polars as pl

from qts.research.features.base import BaseFeature
from qts.research.features.preprocessor import preprocess_ohlcv


class FeaturePipeline:
    """Runs features sequentially."""

    def __init__(self, features: list[BaseFeature]) -> None:
        self.features = features

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        current = preprocess_ohlcv(df)
        for feature in self.features:
            current = feature.fit_transform(current)
        return current
