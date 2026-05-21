"""Feature pipeline utilities."""

from __future__ import annotations

import polars as pl

from qts.research.features.base import BaseFeature
from qts.research.features.preprocessor import preprocess_ohlcv


class FeaturePipeline:
    """Runs features sequentially."""

    def __init__(
        self,
        features: list[BaseFeature],
        transforms: list[BaseFeature] | None = None,
    ) -> None:
        self.transforms: list[BaseFeature] = transforms or []
        self.features = features

    def requires_fundamentals(self) -> bool:
        return any(
            feature.requires_fundamentals()
            for feature in [*self.transforms, *self.features]
        )

    def with_fundamentals(self, fundamentals: pl.DataFrame) -> FeaturePipeline:
        return FeaturePipeline(
            [feature.with_fundamentals(fundamentals) for feature in self.features],
            transforms=[
                transform.with_fundamentals(fundamentals)
                for transform in self.transforms
            ],
        )

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        current = df
        for step in self.transforms:
            current = step.fit_transform(current)
        current = preprocess_ohlcv(current)
        for feature in self.features:
            current = feature.fit_transform(current)
        return current
