"""Research workflow tasks."""

from __future__ import annotations

import polars as pl


def build_features(config, feature_pipeline, df: pl.DataFrame) -> pl.DataFrame:
    """Build feature frame."""

    return feature_pipeline.fit_transform(df)


def run_backtest(config, engine, strategy, df: pl.DataFrame):
    """Run configured backtest."""

    return engine.run(strategy, df, config)
