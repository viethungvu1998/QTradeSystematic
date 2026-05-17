"""Research workflow tasks."""

from __future__ import annotations

import polars as pl

from qts.orchestration.prefect_compat import task


@task(retries=2, retry_delay_seconds=60, name="build-features")
def build_features(config, feature_pipeline, df: pl.DataFrame) -> pl.DataFrame:
    """Build feature frame."""

    return feature_pipeline.fit_transform(df)


@task(retries=2, retry_delay_seconds=60, name="run-backtest")
def run_backtest(
    config,
    engine,
    strategy,
    df: pl.DataFrame,
    pipeline=None,
    ohlcv: pl.DataFrame | None = None,
):
    """Run configured backtest."""

    return engine.run(strategy, df, config, pipeline=pipeline, ohlcv=ohlcv)
