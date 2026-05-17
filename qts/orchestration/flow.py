"""Config-driven workflow entry point."""

from __future__ import annotations

from decimal import Decimal

import polars as pl

from qts.config.builder import Config
from qts.execution.sync import PositionSync
from qts.orchestration.prefect_compat import flow
from qts.orchestration.runtime import build_data_manager, build_order_router, resolved_brokers
from qts.orchestration.tasks.data_tasks import (
    download_fundamentals,
    download_futures_ohlcv,
    download_ohlcv,
)
from qts.orchestration.tasks.execution_tasks import execute_rebalance, sync_positions
from qts.orchestration.tasks.research_tasks import build_features, run_backtest

_RESEARCH_WORKFLOWS = frozenset({"research", "validation"})


def _target_weights_from_signals(signals: pl.DataFrame) -> dict[str, Decimal]:
    latest = signals.sort(["symbol", "date"]).group_by("symbol").agg(
        pl.col("signal").last(),
        pl.col("weight").last(),
    )
    weights = {}
    for record in latest.iter_rows(named=True):
        signal = Decimal(str(record["signal"]))
        weight = Decimal(str(record["weight"]))
        weights[record["symbol"]] = abs(signal * weight)
    return weights


def _backtest_kwargs(config, pipeline, ohlcv: pl.DataFrame) -> dict[str, object]:
    if config.workflow in _RESEARCH_WORKFLOWS:
        return {"pipeline": pipeline, "ohlcv": ohlcv}
    return {}


@flow(name="qts-main", log_prints=True)
async def qts_flow(config_path: str):
    """Single workflow entry point."""

    resolved = Config.build(config_path)
    config = resolved.raw
    manager = build_data_manager(resolved)
    ohlcv = await download_ohlcv(config, manager)
    futures_ohlcv = await download_futures_ohlcv(config, manager)
    if futures_ohlcv.height:
        ohlcv = pl.concat([ohlcv, futures_ohlcv], how="vertical")
    fundamentals = (
        await download_fundamentals(config, manager)
        if resolved.feature_pipeline.requires_fundamentals()
        else pl.DataFrame()
    )
    feature_pipeline = resolved.with_fundamentals(fundamentals)
    featured = build_features(config, feature_pipeline, ohlcv)
    result = run_backtest(
        config,
        resolved.engine,
        resolved.strategy,
        featured,
        **_backtest_kwargs(config, feature_pipeline, ohlcv),
    )
    if config.workflow in _RESEARCH_WORKFLOWS:
        return result

    brokers = resolved_brokers(resolved)
    router = build_order_router(resolved)
    syncer = PositionSync()
    target_weights = _target_weights_from_signals(result.signals)
    orders = await sync_positions(config, syncer, brokers, target_weights, featured)
    fills = await execute_rebalance(config, router, orders)
    return {
        "result": result,
        "orders": orders,
        "fills": fills,
        "schedule": config.schedule,
    }
