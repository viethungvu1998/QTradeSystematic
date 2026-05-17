"""Config-driven workflow entry point."""

from __future__ import annotations

from decimal import Decimal

import polars as pl

from qts.config.builder import Config
from qts.core.instrument import AssetType
from qts.data.manager import DataManager
from qts.execution.router import OrderRouter
from qts.execution.sync import PositionSync
from qts.orchestration.prefect_compat import flow
from qts.orchestration.tasks.data_tasks import (
    download_fundamentals,
    download_futures_ohlcv,
    download_ohlcv,
)
from qts.orchestration.tasks.execution_tasks import execute_rebalance, sync_positions
from qts.orchestration.tasks.research_tasks import build_features, run_backtest


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


@flow(name="qts-main", log_prints=True)
async def qts_flow(config_path: str):
    """Single workflow entry point."""

    resolved = Config.build(config_path)
    config = resolved.raw
    manager = DataManager(
        stock_source=resolved.stock_source,
        vn_stock_source=resolved.vn_stock_source,
        crypto_source=resolved.crypto_source,
        crypto_futures_source=resolved.crypto_futures_source,
        storage=resolved.storage,
        cache=resolved.cache,
        bundle_adapter=resolved.bundle_adapter if "stock" in config.asset_types else None,
    )
    ohlcv = await download_ohlcv(config, manager)
    futures_ohlcv = await download_futures_ohlcv(config, manager)
    if futures_ohlcv.height:
        ohlcv = pl.concat([ohlcv, futures_ohlcv], how="vertical")
    fundamentals = await download_fundamentals(config, manager) if config.features.fundamental else pl.DataFrame()
    if fundamentals.height:
        for feature in resolved.feature_pipeline.features:
            if hasattr(feature, "fundamentals"):
                feature.fundamentals = fundamentals
    featured = build_features(config, resolved.feature_pipeline, ohlcv)
    if config.workflow in {"research", "validation"}:
        result = run_backtest(
            config,
            resolved.engine,
            resolved.strategy,
            featured,
            pipeline=resolved.feature_pipeline,
            ohlcv=ohlcv,
        )
        return result
    result = run_backtest(config, resolved.engine, resolved.strategy, featured)

    brokers = {}
    if resolved.stock_broker is not None:
        brokers[AssetType.STOCK] = resolved.stock_broker
    if resolved.vn_stock_broker is not None:
        brokers[AssetType.VN_STOCK] = resolved.vn_stock_broker
    if resolved.crypto_broker is not None:
        brokers[AssetType.CRYPTO] = resolved.crypto_broker
    router = OrderRouter(brokers)
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
