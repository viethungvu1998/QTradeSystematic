"""Config-driven workflow entry point."""

from __future__ import annotations

from decimal import Decimal

import polars as pl

from qts.config.builder import Config
from qts.core.instrument import AssetType, Instrument
from qts.data.manager import DataManager
from qts.execution.router import OrderRouter
from qts.execution.sync import PositionSync
from qts.orchestration.tasks.data_tasks import download_fundamentals, download_ohlcv
from qts.orchestration.tasks.execution_tasks import execute_rebalance
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


async def qts_flow(config_path: str):
    """Single workflow entry point."""

    resolved = Config.build(config_path)
    config = resolved.raw
    manager = DataManager(
        stock_source=resolved.stock_source,
        crypto_source=resolved.crypto_source,
        storage=resolved.storage,
        cache=resolved.cache,
        bundle_adapter=resolved.bundle_adapter if "stock" in config.asset_types else None,
    )
    ohlcv = await download_ohlcv(config, manager)
    fundamentals = await download_fundamentals(config, manager) if config.features.fundamental else pl.DataFrame()
    if fundamentals.height:
        for feature in resolved.feature_pipeline.features:
            if hasattr(feature, "fundamentals"):
                feature.fundamentals = fundamentals
    featured = build_features(config, resolved.feature_pipeline, ohlcv)
    result = run_backtest(config, resolved.engine, resolved.strategy, featured)
    if config.workflow in {"research", "validation"}:
        return result

    brokers = {}
    if resolved.stock_broker is not None:
        brokers[AssetType.STOCK] = resolved.stock_broker
    if resolved.crypto_broker is not None:
        brokers[AssetType.CRYPTO] = resolved.crypto_broker
    router = OrderRouter(brokers)
    syncer = PositionSync()
    target_weights = _target_weights_from_signals(result.signals)
    instruments = {
        symbol: Instrument(
            symbol=symbol,
            asset_type=AssetType.from_symbol(symbol),
            exchange="AUTO",
            currency=symbol.split("/")[-1] if "/" in symbol else "USD",
        )
        for symbol in target_weights
    }
    latest_prices = {
        row["symbol"]: Decimal(str(row["close"]))
        for row in featured.sort(["symbol", "date"]).group_by("symbol").agg(pl.col("close").last()).iter_rows(named=True)
    }
    orders = syncer.compute_deltas(target_weights, [], instruments, latest_prices, Decimal("100000"))
    fills = await execute_rebalance(router, orders)
    return {
        "result": result,
        "orders": orders,
        "fills": fills,
        "schedule": config.schedule,
    }
