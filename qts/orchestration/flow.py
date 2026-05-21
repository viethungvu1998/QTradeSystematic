"""Config-driven workflow entry point."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import polars as pl

from qts.config.builder import Config
from qts.core.instrument import AssetType, Instrument
from qts.core.observability import PortfolioSnapshot
from qts.core.portfolio import Portfolio, Position
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
from qts.utils.export import export_live_portfolio, export_portfolio_snapshots, export_trade_log
from qts.utils.paths import backtest_exports_dir, live_portfolio_dir

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
    return await run_resolved_config(resolved)


async def run_resolved_config(resolved):
    """Run an already resolved config through the canonical QTS path."""

    config = resolved.raw
    tracker = getattr(resolved, "tracker", None)
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
    _export_backtest_observability(result, tracker)
    if config.workflow in _RESEARCH_WORKFLOWS:
        return result

    brokers = resolved_brokers(resolved)
    router = build_order_router(resolved)
    syncer = PositionSync()
    target_weights = _target_weights_from_signals(result.signals)
    orders, live_snapshot = await sync_positions(config, syncer, brokers, target_weights, featured)
    fills = await execute_rebalance(config, router, orders)
    live_path = _live_export_path(live_snapshot.timestamp)
    export_live_portfolio(
        _portfolio_from_snapshot(live_snapshot),
        live_snapshot.timestamp,
        live_path,
    )
    _log_artifact(tracker, live_path)
    return {
        "result": result,
        "orders": orders,
        "fills": fills,
        "live_snapshot": live_snapshot,
        "schedule": config.schedule,
    }


def _export_backtest_observability(result, tracker) -> None:
    stamp = _timestamp_slug()
    if getattr(result, "trade_log", pl.DataFrame()).height > 0:
        trade_log_path = backtest_exports_dir() / f"{result.engine_name}_{stamp}_trade_log.csv"
        export_trade_log(result, trade_log_path)
        _log_artifact(tracker, trade_log_path)
    if getattr(result, "portfolio_snapshots", pl.DataFrame()).height > 0:
        snapshots_path = backtest_exports_dir() / f"{result.engine_name}_{stamp}_snapshots.csv"
        export_portfolio_snapshots(result, snapshots_path)
        _log_artifact(tracker, snapshots_path)


def _live_export_path(timestamp: datetime) -> Path:
    return live_portfolio_dir() / f"{_timestamp_slug(timestamp)}_portfolio.csv"


def _portfolio_from_snapshot(snapshot: PortfolioSnapshot) -> Portfolio:
    positions = [
        Position(
            Instrument(
                token.token,
                AssetType.from_symbol(token.token),
                "AUTO",
                "",
            ),
            Decimal(str(token.quantity)),
            Decimal(str(token.current_price)),
            Decimal(str(token.avg_buy_price)),
        )
        for token in snapshot.tokens
    ]
    return Portfolio(positions=positions, cash=Decimal("0"))


def _timestamp_slug(timestamp: datetime | None = None) -> str:
    value = timestamp or datetime.now(UTC)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _log_artifact(tracker, path: Path) -> None:
    if tracker is not None and hasattr(tracker, "log_artifact"):
        tracker.log_artifact(str(path))
