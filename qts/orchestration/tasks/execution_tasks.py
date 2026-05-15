"""Execution workflow tasks."""

from __future__ import annotations

from decimal import Decimal

import polars as pl

from qts.core.errors import BrokerError


async def sync_positions(config, syncer, brokers, target_weights: dict[str, Decimal], data: pl.DataFrame):
    """Compute rebalance orders."""

    positions = []
    account_value = Decimal("0")
    for broker in brokers.values():
        positions.extend(await broker.get_positions())
        account_value += await broker.get_account_value()
    latest = (
        data.sort(["symbol", "date"])
        .group_by("symbol")
        .agg(pl.col("close").last())
    )
    latest_prices = {
        row["symbol"]: Decimal(str(row["close"])) for row in latest.iter_rows(named=True)
    }
    instruments = {position.instrument.symbol: position.instrument for position in positions}
    for symbol, price in latest_prices.items():
        instruments.setdefault(symbol, next(position.instrument for position in positions if position.instrument.symbol == symbol) if positions and symbol in {position.instrument.symbol for position in positions} else None)
    instruments = {symbol: instrument for symbol, instrument in instruments.items() if instrument is not None}
    return syncer.compute_deltas(target_weights, positions, instruments, latest_prices, account_value)


async def execute_rebalance(router, orders):
    """Execute generated orders."""

    try:
        return await router.execute(orders)
    except BrokerError:
        raise
