"""Execution workflow tasks."""

from __future__ import annotations

from decimal import Decimal

import polars as pl

from qts.core.instrument import AssetType, Instrument
from qts.orchestration.prefect_compat import task

_CURRENCY_DEFAULT: dict[AssetType, str] = {
    AssetType.STOCK: "USD",
    AssetType.VN_STOCK: "VND",
    AssetType.COMMODITY: "USD",
    AssetType.CRYPTO: "",
}


def _instrument_for_symbol(symbol: str) -> Instrument:
    asset_type = AssetType.from_symbol(symbol)
    currency = symbol.split("/")[-1] if asset_type is AssetType.CRYPTO else _CURRENCY_DEFAULT[asset_type]
    return Instrument(
        symbol=symbol,
        asset_type=asset_type,
        exchange="AUTO",
        currency=currency,
    )


@task(retries=2, retry_delay_seconds=60, name="sync-positions")
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
    for symbol in target_weights:
        instruments.setdefault(symbol, _instrument_for_symbol(symbol))
    return syncer.compute_deltas(target_weights, positions, instruments, latest_prices, account_value)


@task(retries=2, retry_delay_seconds=60, name="execute-rebalance")
async def execute_rebalance(config, router, orders):
    """Execute generated orders."""

    return await router.execute(orders)
