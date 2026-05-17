"""Execution workflow tasks."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

import polars as pl

from qts.core.instrument import AssetType, Instrument
from qts.orchestration.prefect_compat import task


def _static_currency(code: str) -> Callable[[str], str]:
    return lambda symbol: code


def _quote_currency(symbol: str) -> str:
    return symbol.removeprefix("PERP:").split("/")[-1]


_CURRENCY_BY_ASSET_TYPE: dict[AssetType, Callable[[str], str]] = {
    AssetType.STOCK: _static_currency("USD"),
    AssetType.VN_STOCK: _static_currency("VND"),
    AssetType.VN_WARRANT: _static_currency("VND"),
    AssetType.VN_FUTURES: _static_currency("VND"),
    AssetType.COMMODITY: _static_currency("USD"),
    AssetType.CRYPTO: _quote_currency,
    AssetType.CRYPTO_FUTURES: _quote_currency,
}


def _instrument_for_symbol(symbol: str) -> Instrument:
    asset_type = AssetType.from_symbol(symbol)
    return Instrument(
        symbol=symbol,
        asset_type=asset_type,
        exchange="AUTO",
        currency=_CURRENCY_BY_ASSET_TYPE[asset_type](symbol),
    )


@task(retries=2, retry_delay_seconds=60, name="sync-positions")
async def sync_positions(
    config,
    syncer,
    brokers,
    target_weights: dict[str, Decimal],
    data: pl.DataFrame,
):
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
    return syncer.compute_deltas(
        target_weights,
        positions,
        instruments,
        latest_prices,
        account_value,
    )


@task(retries=2, retry_delay_seconds=60, name="execute-rebalance")
async def execute_rebalance(config, router, orders):
    """Execute generated orders."""

    return await router.execute(orders)
