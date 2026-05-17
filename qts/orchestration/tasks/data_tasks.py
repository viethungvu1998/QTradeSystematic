"""Data-related workflow tasks."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import polars as pl

from qts.core.instrument import AssetType
from qts.data._schemas import DataType
from qts.data.vn_symbols import to_vn_futures_symbol, to_vn_warrant_request
from qts.orchestration.prefect_compat import task

SymbolResolver = Callable[[object], Iterable[str]]


def _stock_symbols(config) -> list[str]:
    return list(config.universe.stock)


def _vn_stock_symbols(config) -> list[str]:
    return list(config.universe.vn_stock)


def _vn_warrant_symbols(config) -> list[str]:
    return [to_vn_warrant_request(symbol) for symbol in config.universe.vn_warrant]


def _vn_futures_symbols(config) -> list[str]:
    return [to_vn_futures_symbol(symbol) for symbol in config.universe.vn_futures]


def _crypto_symbols(config) -> list[str]:
    return list(config.universe.crypto)


def _crypto_futures_symbols(config) -> list[str]:
    return list(config.universe.crypto_futures)


_SYMBOL_RESOLVERS: dict[AssetType, SymbolResolver] = {
    AssetType.STOCK: _stock_symbols,
    AssetType.VN_STOCK: _vn_stock_symbols,
    AssetType.VN_WARRANT: _vn_warrant_symbols,
    AssetType.VN_FUTURES: _vn_futures_symbols,
    AssetType.CRYPTO: _crypto_symbols,
    AssetType.CRYPTO_FUTURES: _crypto_futures_symbols,
}

_OHLCV_ASSET_TYPES = (
    AssetType.STOCK,
    AssetType.VN_STOCK,
    AssetType.VN_WARRANT,
    AssetType.CRYPTO,
)
_FUTURES_ASSET_TYPES = (
    AssetType.VN_FUTURES,
    AssetType.CRYPTO_FUTURES,
)


def requested_symbols(config, asset_types: Iterable[str]) -> list[str]:
    resolved_types: list[AssetType] = []
    for asset_type in asset_types:
        try:
            resolved_types.append(AssetType(asset_type))
        except ValueError:
            continue
    return _symbols_for_asset_types(config, resolved_types)


def _symbols_for_asset_types(config, asset_types: Iterable[AssetType]) -> list[str]:
    symbols: list[str] = []
    for asset_type in asset_types:
        symbols.extend(_SYMBOL_RESOLVERS[asset_type](config))
    return symbols


@task(retries=2, retry_delay_seconds=60, name="download-ohlcv")
async def download_ohlcv(config, manager):
    """Download OHLCV for the configured universe."""

    symbols = _symbols_for_asset_types(config, _OHLCV_ASSET_TYPES)
    return await manager.get(DataType.OHLCV, symbols, start=config.start_date, end=config.end_date)


@task(retries=2, retry_delay_seconds=60, name="download-fundamentals")
async def download_fundamentals(config, manager):
    """Download fundamentals for the stock universe."""

    return await manager.get(DataType.FUNDAMENTALS, config.universe.stock)


@task(retries=2, retry_delay_seconds=60, name="download-futures-ohlcv")
async def download_futures_ohlcv(config, manager):
    """Download futures OHLCV for the configured futures universe."""

    symbols = _symbols_for_asset_types(config, _FUTURES_ASSET_TYPES)
    if not symbols:
        return pl.DataFrame()
    return await manager.get(
        DataType.FUTURES_OHLCV,
        symbols,
        start=config.start_date,
        end=config.end_date,
    )
