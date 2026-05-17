"""Standalone data ingestion flow."""

from __future__ import annotations

from qts.config.builder import Config
from qts.data._schemas import DataType
from qts.data.manager import DataManager
from qts.orchestration.prefect_compat import flow


@flow(name="qts-data-fetch", log_prints=True)
async def data_fetch_flow(config_path: str, asset_types: list[str], data_types: list[str]) -> None:
    resolved = Config.build(config_path)
    config = resolved.raw
    manager = DataManager(
        stock_source=resolved.stock_source,
        vn_stock_source=resolved.vn_stock_source,
        crypto_source=resolved.crypto_source,
        crypto_futures_source=resolved.crypto_futures_source,
        storage=resolved.storage,
        cache=resolved.cache,
    )
    universe = {
        "stock": config.universe.stock,
        "vn_stock": config.universe.vn_stock,
        "crypto": config.universe.crypto,
        "crypto_futures": config.universe.crypto_futures,
    }
    symbols = [symbol for asset_type in asset_types for symbol in universe.get(asset_type, [])]
    for data_type in (DataType(value) for value in data_types):
        await manager.get(data_type, symbols, start=config.start_date, end=config.end_date)
