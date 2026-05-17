"""Standalone data ingestion flow."""

from __future__ import annotations

from qts.config.builder import Config
from qts.data._schemas import DataType
from qts.orchestration.prefect_compat import flow
from qts.orchestration.runtime import build_data_manager
from qts.orchestration.tasks.data_tasks import requested_symbols


@flow(name="qts-data-fetch", log_prints=True)
async def data_fetch_flow(config_path: str, asset_types: list[str], data_types: list[str]) -> None:
    resolved = Config.build(config_path)
    config = resolved.raw
    manager = build_data_manager(resolved, include_bundle=False)
    symbols = requested_symbols(config, asset_types)
    for data_type in (DataType(value) for value in data_types):
        await manager.get(data_type, symbols, start=config.start_date, end=config.end_date)
