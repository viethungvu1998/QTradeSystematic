"""Data-related workflow tasks."""

from __future__ import annotations

import polars as pl

from qts.data._schemas import DataType
from qts.orchestration.prefect_compat import task


@task(retries=2, retry_delay_seconds=60, name="download-ohlcv")
async def download_ohlcv(config, manager):
    """Download OHLCV for the configured universe."""

    symbols = [*config.universe.stock, *config.universe.vn_stock, *config.universe.crypto]
    return await manager.get(DataType.OHLCV, symbols, start=config.start_date, end=config.end_date)


@task(retries=2, retry_delay_seconds=60, name="download-fundamentals")
async def download_fundamentals(config, manager):
    """Download fundamentals for the stock universe."""

    return await manager.get(DataType.FUNDAMENTALS, config.universe.stock)


@task(retries=2, retry_delay_seconds=60, name="download-futures-ohlcv")
async def download_futures_ohlcv(config, manager):
    """Download futures OHLCV for the configured futures universe."""

    symbols = config.universe.crypto_futures
    if not symbols:
        return pl.DataFrame()
    return await manager.get(DataType.FUTURES_OHLCV, symbols, start=config.start_date, end=config.end_date)
