"""Data-related workflow tasks."""

from __future__ import annotations

import asyncio

from qts.core.errors import DataSourceError


async def _retry(coro_factory, retries: int = 2):
    attempt = 0
    while True:
        try:
            return await coro_factory()
        except DataSourceError:
            attempt += 1
            if attempt > retries:
                raise


async def download_ohlcv(config, manager):
    """Download OHLCV for the configured universe."""

    symbols = [*config.universe.stock, *config.universe.crypto]
    return await _retry(lambda: manager.get_ohlcv(symbols, config.start_date, config.end_date))


async def download_fundamentals(config, manager):
    """Download fundamentals for the stock universe."""

    return await _retry(lambda: manager.get_fundamentals(config.universe.stock))
