"""Data source abstractions."""

from __future__ import annotations

from datetime import date
from typing import AsyncIterator

import polars as pl

from qts.core.events import Tick
from qts.data._schemas import DataType


class BaseDataSource:
    """Contract for data providers."""

    CAPABILITIES: frozenset[DataType] = frozenset()

    def expand_symbols(self, data_type: DataType, symbol: str, **kwargs) -> list[str]:
        """Resolve a request symbol into one or more concrete fetch symbols."""
        return [symbol]

    async def fetch(self, data_type: DataType, symbol: str, **kwargs) -> pl.DataFrame:
        raise NotImplementedError

    async def get_ohlcv(
        self,
        symbol: str,
        start: date | None,
        end: date | None,
        interval: str,
    ) -> pl.DataFrame:
        raise NotImplementedError

    async def get_fundamentals(self, symbol: str) -> pl.DataFrame:
        raise NotImplementedError

    async def stream_ticks(self, symbols: list[str]) -> AsyncIterator[Tick]:
        raise NotImplementedError
