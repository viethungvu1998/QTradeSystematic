"""Yahoo Finance adapter."""

from __future__ import annotations

from datetime import date

import polars as pl

from qts.core.errors import DataSourceError
from qts.core.registry import Registry
from qts.data._schemas import OHLCV_COLUMNS
from qts.data.base import BaseDataSource


@Registry.register_data_source("yahoo")
class YahooDataSource(BaseDataSource):
    """Fixture-friendly Yahoo adapter."""

    def __init__(self, ohlcv_payloads: dict[str, pl.DataFrame] | None = None) -> None:
        self.ohlcv_payloads = ohlcv_payloads or {}

    async def get_ohlcv(
        self,
        symbol: str,
        start: date | None,
        end: date | None,
        interval: str,
    ) -> pl.DataFrame:
        try:
            frame = self.ohlcv_payloads[symbol]
        except KeyError as exc:
            raise DataSourceError("Unknown Yahoo symbol", symbol, (start, end)) from exc
        frame = frame.select(OHLCV_COLUMNS)
        if start is None or end is None:
            return frame
        return frame.filter(pl.col("date").is_between(start, end))

    async def get_fundamentals(self, symbol: str) -> pl.DataFrame:
        raise NotImplementedError("Yahoo fundamentals are not supported.")

    async def stream_ticks(self, symbols: list[str]):  # pragma: no cover - unsupported
        raise NotImplementedError("Yahoo does not support streaming ticks.")
