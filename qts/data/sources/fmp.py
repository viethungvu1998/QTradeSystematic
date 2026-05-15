"""Financial Modeling Prep adapter."""

from __future__ import annotations

from datetime import date

import polars as pl

from qts.core.errors import DataSourceError
from qts.core.registry import Registry
from qts.data._schemas import OHLCV_COLUMNS
from qts.data.base import BaseDataSource


def _normalize_ohlcv(symbol: str, payload: pl.DataFrame) -> pl.DataFrame:
    if set(OHLCV_COLUMNS).issubset(payload.columns):
        return payload.select(OHLCV_COLUMNS).sort("date")
    renamed = payload.rename(
        {
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    if "symbol" not in renamed.columns:
        renamed = renamed.with_columns(pl.lit(symbol).alias("symbol"))
    return renamed.select(OHLCV_COLUMNS).sort("date")


@Registry.register_data_source("fmp")
class FMPDataSource(BaseDataSource):
    """Fixture-friendly FMP adapter."""

    def __init__(
        self,
        ohlcv_payloads: dict[str, pl.DataFrame] | None = None,
        fundamentals_payloads: dict[str, pl.DataFrame] | None = None,
    ) -> None:
        self.ohlcv_payloads = ohlcv_payloads or {}
        self.fundamentals_payloads = fundamentals_payloads or {}

    async def get_ohlcv(
        self,
        symbol: str,
        start: date | None,
        end: date | None,
        interval: str,
    ) -> pl.DataFrame:
        if symbol not in self.ohlcv_payloads:
            raise DataSourceError("Unknown FMP symbol", symbol, (start, end))
        frame = _normalize_ohlcv(symbol, self.ohlcv_payloads[symbol])
        if start is None or end is None:
            return frame
        return frame.filter(pl.col("date").is_between(start, end))

    async def get_fundamentals(self, symbol: str) -> pl.DataFrame:
        if symbol not in self.fundamentals_payloads:
            raise DataSourceError("Unknown FMP fundamentals symbol", symbol)
        return self.fundamentals_payloads[symbol]

    async def stream_ticks(self, symbols: list[str]):  # pragma: no cover - unsupported
        raise NotImplementedError("FMP does not support streaming ticks.")
