"""Shared data schemas."""

from __future__ import annotations

from enum import StrEnum

OHLCV_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]


class DataType(StrEnum):
    OHLCV = "ohlcv"
    FUNDAMENTALS = "fundamentals"
    OPTIONS_CHAIN = "options_chain"
    FUNDING_RATES = "funding_rates"
    OPEN_INTEREST = "open_interest"
    FUTURES_OHLCV = "futures_ohlcv"


TIME_COLUMN: dict[DataType, str | None] = {
    DataType.OHLCV: "date",
    DataType.FUTURES_OHLCV: "date",
    DataType.FUNDING_RATES: "funding_time",
    DataType.OPEN_INTEREST: "timestamp",
    DataType.OPTIONS_CHAIN: None,
    DataType.FUNDAMENTALS: None,
}
