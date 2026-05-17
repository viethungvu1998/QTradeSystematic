"""Base strategy contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl


SIGNAL_COLUMNS = ["date", "symbol", "signal", "weight"]
SIGNAL_SCHEMA = {
    "date": pl.Date,
    "symbol": pl.String,
    "signal": pl.Int32,
    "weight": pl.Float64,
}


class BaseStrategy(ABC):
    """Strategy contract."""

    @classmethod
    def empty_signal_frame(cls) -> pl.DataFrame:
        return pl.DataFrame(schema=SIGNAL_SCHEMA)

    @abstractmethod
    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """Produce a standard signal frame."""

    def validate_signal_frame(self, frame: pl.DataFrame) -> pl.DataFrame:
        missing = set(SIGNAL_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing signal columns: {sorted(missing)}")

        normalized = frame.select(SIGNAL_COLUMNS).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("symbol").cast(pl.String),
            pl.col("signal").cast(pl.Int32),
            pl.col("weight").cast(pl.Float64),
        )
        if normalized.filter(pl.col("signal").is_null() | (~pl.col("signal").is_in([-1, 0, 1]))).height:
            raise ValueError("signal must be in {-1, 0, 1}")
        if normalized.filter(
            pl.col("weight").is_null() | (pl.col("weight") < 0) | (pl.col("weight") > 1)
        ).height:
            raise ValueError("weight must be within [0, 1]")
        return normalized
