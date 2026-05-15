"""Base strategy contract."""

from __future__ import annotations

import polars as pl


SIGNAL_COLUMNS = ["date", "symbol", "signal", "weight"]


class BaseStrategy:
    """Strategy contract."""

    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError

    def validate_signal_frame(self, frame: pl.DataFrame) -> pl.DataFrame:
        missing = set(SIGNAL_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing signal columns: {sorted(missing)}")
        if frame.filter(~pl.col("signal").is_in([-1, 0, 1])).height:
            raise ValueError("signal must be in {-1, 0, 1}")
        if frame.filter((pl.col("weight") < 0) | (pl.col("weight") > 1)).height:
            raise ValueError("weight must be within [0, 1]")
        return frame.select(SIGNAL_COLUMNS)
