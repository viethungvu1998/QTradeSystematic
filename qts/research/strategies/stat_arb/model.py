"""Pairs trading strategy."""

from __future__ import annotations

import polars as pl

from qts.core.registry import Registry
from qts.research.strategies.base import BaseStrategy


@Registry.register_strategy("stat_arb")
class StatArbStrategy(BaseStrategy):
    """Simple mean-reversion spread strategy."""

    def __init__(self, entry_zscore: float = 1.0) -> None:
        self.entry_zscore = entry_zscore

    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        symbols = sorted(data["symbol"].unique().to_list())
        if len(symbols) != 2:
            raise ValueError("StatArbStrategy requires exactly two symbols.")
        left = data.filter(pl.col("symbol") == symbols[0]).select("date", pl.col("close").alias("left_close"))
        right = data.filter(pl.col("symbol") == symbols[1]).select("date", pl.col("close").alias("right_close"))
        spread = (
            left.join(right, on="date", how="inner")
            .with_columns((pl.col("left_close") - pl.col("right_close")).alias("spread"))
            .with_columns(
                (
                    (pl.col("spread") - pl.col("spread").rolling_mean(20))
                    / pl.col("spread").rolling_std(20)
                ).alias("spread_zscore")
            )
        )
        rows: list[dict[str, object]] = []
        for record in spread.iter_rows(named=True):
            zscore = record["spread_zscore"]
            if zscore is None:
                signal_left = 0
                signal_right = 0
            elif zscore > self.entry_zscore:
                signal_left = -1
                signal_right = 1
            elif zscore < -self.entry_zscore:
                signal_left = 1
                signal_right = -1
            else:
                signal_left = 0
                signal_right = 0
            rows.extend(
                [
                    {
                        "date": record["date"],
                        "symbol": symbols[0],
                        "signal": signal_left,
                        "weight": 1.0 if signal_left else 0.0,
                    },
                    {
                        "date": record["date"],
                        "symbol": symbols[1],
                        "signal": signal_right,
                        "weight": 1.0 if signal_right else 0.0,
                    },
                ]
            )
        return self.validate_signal_frame(pl.DataFrame(rows))
