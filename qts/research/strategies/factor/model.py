"""Factor model strategy."""

from __future__ import annotations

import numpy as np
import polars as pl

from qts.core.registry import Registry
from qts.research.strategies.base import BaseStrategy


@Registry.register_strategy("factor")
class FactorStrategy(BaseStrategy):
    """Simple factor ranking strategy."""

    def __init__(self, long_quantile: float = 0.7, short_quantile: float = 0.3) -> None:
        self.long_quantile = long_quantile
        self.short_quantile = short_quantile

    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        feature_columns = [
            column
            for column in data.columns
            if column
            not in {
                "date",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
            }
            and not column.startswith("forward_return_")
        ]
        if not feature_columns:
            frame = data.select("date", "symbol").with_columns(
                pl.lit(0).alias("signal"),
                pl.lit(0.0).alias("weight"),
            )
            return self.validate_signal_frame(frame)

        score_expr = sum(pl.col(column).fill_null(0) for column in feature_columns) / len(feature_columns)
        scored = data.with_columns(score_expr.alias("factor_score"))
        rows: list[dict[str, object]] = []
        for item in scored.partition_by("date", as_dict=False):
            scores = item["factor_score"].fill_null(0).to_numpy()
            long_cutoff = float(np.quantile(scores, self.long_quantile))
            short_cutoff = float(np.quantile(scores, self.short_quantile))
            for record in item.iter_rows(named=True):
                score = float(record["factor_score"] or 0)
                signal = 1 if score >= long_cutoff else -1 if score <= short_cutoff else 0
                weight = min(1.0, abs(score) / (abs(long_cutoff) + 1e-6)) if signal else 0.0
                rows.append(
                    {
                        "date": record["date"],
                        "symbol": record["symbol"],
                        "signal": signal,
                        "weight": float(weight),
                    }
                )
        return self.validate_signal_frame(pl.DataFrame(rows))
