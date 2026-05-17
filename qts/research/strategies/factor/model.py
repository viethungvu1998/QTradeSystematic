"""Factor model strategy."""

from __future__ import annotations

import numpy as np
import polars as pl

from qts.core.registry import Registry

from .base import BaseFactorStrategy


@Registry.register_strategy("factor")
class FactorStrategy(BaseFactorStrategy):
    """Simple factor ranking strategy."""

    def __init__(self, long_quantile: float = 0.7, short_quantile: float = 0.3) -> None:
        self.long_quantile = long_quantile
        self.short_quantile = short_quantile

    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        if data.is_empty():
            return self.empty_signal_frame()

        feature_columns = self.feature_columns(data)
        if not feature_columns:
            frame = data.select("date", "symbol").with_columns(
                pl.lit(0).alias("signal"),
                pl.lit(0.0).alias("weight"),
            )
            return self.validate_signal_frame(frame)

        zscore_columns = [f"_{column}_zscore" for column in feature_columns]
        scored = data.with_columns(
            [
                (
                    (pl.col(column) - pl.col(column).mean().over("date"))
                    / (pl.col(column).std().over("date") + 1e-8)
                )
                .fill_null(0.0)
                .alias(alias)
                for column, alias in zip(feature_columns, zscore_columns, strict=True)
            ]
        ).with_columns(
            (sum(pl.col(column) for column in zscore_columns) / len(zscore_columns)).alias("factor_score")
        )
        rows: list[dict[str, object]] = []
        for item in scored.partition_by("date", as_dict=False):
            scores = item["factor_score"].fill_null(0).to_numpy()
            long_cutoff = float(np.quantile(scores, self.long_quantile))
            short_cutoff = float(np.quantile(scores, self.short_quantile))
            max_abs_score = float(np.abs(scores).max()) if scores.size else 0.0
            for record in item.iter_rows(named=True):
                score = float(record["factor_score"] or 0)
                signal = 1 if score >= long_cutoff else -1 if score <= short_cutoff else 0
                weight = min(1.0, abs(score) / (max_abs_score + 1e-8)) if signal else 0.0
                rows.append(
                    {
                        "date": record["date"],
                        "symbol": record["symbol"],
                        "signal": signal,
                        "weight": float(weight),
                    }
                )
        if not rows:
            return self.empty_signal_frame()
        return self.validate_signal_frame(pl.DataFrame(rows))
