"""Family base for factor strategies."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

import polars as pl

from qts.research.strategies.base import BaseStrategy

from .core import factor_feature_columns, normalize_signed_weights


class BaseFactorStrategy(BaseStrategy):
    """Shared helpers for factor-family strategies."""

    def feature_columns(self, data: pl.DataFrame) -> list[str]:
        return factor_feature_columns(data.columns)

    def signal_frame_from_weights(self, trade_date: date, weights: Mapping[str, float]) -> pl.DataFrame:
        normalized = normalize_signed_weights(weights)
        if not normalized:
            return self.empty_signal_frame()

        rows = []
        for symbol, signed_weight in normalized.items():
            signal = 1 if signed_weight > 0 else -1
            rows.append(
                {
                    "date": trade_date,
                    "symbol": symbol,
                    "signal": signal,
                    "weight": abs(float(signed_weight)),
                }
            )
        return self.validate_signal_frame(pl.DataFrame(rows))
