"""Kelly-sized portfolio constructor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import (
    BasePortfolioConstructor,
    _apply_legs,
    _check_counts,
    _prepare_returns_matrix,
    _split_long_short,
)
from .equal_weight import long_short_equal_weight_portfolio


class KellyPortfolio(BasePortfolioConstructor):
    """Kelly-sized long/short portfolio (µ/σ² sizing)."""

    def __init__(
        self,
        *,
        price_col: str = "close",
        lookback_days: int = 63,
        kelly_fraction: float = 0.5,
        max_abs_weight: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.price_col = price_col
        self.lookback_days = lookback_days
        self.kelly_fraction = kelly_fraction
        self.max_abs_weight = max_abs_weight

    def compute(
        self, predictions: pd.Series, *, history_df: pd.DataFrame | None = None
    ) -> dict[str, float]:
        long_stocks, short_stocks = _split_long_short(
            predictions,
            self.num_long_positions,
            self.num_short_positions,
            self.long_threshold,
            self.short_threshold,
        )
        if not long_stocks and not short_stocks:
            return {}
        returns = _prepare_returns_matrix(
            history_df, long_stocks + short_stocks, self.price_col, self.lookback_days
        )
        if returns.empty:
            return long_short_equal_weight_portfolio(
                predictions,
                self.num_long_positions,
                self.num_short_positions,
                self.long_threshold,
                self.short_threshold,
            )
        var = returns.var(ddof=0).replace(0, np.nan)
        mu = predictions.copy()

        def _kelly_w(stocks: list[str]) -> pd.Series:
            raw = (
                (mu.reindex(stocks) / var.reindex(stocks))
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if raw.empty:
                return pd.Series(1.0 / len(stocks), index=stocks)
            return (raw / raw.sum()).reindex(stocks).fillna(0.0)

        weights = _apply_legs(
            long_stocks,
            short_stocks,
            self.long_exposure * self.kelly_fraction,
            self.short_exposure * self.kelly_fraction,
            _kelly_w,
        )
        if self.max_abs_weight is not None:
            maw = float(self.max_abs_weight)
            weights = {k: float(np.clip(v, -maw, maw)) for k, v in weights.items()}
        return weights


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def long_short_kelly_portfolio(
    predictions: pd.Series,
    num_long_positions: int = 20,
    num_short_positions: int = 0,
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    history_df: pd.DataFrame | None = None,
    price_col: str = "close",
    lookback_days: int = 63,
    kelly_fraction: float = 0.5,
    max_abs_weight: float | None = None,
    long_exposure: float = 1.0,
    short_exposure: float = 1.0,
) -> dict[str, float]:
    _check_counts(num_long_positions, num_short_positions)
    return KellyPortfolio(
        num_long_positions=num_long_positions,
        num_short_positions=num_short_positions,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        long_exposure=long_exposure,
        short_exposure=short_exposure,
        price_col=price_col,
        lookback_days=lookback_days,
        kelly_fraction=kelly_fraction,
        max_abs_weight=max_abs_weight,
    ).compute(predictions, history_df=history_df)
