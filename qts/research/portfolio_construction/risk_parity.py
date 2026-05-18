"""Risk parity portfolio constructor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import (
    BasePortfolioConstructor,
    _apply_legs,
    _check_counts,
    _prepare_returns_matrix,
    _risk_parity_weights,
    _shrink_covariance,
    _split_long_short,
)
from .volatility import long_short_inverse_volatility_portfolio


class RiskParityPortfolio(BasePortfolioConstructor):
    """Equal risk contribution long/short portfolio."""

    def __init__(
        self,
        *,
        price_col: str = "close",
        lookback_days: int = 63,
        shrinkage: float = 0.1,
        shrinkage_method: str = "diagonal",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.price_col = price_col
        self.lookback_days = lookback_days
        self.shrinkage = shrinkage
        self.shrinkage_method = shrinkage_method

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
            return long_short_inverse_volatility_portfolio(
                predictions,
                self.num_long_positions,
                self.num_short_positions,
                self.long_threshold,
                self.short_threshold,
            )

        def _rp_w(stocks: list[str]) -> np.ndarray:
            cov = _shrink_covariance(
                returns[stocks].dropna(how="all"), self.shrinkage_method, self.shrinkage
            )
            w = _risk_parity_weights(cov)
            return w if w.size > 0 else np.full(len(stocks), 1.0 / len(stocks))

        return _apply_legs(
            long_stocks, short_stocks, self.long_exposure, self.short_exposure, _rp_w
        )


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def long_short_risk_parity_portfolio(
    predictions: pd.Series,
    num_long_positions: int = 20,
    num_short_positions: int = 0,
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    history_df: pd.DataFrame | None = None,
    price_col: str = "close",
    lookback_days: int = 63,
    shrinkage: float = 0.1,
    shrinkage_method: str = "diagonal",
    long_exposure: float = 1.0,
    short_exposure: float = 1.0,
) -> dict[str, float]:
    _check_counts(num_long_positions, num_short_positions)
    return RiskParityPortfolio(
        num_long_positions=num_long_positions,
        num_short_positions=num_short_positions,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        long_exposure=long_exposure,
        short_exposure=short_exposure,
        price_col=price_col,
        lookback_days=lookback_days,
        shrinkage=shrinkage,
        shrinkage_method=shrinkage_method,
    ).compute(predictions, history_df=history_df)
