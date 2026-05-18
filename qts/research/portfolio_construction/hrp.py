"""Hierarchical risk parity portfolio constructor."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .base import (
    BasePortfolioConstructor,
    _apply_legs,
    _check_counts,
    _cov_to_corr,
    _get_quasi_diag,
    _hrp_allocation,
    _prepare_returns_matrix,
    _risk_parity_weights,
    _shrink_covariance,
    _split_long_short,
)
from .volatility import long_short_inverse_volatility_portfolio

logger = logging.getLogger(__name__)


class HRPPortfolio(BasePortfolioConstructor):
    """Hierarchical risk parity long/short portfolio.

    Falls back to risk parity when scipy is unavailable.
    """

    def __init__(
        self,
        *,
        price_col: str = "close",
        lookback_days: int = 63,
        shrinkage: float = 0.1,
        shrinkage_method: str = "diagonal",
        linkage_method: str = "single",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.price_col = price_col
        self.lookback_days = lookback_days
        self.shrinkage = shrinkage
        self.shrinkage_method = shrinkage_method
        self.linkage_method = linkage_method

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

        def _hrp_w(stocks: list[str]) -> np.ndarray:
            sub = returns[stocks].dropna(how="all")
            cov = _shrink_covariance(sub, self.shrinkage_method, self.shrinkage)
            if cov.size == 0:
                return np.full(len(stocks), 1.0 / len(stocks))
            corr = _cov_to_corr(cov)
            try:
                from scipy.cluster.hierarchy import linkage
                from scipy.spatial.distance import squareform
            except ImportError:
                logger.warning("HRP requires scipy; falling back to risk parity.")
                w = _risk_parity_weights(cov)
                return w if w.size > 0 else np.full(len(stocks), 1.0 / len(stocks))
            dist = np.clip(np.sqrt(0.5 * (1 - corr)), 0.0, None)
            link = linkage(squareform(dist, checks=False), method=self.linkage_method)
            sort_ix = _get_quasi_diag(link)
            if sort_ix:
                w = _hrp_allocation(cov, sort_ix)
                if w.size > 0:
                    return (
                        pd.Series(w, index=sort_ix)
                        .reindex(range(cov.shape[0]))
                        .fillna(0.0)
                        .to_numpy()
                    )
            w = _risk_parity_weights(cov)
            return w if w.size > 0 else np.full(len(stocks), 1.0 / len(stocks))

        return _apply_legs(
            long_stocks, short_stocks, self.long_exposure, self.short_exposure, _hrp_w
        )


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def long_short_hrp_portfolio(
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
    linkage_method: str = "single",
    long_exposure: float = 1.0,
    short_exposure: float = 1.0,
) -> dict[str, float]:
    _check_counts(num_long_positions, num_short_positions)
    return HRPPortfolio(
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
        linkage_method=linkage_method,
    ).compute(predictions, history_df=history_df)
