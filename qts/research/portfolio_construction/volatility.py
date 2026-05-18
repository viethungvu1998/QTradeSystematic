"""Inverse-volatility and volatility-target portfolio constructors."""

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


class InverseVolatilityPortfolio(BasePortfolioConstructor):
    """Inverse-volatility weighted long/short portfolio."""

    def __init__(
        self,
        *,
        price_col: str = "close",
        lookback_days: int = 63,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.price_col = price_col
        self.lookback_days = lookback_days

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
        vol = returns.std(ddof=0).replace(0, np.nan)

        def _inv_vol_w(stocks: list[str]) -> pd.Series:
            inv = (
                (1 / vol.reindex(stocks)).replace([np.inf, -np.inf], np.nan).dropna()
            )
            if inv.empty:
                return pd.Series(1.0 / len(stocks), index=stocks)
            return (inv / inv.sum()).reindex(stocks).fillna(0.0)

        return _apply_legs(
            long_stocks, short_stocks, self.long_exposure, self.short_exposure, _inv_vol_w
        )


class VolatilityTargetPortfolio(InverseVolatilityPortfolio):
    """Inverse-vol portfolio scaled to a target annualized volatility."""

    def __init__(
        self,
        *,
        target_vol: float = 0.15,
        max_scale: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.target_vol = target_vol
        self.max_scale = max_scale

    def compute(
        self, predictions: pd.Series, *, history_df: pd.DataFrame | None = None
    ) -> dict[str, float]:
        weights = super().compute(predictions, history_df=history_df)
        if not weights or history_df is None or history_df.empty:
            return weights
        returns = _prepare_returns_matrix(
            history_df, list(weights.keys()), self.price_col, self.lookback_days
        )
        if returns.empty:
            return weights
        w_series = pd.Series(weights).reindex(returns.columns).fillna(0.0)
        realized_vol = returns.mul(w_series, axis=1).sum(axis=1).std(ddof=0) * np.sqrt(252)
        if realized_vol == 0 or np.isnan(realized_vol):
            return weights
        scale = float(self.target_vol) / realized_vol
        if self.max_scale is not None:
            scale = min(scale, float(self.max_scale))
        if scale <= 0:
            return weights
        return (w_series * scale).to_dict()


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def long_short_inverse_volatility_portfolio(
    predictions: pd.Series,
    num_long_positions: int = 20,
    num_short_positions: int = 0,
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    history_df: pd.DataFrame | None = None,
    price_col: str = "close",
    lookback_days: int = 63,
    long_exposure: float = 1.0,
    short_exposure: float = 1.0,
) -> dict[str, float]:
    _check_counts(num_long_positions, num_short_positions)
    return InverseVolatilityPortfolio(
        num_long_positions=num_long_positions,
        num_short_positions=num_short_positions,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        long_exposure=long_exposure,
        short_exposure=short_exposure,
        price_col=price_col,
        lookback_days=lookback_days,
    ).compute(predictions, history_df=history_df)


def long_short_volatility_target_portfolio(
    predictions: pd.Series,
    num_long_positions: int = 20,
    num_short_positions: int = 0,
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    history_df: pd.DataFrame | None = None,
    price_col: str = "close",
    lookback_days: int = 63,
    target_vol: float = 0.15,
    max_scale: float | None = None,
    long_exposure: float = 1.0,
    short_exposure: float = 1.0,
) -> dict[str, float]:
    _check_counts(num_long_positions, num_short_positions)
    return VolatilityTargetPortfolio(
        num_long_positions=num_long_positions,
        num_short_positions=num_short_positions,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        long_exposure=long_exposure,
        short_exposure=short_exposure,
        price_col=price_col,
        lookback_days=lookback_days,
        target_vol=target_vol,
        max_scale=max_scale,
    ).compute(predictions, history_df=history_df)
