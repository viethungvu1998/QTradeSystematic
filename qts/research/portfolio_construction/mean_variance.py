"""Mean-variance, minimum-variance, and turnover-penalised MV constructors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import (
    BasePortfolioConstructor,
    _adjust_predictions_for_costs,
    _apply_legs,
    _check_counts,
    _prepare_returns_matrix,
    _shrink_covariance,
    _split_long_short,
)
from .equal_weight import long_short_cost_adjusted_portfolio
from .volatility import long_short_inverse_volatility_portfolio


class MeanVariancePortfolio(BasePortfolioConstructor):
    """Mean-variance optimal long/short portfolio."""

    def __init__(
        self,
        *,
        price_col: str = "close",
        lookback_days: int = 63,
        shrinkage: float = 0.1,
        shrinkage_method: str = "diagonal",
        ridge: float = 1e-6,
        risk_aversion: float = 1.0,
        transaction_costs: dict | None = None,
        cost_per_share: float | None = None,
        cost_per_dollar: pd.Series | dict | None = None,
        cost_penalty: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.price_col = price_col
        self.lookback_days = lookback_days
        self.shrinkage = shrinkage
        self.shrinkage_method = shrinkage_method
        self.ridge = ridge
        self.risk_aversion = risk_aversion
        self.transaction_costs = transaction_costs
        self.cost_per_share = cost_per_share
        self.cost_per_dollar = cost_per_dollar
        self.cost_penalty = cost_penalty

    def compute(
        self, predictions: pd.Series, *, history_df: pd.DataFrame | None = None
    ) -> dict[str, float]:
        adjusted = _adjust_predictions_for_costs(
            pd.Series(predictions).copy(),
            history_df,
            self.price_col,
            self.transaction_costs,
            self.cost_per_share,
            self.cost_per_dollar,
            self.cost_penalty,
        )
        long_stocks, short_stocks = _split_long_short(
            adjusted,
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
                adjusted,
                self.num_long_positions,
                self.num_short_positions,
                self.long_threshold,
                self.short_threshold,
                history_df,
                self.price_col,
                self.lookback_days,
                self.long_exposure,
                self.short_exposure,
            )

        def _mv_weights(stocks: list[str]) -> np.ndarray:
            mu = adjusted.reindex(stocks).fillna(0.0).abs()
            cov = _shrink_covariance(
                returns[stocks].dropna(how="all"), self.shrinkage_method, self.shrinkage
            ) + np.eye(len(mu)) * self.ridge
            w = np.linalg.pinv(cov) @ mu.to_numpy(dtype=float)
            if self.risk_aversion > 0:
                w = w / self.risk_aversion
            w = np.clip(np.nan_to_num(w), 0.0, None)
            return w / w.sum() if w.sum() > 0 else np.full(len(stocks), 1.0 / len(stocks))

        return _apply_legs(
            long_stocks, short_stocks, self.long_exposure, self.short_exposure, _mv_weights
        )


class MinVariancePortfolio(BasePortfolioConstructor):
    """Minimum variance long/short portfolio."""

    def __init__(
        self,
        *,
        price_col: str = "close",
        lookback_days: int = 63,
        shrinkage: float = 0.1,
        shrinkage_method: str = "diagonal",
        ridge: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.price_col = price_col
        self.lookback_days = lookback_days
        self.shrinkage = shrinkage
        self.shrinkage_method = shrinkage_method
        self.ridge = ridge

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

        def _minvar(stocks: list[str]) -> np.ndarray:
            sub = returns[stocks].dropna(how="all")
            cov = _shrink_covariance(sub, self.shrinkage_method, self.shrinkage) + np.eye(sub.shape[1]) * self.ridge
            inv_cov = np.linalg.pinv(cov)
            ones = np.ones(cov.shape[0])
            w = inv_cov @ ones
            return w / w.sum() if w.sum() != 0 else np.full(cov.shape[0], 1.0 / cov.shape[0])

        return _apply_legs(
            long_stocks, short_stocks, self.long_exposure, self.short_exposure, _minvar
        )


class MeanVarianceTurnoverPortfolio(BasePortfolioConstructor):
    """Mean-variance portfolio with L2 turnover penalty anchored to prev_weights."""

    def __init__(
        self,
        *,
        price_col: str = "close",
        lookback_days: int = 63,
        shrinkage: float = 0.1,
        shrinkage_method: str = "diagonal",
        ridge: float = 1e-6,
        risk_aversion: float = 1.0,
        prev_weights: dict[str, float] | None = None,
        turnover_penalty: float = 0.0,
        transaction_costs: dict | None = None,
        cost_per_share: float | None = None,
        cost_per_dollar: pd.Series | dict | None = None,
        cost_penalty: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.price_col = price_col
        self.lookback_days = lookback_days
        self.shrinkage = shrinkage
        self.shrinkage_method = shrinkage_method
        self.ridge = ridge
        self.risk_aversion = risk_aversion
        self.prev_weights = prev_weights
        self.turnover_penalty = turnover_penalty
        self.transaction_costs = transaction_costs
        self.cost_per_share = cost_per_share
        self.cost_per_dollar = cost_per_dollar
        self.cost_penalty = cost_penalty

    def compute(
        self, predictions: pd.Series, *, history_df: pd.DataFrame | None = None
    ) -> dict[str, float]:
        adjusted = _adjust_predictions_for_costs(
            pd.Series(predictions).copy(),
            history_df,
            self.price_col,
            self.transaction_costs,
            self.cost_per_share,
            self.cost_per_dollar,
            self.cost_penalty,
        )
        long_stocks, short_stocks = _split_long_short(
            adjusted,
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
            return long_short_cost_adjusted_portfolio(
                adjusted,
                self.num_long_positions,
                self.num_short_positions,
                self.long_threshold,
                self.short_threshold,
                history_df,
                self.price_col,
                self.transaction_costs,
                self.cost_per_share,
                self.cost_per_dollar,
                self.cost_penalty,
                self.long_exposure,
                self.short_exposure,
            )
        tp = max(0.0, float(self.turnover_penalty))
        ra = float(self.risk_aversion) if self.risk_aversion > 0 else 1.0
        prev = pd.Series(self.prev_weights or {}, dtype=float)

        def _mvt_w(stocks: list[str]) -> np.ndarray:
            mu = adjusted.reindex(stocks).fillna(0.0).abs()
            p = prev.reindex(stocks).fillna(0.0).abs()
            sub = returns[stocks].dropna(how="all")
            cov = _shrink_covariance(sub, self.shrinkage_method, self.shrinkage) + np.eye(len(mu)) * self.ridge
            A = ra * cov + 2 * tp * np.eye(cov.shape[0])
            b = mu.to_numpy(dtype=float) + 2 * tp * p.to_numpy(dtype=float)
            try:
                w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                w = np.linalg.pinv(A) @ b
            w = np.clip(np.nan_to_num(w), 0.0, None)
            return w / w.sum() if w.sum() > 0 else np.full(len(stocks), 1.0 / len(stocks))

        return _apply_legs(
            long_stocks, short_stocks, self.long_exposure, self.short_exposure, _mvt_w
        )


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def long_short_mean_variance_portfolio(
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
    ridge: float = 1e-6,
    risk_aversion: float = 1.0,
    transaction_costs: dict | None = None,
    cost_per_share: float | None = None,
    cost_per_dollar: pd.Series | dict | None = None,
    cost_penalty: float = 0.0,
    long_exposure: float = 1.0,
    short_exposure: float = 1.0,
) -> dict[str, float]:
    _check_counts(num_long_positions, num_short_positions)
    return MeanVariancePortfolio(
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
        ridge=ridge,
        risk_aversion=risk_aversion,
        transaction_costs=transaction_costs,
        cost_per_share=cost_per_share,
        cost_per_dollar=cost_per_dollar,
        cost_penalty=cost_penalty,
    ).compute(predictions, history_df=history_df)


def long_short_min_variance_portfolio(
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
    ridge: float = 1e-6,
    long_exposure: float = 1.0,
    short_exposure: float = 1.0,
) -> dict[str, float]:
    _check_counts(num_long_positions, num_short_positions)
    return MinVariancePortfolio(
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
        ridge=ridge,
    ).compute(predictions, history_df=history_df)


def long_short_mean_variance_turnover_portfolio(
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
    ridge: float = 1e-6,
    risk_aversion: float = 1.0,
    prev_weights: dict[str, float] | None = None,
    turnover_penalty: float = 0.0,
    transaction_costs: dict | None = None,
    cost_per_share: float | None = None,
    cost_per_dollar: pd.Series | dict | None = None,
    cost_penalty: float = 0.0,
    long_exposure: float = 1.0,
    short_exposure: float = 1.0,
) -> dict[str, float]:
    _check_counts(num_long_positions, num_short_positions)
    return MeanVarianceTurnoverPortfolio(
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
        ridge=ridge,
        risk_aversion=risk_aversion,
        prev_weights=prev_weights,
        turnover_penalty=turnover_penalty,
        transaction_costs=transaction_costs,
        cost_per_share=cost_per_share,
        cost_per_dollar=cost_per_dollar,
        cost_penalty=cost_penalty,
    ).compute(predictions, history_df=history_df)
