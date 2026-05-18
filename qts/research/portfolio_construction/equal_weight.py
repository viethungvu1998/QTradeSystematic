"""Equal-weight, exponential-weight, and cost-adjusted portfolio constructors."""

from __future__ import annotations

import pandas as pd

from .base import (
    BasePortfolioConstructor,
    _adjust_predictions_for_costs,
    _apply_legs,
    _check_counts,
    _split_long_short,
)


class EqualWeightPortfolio(BasePortfolioConstructor):
    """Equal-weight long/short portfolio from ranked predictions."""

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
        return _apply_legs(
            long_stocks,
            short_stocks,
            self.long_exposure,
            self.short_exposure,
            lambda stocks: pd.Series(1.0 / len(stocks), index=stocks),
        )


class ExponentialWeightPortfolio(BasePortfolioConstructor):
    """Exponential-decay ranked long/short portfolio."""

    def __init__(self, *, decay: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        if not 0 < decay < 1:
            raise ValueError("decay must be between 0 and 1 (exclusive)")
        self.decay = decay

    def compute(
        self, predictions: pd.Series, *, history_df: pd.DataFrame | None = None
    ) -> dict[str, float]:
        long_pool = (
            predictions[predictions > self.long_threshold]
            if self.long_threshold is not None
            else predictions[predictions > 0]
        )
        short_pool = (
            predictions[predictions < self.short_threshold]
            if self.short_threshold is not None
            else predictions[predictions < 0]
        )
        long_stocks = list(
            long_pool.sort_values(ascending=False).index[: self.num_long_positions]
        )
        short_stocks = list(
            short_pool.drop(long_stocks, errors="ignore")
            .sort_values()
            .index[: self.num_short_positions]
        )
        weights: dict[str, float] = {}
        if long_stocks:
            raw = [(1 - self.decay) * self.decay**i for i in range(len(long_stocks))]
            total = sum(raw)
            scale = 0.5 / total if total else 0
            weights.update({s: w * scale for s, w in zip(long_stocks, raw)})
        if short_stocks:
            raw = [-(1 - self.decay) * self.decay**i for i in range(len(short_stocks))]
            total = sum(raw)
            scale = -0.5 / total if total else 0
            weights.update({s: w * scale for s, w in zip(short_stocks, raw)})
        return weights


class CostAdjustedPortfolio(BasePortfolioConstructor):
    """Equal-weight portfolio after adjusting signal magnitudes for transaction costs."""

    def __init__(
        self,
        *,
        price_col: str = "close",
        transaction_costs: dict | None = None,
        cost_per_share: float | None = None,
        cost_per_dollar: pd.Series | dict | None = None,
        cost_penalty: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.price_col = price_col
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
        return _apply_legs(
            long_stocks,
            short_stocks,
            self.long_exposure,
            self.short_exposure,
            lambda stocks: pd.Series(1.0 / len(stocks), index=stocks),
        )


# ---------------------------------------------------------------------------
# Functional API (backward-compatible wrappers)
# ---------------------------------------------------------------------------


def long_short_equal_weight_portfolio(
    predictions: pd.Series,
    num_long_positions: int = 20,
    num_short_positions: int = 0,
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    history_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    _check_counts(num_long_positions, num_short_positions)
    return EqualWeightPortfolio(
        num_long_positions=num_long_positions,
        num_short_positions=num_short_positions,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
    ).compute(predictions, history_df=history_df)


def long_short_exponential_weight_portfolio(
    predictions: pd.Series,
    num_long_positions: int = 20,
    num_short_positions: int = 0,
    decay: float = 0.5,
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    history_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    _check_counts(num_long_positions, num_short_positions)
    return ExponentialWeightPortfolio(
        decay=decay,
        num_long_positions=num_long_positions,
        num_short_positions=num_short_positions,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
    ).compute(predictions, history_df=history_df)


def long_short_cost_adjusted_portfolio(
    predictions: pd.Series,
    num_long_positions: int = 20,
    num_short_positions: int = 0,
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    history_df: pd.DataFrame | None = None,
    price_col: str = "close",
    transaction_costs: dict | None = None,
    cost_per_share: float | None = None,
    cost_per_dollar: pd.Series | dict | None = None,
    cost_penalty: float = 1.0,
    long_exposure: float = 1.0,
    short_exposure: float = 1.0,
) -> dict[str, float]:
    _check_counts(num_long_positions, num_short_positions)
    return CostAdjustedPortfolio(
        num_long_positions=num_long_positions,
        num_short_positions=num_short_positions,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        long_exposure=long_exposure,
        short_exposure=short_exposure,
        price_col=price_col,
        transaction_costs=transaction_costs,
        cost_per_share=cost_per_share,
        cost_per_dollar=cost_per_dollar,
        cost_penalty=cost_penalty,
    ).compute(predictions, history_df=history_df)
