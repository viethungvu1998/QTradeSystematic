"""Shared utilities for factor strategies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .algorithms import (
    train_and_predict_ic_composite,
    train_and_predict_linear_regression,
    train_and_predict_xgb_ranker,
    train_and_predict_xgb_regressor,
)
from .portfolio_construction import (
    long_short_cost_adjusted_portfolio,
    long_short_equal_weight_portfolio,
    long_short_exponential_weight_portfolio,
    long_short_hrp_portfolio,
    long_short_inverse_volatility_portfolio,
    long_short_kelly_portfolio,
    long_short_mean_variance_portfolio,
    long_short_mean_variance_turnover_portfolio,
    long_short_min_variance_portfolio,
    long_short_risk_parity_portfolio,
    long_short_volatility_target_portfolio,
)

DEFAULT_NON_FACTOR_COLUMNS = {
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "signal",
    "weight",
}
FORWARD_RETURN_PREFIX = "forward_return_"


def factor_feature_columns(
    columns: Sequence[str],
    *,
    excluded_columns: set[str] | None = None,
    excluded_prefixes: Sequence[str] = (FORWARD_RETURN_PREFIX,),
) -> list[str]:
    excluded = DEFAULT_NON_FACTOR_COLUMNS | (excluded_columns or set())
    return [
        column
        for column in columns
        if column not in excluded and not any(column.startswith(prefix) for prefix in excluded_prefixes)
    ]


def normalize_signed_weights(weights: Mapping[str, float]) -> dict[str, float]:
    filtered = {symbol: float(weight) for symbol, weight in weights.items() if float(weight) != 0.0}
    if not filtered:
        return {}
    max_abs = max(abs(weight) for weight in filtered.values())
    scale = max(1.0, max_abs)
    return {symbol: weight / scale for symbol, weight in filtered.items()}


__all__ = [
    "FORWARD_RETURN_PREFIX",
    "DEFAULT_NON_FACTOR_COLUMNS",
    "factor_feature_columns",
    "normalize_signed_weights",
    "long_short_cost_adjusted_portfolio",
    "long_short_equal_weight_portfolio",
    "long_short_exponential_weight_portfolio",
    "long_short_hrp_portfolio",
    "long_short_inverse_volatility_portfolio",
    "long_short_kelly_portfolio",
    "long_short_mean_variance_portfolio",
    "long_short_mean_variance_turnover_portfolio",
    "long_short_min_variance_portfolio",
    "long_short_risk_parity_portfolio",
    "long_short_volatility_target_portfolio",
    "train_and_predict_ic_composite",
    "train_and_predict_linear_regression",
    "train_and_predict_xgb_ranker",
    "train_and_predict_xgb_regressor",
]
