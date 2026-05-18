"""Registered factor strategy factories."""

from __future__ import annotations

from qts.core.registry import Registry

from .algorithms import (
    train_and_predict_ic_composite,
    train_and_predict_xgb_ranker,
    train_and_predict_xgb_regressor,
)
from qts.research.portfolio_construction import (
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

Registry.register_factor_trainer("xgb_regressor")(train_and_predict_xgb_regressor)
Registry.register_factor_trainer("xgb_ranker")(train_and_predict_xgb_ranker)
Registry.register_factor_trainer("ic_composite")(train_and_predict_ic_composite)

Registry.register_portfolio_constructor("equal_weight")(long_short_equal_weight_portfolio)
Registry.register_portfolio_constructor("exponential_weight")(long_short_exponential_weight_portfolio)
Registry.register_portfolio_constructor("inverse_volatility")(long_short_inverse_volatility_portfolio)
Registry.register_portfolio_constructor("mean_variance")(long_short_mean_variance_portfolio)
Registry.register_portfolio_constructor("risk_parity")(long_short_risk_parity_portfolio)
Registry.register_portfolio_constructor("min_variance")(long_short_min_variance_portfolio)
Registry.register_portfolio_constructor("hrp")(long_short_hrp_portfolio)
Registry.register_portfolio_constructor("volatility_target")(long_short_volatility_target_portfolio)
Registry.register_portfolio_constructor("kelly")(long_short_kelly_portfolio)
Registry.register_portfolio_constructor("cost_adjusted")(long_short_cost_adjusted_portfolio)
Registry.register_portfolio_constructor("mean_variance_turnover")(long_short_mean_variance_turnover_portfolio)
