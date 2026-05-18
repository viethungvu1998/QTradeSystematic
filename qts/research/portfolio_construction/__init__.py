"""Portfolio construction — class-based and functional APIs.

Class hierarchy
---------------
BasePortfolioConstructor (abstract)
├── EqualWeightPortfolio
├── ExponentialWeightPortfolio
├── CostAdjustedPortfolio
├── InverseVolatilityPortfolio
│   └── VolatilityTargetPortfolio
├── MeanVariancePortfolio
├── MinVariancePortfolio
├── MeanVarianceTurnoverPortfolio
├── RiskParityPortfolio
├── HRPPortfolio
└── KellyPortfolio

Constraint adjusters (apply after construction)
------------------------------------------------
apply_weight_constraints
apply_factor_neutrality
apply_volatility_cap
apply_correlation_penalty
apply_liquidity_cap
"""

from .base import BasePortfolioConstructor
from .constraints import (
    apply_correlation_penalty,
    apply_factor_neutrality,
    apply_liquidity_cap,
    apply_volatility_cap,
    apply_weight_constraints,
)
from .equal_weight import (
    CostAdjustedPortfolio,
    EqualWeightPortfolio,
    ExponentialWeightPortfolio,
    long_short_cost_adjusted_portfolio,
    long_short_equal_weight_portfolio,
    long_short_exponential_weight_portfolio,
)
from .hrp import HRPPortfolio, long_short_hrp_portfolio
from .kelly import KellyPortfolio, long_short_kelly_portfolio
from .mean_variance import (
    MeanVariancePortfolio,
    MeanVarianceTurnoverPortfolio,
    MinVariancePortfolio,
    long_short_mean_variance_portfolio,
    long_short_mean_variance_turnover_portfolio,
    long_short_min_variance_portfolio,
)
from .risk_parity import RiskParityPortfolio, long_short_risk_parity_portfolio
from .volatility import (
    InverseVolatilityPortfolio,
    VolatilityTargetPortfolio,
    long_short_inverse_volatility_portfolio,
    long_short_volatility_target_portfolio,
)

__all__ = [
    # Base
    "BasePortfolioConstructor",
    # Classes
    "CostAdjustedPortfolio",
    "EqualWeightPortfolio",
    "ExponentialWeightPortfolio",
    "HRPPortfolio",
    "InverseVolatilityPortfolio",
    "KellyPortfolio",
    "MeanVariancePortfolio",
    "MeanVarianceTurnoverPortfolio",
    "MinVariancePortfolio",
    "RiskParityPortfolio",
    "VolatilityTargetPortfolio",
    # Functional API
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
    # Constraints
    "apply_correlation_penalty",
    "apply_factor_neutrality",
    "apply_liquidity_cap",
    "apply_volatility_cap",
    "apply_weight_constraints",
]
