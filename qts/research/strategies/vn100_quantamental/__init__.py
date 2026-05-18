"""VN100 quantamental ML factor strategy package.

Class hierarchy
---------------
BaseFactorStrategy (qts.research.strategies.factor.base)
└── VN100QuantamentalStrategy

Pipeline
--------
raw OHLCV
  → build_model_frame  (features.py)
      → screen_liquid_universe
      → add_qsmom_features
      → add_technical_features
      → VNFundamentalFeatures
      → add_factor_scores
      → ForwardReturns
  → walk_forward_ml_signals  (signals.py)
      → XGBoost regressor (rolling window)
      → long_short_equal_weight_portfolio
  → BacktestResult
"""

from . import strategy  # noqa: F401 — triggers @Registry.register_strategy side effect
from .config import (
    TECHNICAL_BASE_COLUMNS,
    MODEL_PARAMS,
    ExperimentConfig,
    FeatureConfig,
    qsmom_column,
)
from .data import (
    fetch_prices_and_fundamentals,
    fundamental_cache_report,
    load_vn100_symbols,
    make_vn_manager,
    normalize_vn_symbol,
)
from .features import (
    add_factor_scores,
    add_qsmom_features,
    add_technical_features,
    build_model_frame,
    feature_coverage_report,
    screen_liquid_universe,
    technical_columns,
)
from .signals import (
    available_predictors,
    choose_predictors,
    default_predictor_candidates,
    effective_long_threshold,
    signals_from_weights,
    walk_forward_ml_signals,
)
from .strategy import VN100QuantamentalStrategy, make_sweep_arms

__all__ = [
    # Strategy class
    "VN100QuantamentalStrategy",
    # Config
    "ExperimentConfig",
    "FeatureConfig",
    "MODEL_PARAMS",
    "TECHNICAL_BASE_COLUMNS",
    "qsmom_column",
    # Data
    "fetch_prices_and_fundamentals",
    "fundamental_cache_report",
    "load_vn100_symbols",
    "make_vn_manager",
    "normalize_vn_symbol",
    # Features
    "add_factor_scores",
    "add_qsmom_features",
    "add_technical_features",
    "build_model_frame",
    "feature_coverage_report",
    "screen_liquid_universe",
    "technical_columns",
    # Signals
    "available_predictors",
    "choose_predictors",
    "default_predictor_candidates",
    "effective_long_threshold",
    "signals_from_weights",
    "walk_forward_ml_signals",
    # Sweeps
    "make_sweep_arms",
]
