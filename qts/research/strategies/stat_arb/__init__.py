"""Stat-arb strategies."""

from qts.research.strategies.stat_arb import factories as _stat_arb_factories  # noqa: F401

from .base import BaseStatArbStrategy
from .core import (
    PairCandidate,
    compute_spread,
    compute_zscore,
    estimate_hedge_ratio,
    find_cointegrated_pairs,
    generate_zscore_signals,
    preselect_pairs_by_correlation,
    stat_arb_universe_screener,
)
from .model import StatArbStrategy

__all__ = [
    "BaseStatArbStrategy",
    "StatArbStrategy",
    "PairCandidate",
    "estimate_hedge_ratio",
    "find_cointegrated_pairs",
    "preselect_pairs_by_correlation",
    "compute_spread",
    "compute_zscore",
    "generate_zscore_signals",
    "stat_arb_universe_screener",
]
