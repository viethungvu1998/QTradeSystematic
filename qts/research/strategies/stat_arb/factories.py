"""Stat-arb spread model and signal rule registrations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qts.core.registry import Registry


@Registry.register_spread_model("ols")
def ols_spread(
    pair_prices: pd.DataFrame,
    symbol_a: str,
    symbol_b: str,
    *,
    hedge_method: str = "ols",
    ratio_min: float | None = None,
    ratio_max: float | None = None,
    allow_negative: bool = True,
    **kwargs,
) -> tuple[pd.Series, pd.Series]:
    """OLS hedge ratio + residual spread."""
    from qts.research.strategies.stat_arb.core import (
        clip_hedge_ratio,
        compute_spread,
        estimate_hedge_ratio,
    )

    hedge_ratio = estimate_hedge_ratio(
        pair_prices[symbol_a],
        pair_prices[symbol_b],
        method=hedge_method,
    )
    if not np.isfinite(hedge_ratio):
        empty = pd.Series(dtype=float)
        return empty, empty
    hedge_ratio = clip_hedge_ratio(hedge_ratio, ratio_min, ratio_max)
    if not allow_negative:
        hedge_ratio = abs(hedge_ratio)
    hedge_series = pd.Series(float(hedge_ratio), index=pair_prices.index)
    spread = compute_spread(pair_prices[symbol_a], pair_prices[symbol_b], hedge_ratio)
    return spread, hedge_series


@Registry.register_spread_model("rolling_ols")
def rolling_ols_spread(
    pair_prices: pd.DataFrame,
    symbol_a: str,
    symbol_b: str,
    *,
    window: int = 60,
    lookback_days: int = 120,
    ewm_span: int | None = None,
    ratio_min: float | None = None,
    ratio_max: float | None = None,
    allow_negative: bool = True,
    **kwargs,
) -> tuple[pd.Series, pd.Series]:
    """Rolling OLS spread."""
    from qts.research.strategies.stat_arb.core import compute_rolling_ols_spread

    return compute_rolling_ols_spread(
        pair_prices,
        symbol_a,
        symbol_b,
        window,
        lookback_days,
        ewm_span,
        ratio_min,
        ratio_max,
        allow_negative,
    )


@Registry.register_signal_rule("zscore_threshold")
def zscore_threshold_rule(
    spread: pd.Series,
    *,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
    zscore_window: int = 22,
    side: str = "long_short",
    stop_z: float | None = None,
    max_holding_bars: int | None = None,
    **kwargs,
):
    """Z-score band entry/exit rule."""
    from qts.research.strategies.stat_arb.core import (
        compute_zscore,
        generate_zscore_signals,
    )

    zscore = compute_zscore(spread, zscore_window).replace([np.inf, -np.inf], np.nan).dropna()
    return generate_zscore_signals(
        zscore,
        entry_z=entry_z,
        exit_z=exit_z,
        side=side,
        stop_z=stop_z,
        max_holding_bars=max_holding_bars,
    )
