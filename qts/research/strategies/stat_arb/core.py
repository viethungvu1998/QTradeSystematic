"""Shared utilities for stat-arb strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pair_selection import (
    PairCandidate,
    compute_adf_pvalue,
    compute_half_life,
    ensure_pair_list,
    estimate_hedge_ratio,
    find_cointegrated_pairs,
    preselect_pairs_by_correlation,
)
from .signals import compute_spread, compute_zscore, generate_zscore_signals
from .universe import stat_arb_universe_screener

_WINDOW_FREQ: dict[str, str] = {"monthly": "MS", "quarterly": "QS", "yearly": "AS"}


def build_windows(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    selection_mode: str,
    lookback_days: int,
    reselection_frequency: str,
    trade_window_days: int | None,
) -> list[dict[str, pd.Timestamp]]:
    """Return {selection_start, selection_end, trade_start, trade_end} dicts."""
    if selection_mode == "rolling":
        first_trade_start = start_ts + pd.Timedelta(days=lookback_days + 1)
        freq = _WINDOW_FREQ.get(reselection_frequency, "QS")
        window_starts = pd.date_range(start=first_trade_start, end=end_ts, freq=freq)
        if window_starts.empty:
            window_starts = pd.DatetimeIndex([first_trade_start])
        windows = []
        for idx, trade_start in enumerate(window_starts):
            selection_end = trade_start - pd.Timedelta(days=1)
            selection_start = selection_end - pd.Timedelta(days=lookback_days)
            if idx + 1 < len(window_starts):
                trade_end = window_starts[idx + 1] - pd.Timedelta(days=1)
            else:
                trade_end = end_ts
            if trade_window_days:
                trade_end = min(
                    trade_end,
                    trade_start + pd.Timedelta(days=int(trade_window_days) - 1),
                )
            windows.append(
                {
                    "selection_start": selection_start,
                    "selection_end": selection_end,
                    "trade_start": trade_start,
                    "trade_end": trade_end,
                }
            )
        return windows

    if selection_mode == "formation":
        selection_start = start_ts
        selection_end = start_ts + pd.Timedelta(days=lookback_days)
        trade_start = selection_end + pd.Timedelta(days=1)
        trade_end = end_ts
    elif selection_mode == "recent":
        selection_end = end_ts
        selection_start = end_ts - pd.Timedelta(days=lookback_days)
        trade_start = start_ts
        trade_end = end_ts
    else:
        raise ValueError(f"Unsupported selection_mode '{selection_mode}'.")
    return [
        {
            "selection_start": selection_start,
            "selection_end": selection_end,
            "trade_start": trade_start,
            "trade_end": trade_end,
        }
    ]


def clip_hedge_ratio(
    ratio: float,
    lo: float | None,
    hi: float | None,
) -> float:
    """Clip ratio to [lo, hi]; None means unbounded on that side."""
    if lo is None and hi is None:
        return ratio
    lower = -float("inf") if lo is None else float(lo)
    upper = float("inf") if hi is None else float(hi)
    return float(np.clip(ratio, lower, upper))


def compute_rolling_ols_spread(
    spread_source: pd.DataFrame,
    sym_a: str,
    sym_b: str,
    hedge_window: int | None,
    lookback_days: int,
    ewm_span: int | None,
    ratio_min: float | None,
    ratio_max: float | None,
    allow_negative: bool,
) -> tuple[pd.Series, pd.Series]:
    """Compute rolling-OLS spread and ratio series."""
    hedge_window_days = int(hedge_window or lookback_days or 60)
    rolling_cov = spread_source[sym_a].rolling(hedge_window_days).cov(spread_source[sym_b])
    rolling_var = spread_source[sym_b].rolling(hedge_window_days).var()
    rolling_ratio = rolling_cov / rolling_var.replace(0, np.nan)
    if ewm_span:
        rolling_ratio = rolling_ratio.ewm(span=int(ewm_span), adjust=False).mean()
    if ratio_min is not None or ratio_max is not None:
        lower = -float("inf") if ratio_min is None else float(ratio_min)
        upper = float("inf") if ratio_max is None else float(ratio_max)
        rolling_ratio = rolling_ratio.clip(lower=lower, upper=upper)
    if not allow_negative:
        rolling_ratio = rolling_ratio.abs()
    spread = spread_source[sym_a] - rolling_ratio * spread_source[sym_b]
    return spread, rolling_ratio


def aggregate_pair_returns(
    pair_returns: list[pd.Series],
    summary_df: pd.DataFrame | None,
    aggregate_method: str,
    target_vol: float | None,
    max_leverage: float | None,
) -> tuple[pd.Series, float | None, pd.Series]:
    """Combine per-pair return series into a weighted portfolio return series."""
    returns_df = pd.concat(pair_returns, axis=1).sort_index()

    if aggregate_method == "equal_weight":
        weights = pd.Series(1.0, index=returns_df.columns)
    elif aggregate_method == "inverse_vol":
        vol = returns_df.std(skipna=True)
        weights = 1.0 / vol.replace(0, np.nan)
    elif aggregate_method == "sharpe_weight":
        vol = returns_df.std(skipna=True)
        sharpe = returns_df.mean(skipna=True) / vol.replace(0, np.nan)
        weights = sharpe.replace([np.inf, -np.inf], np.nan).clip(lower=0.0)
    elif aggregate_method == "half_life_weight":
        if summary_df is None:
            raise ValueError("summary_df required for aggregate_method='half_life_weight'")
        weights = summary_df.set_index("pair_id")["half_life"]
        weights = 1.0 / weights.replace(0, np.nan)
    elif aggregate_method == "pvalue_weight":
        if summary_df is None:
            raise ValueError("summary_df required for aggregate_method='pvalue_weight'")
        weights = summary_df.set_index("pair_id")["pvalue"]
        weights = 1.0 / weights.replace(0, np.nan)
    else:
        raise ValueError(f"Unsupported aggregate_method '{aggregate_method}'.")

    weights = weights.replace([np.inf, -np.inf], np.nan).dropna()
    weights = weights.reindex(returns_df.columns).fillna(0.0)
    if weights.sum() == 0:
        weights = pd.Series(1.0, index=returns_df.columns)
    pair_weights_norm = weights / weights.sum() if weights.sum() else weights

    weight_df = pd.DataFrame(
        np.tile(weights.values, (len(returns_df.index), 1)),
        index=returns_df.index,
        columns=returns_df.columns,
    )
    weight_df = weight_df.where(returns_df.notna())
    weight_sums = weight_df.sum(axis=1)
    aggregate_returns = (returns_df * weight_df).sum(axis=1).div(weight_sums).fillna(0.0)

    aggregate_scale: float | None = None
    if target_vol:
        ann_vol = aggregate_returns.std() * np.sqrt(252)
        if ann_vol and ann_vol > 0:
            aggregate_scale = float(target_vol) / float(ann_vol)
            if max_leverage is not None:
                aggregate_scale = min(aggregate_scale, float(max_leverage))
            aggregate_returns = aggregate_returns * aggregate_scale

    return aggregate_returns, aggregate_scale, pair_weights_norm


__all__ = [
    "PairCandidate",
    "aggregate_pair_returns",
    "build_windows",
    "clip_hedge_ratio",
    "compute_adf_pvalue",
    "compute_half_life",
    "compute_rolling_ols_spread",
    "compute_spread",
    "compute_zscore",
    "ensure_pair_list",
    "estimate_hedge_ratio",
    "find_cointegrated_pairs",
    "generate_zscore_signals",
    "preselect_pairs_by_correlation",
    "stat_arb_universe_screener",
]
