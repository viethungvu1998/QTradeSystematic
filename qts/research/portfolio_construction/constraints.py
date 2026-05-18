"""Post-construction constraint adjusters.

These functions accept a weights dict and return a modified weights dict.
They are applied after a constructor produces its initial allocation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .base import _prepare_adv_series, _prepare_returns_matrix

logger = logging.getLogger(__name__)


def apply_weight_constraints(
    weights: dict[str, float],
    max_weight: float | None = None,
    sector_map: dict[str, str] | None = None,
    sector_max_weight: float | None = None,
    target_beta: float | None = None,
    beta_map: dict[str, float] | None = None,
    beta_shrink: float = 0.0,
) -> dict[str, float]:
    """Clip per-symbol and per-sector weight caps; optionally adjust portfolio beta."""
    if not weights:
        return weights
    w = pd.Series(weights, dtype=float)
    if max_weight is not None:
        cap = float(max_weight)
        w = w.clip(lower=-cap, upper=cap)
    if sector_map and sector_max_weight is not None:
        sector_cap = float(sector_max_weight)
        sector_totals = w.groupby(w.index.map(sector_map)).sum()
        for sector, total in sector_totals.items():
            if pd.isna(sector) or total == 0:
                continue
            if abs(total) > sector_cap:
                syms = [s for s in w.index if sector_map.get(s) == sector]
                w.loc[syms] = w.loc[syms] * (sector_cap / abs(total))
    if target_beta is not None and beta_map:
        beta_s = pd.Series(beta_map).reindex(w.index).fillna(0.0)
        current_beta = float((w * beta_s).sum())
        if current_beta != 0:
            if beta_shrink <= 0:
                w = w * (float(target_beta) / current_beta)
            else:
                w = w * (1 + beta_shrink * (float(target_beta) - current_beta) / current_beta)
    return w.to_dict()


def apply_factor_neutrality(
    weights: dict[str, float],
    exposures: pd.DataFrame,
    *,
    target_exposure: dict[str, float] | None = None,
    preserve_gross: bool = True,
    ridge: float = 1e-8,
) -> dict[str, float]:
    """Project weights to remove factor exposure via OLS adjustment."""
    if not weights or exposures is None or exposures.empty:
        return weights
    w_series = pd.Series(weights, dtype=float)
    exp = exposures.reindex(w_series.index).fillna(0.0)
    factors = exp.columns.tolist()
    if not factors:
        return weights
    X = exp.to_numpy()
    target = np.zeros(len(factors))
    if target_exposure:
        target = np.array([float(target_exposure.get(f, 0.0)) for f in factors])
    w = w_series.to_numpy(dtype=float)
    xtx = X.T @ X + np.eye(X.shape[1]) * ridge
    try:
        adj = np.linalg.solve(xtx, X.T @ w - target)
    except np.linalg.LinAlgError:
        adj = np.linalg.pinv(xtx) @ (X.T @ w - target)
    w_adj = pd.Series(w - X @ adj, index=w_series.index, dtype=float)
    if preserve_gross:
        gb, ga = float(np.abs(w_series).sum()), float(np.abs(w_adj).sum())
        if ga > 0 and gb > 0:
            w_adj = w_adj * (gb / ga)
    return w_adj.to_dict()


def apply_volatility_cap(
    weights: dict[str, float],
    history_df: pd.DataFrame | None = None,
    price_col: str = "close",
    lookback_days: int = 63,
    max_position_vol: float | None = None,
    annualize: bool = True,
) -> dict[str, float]:
    """Cap each position weight so its annualized vol contribution stays below threshold."""
    if not weights or history_df is None or max_position_vol is None:
        return weights
    try:
        max_position_vol = float(max_position_vol)
    except (TypeError, ValueError):
        return weights
    if max_position_vol <= 0:
        return weights
    returns = _prepare_returns_matrix(history_df, list(weights.keys()), price_col, lookback_days)
    if returns.empty:
        return weights
    vol = returns.std(ddof=0)
    if annualize:
        vol = vol * np.sqrt(252)
    w = pd.Series(weights, dtype=float)
    capped = []
    for symbol, weight in w.items():
        v = vol.get(symbol)
        if pd.isna(v) or v <= 0:
            capped.append(weight)
            continue
        capped.append(float(np.clip(weight, -max_position_vol / v, max_position_vol / v)))
    return dict(zip(w.index, capped))


def apply_correlation_penalty(
    weights: dict[str, float],
    history_df: pd.DataFrame | None = None,
    price_col: str = "close",
    lookback_days: int = 63,
    penalty_strength: float = 0.5,
) -> dict[str, float]:
    """Scale down weights for symbols that are highly correlated with peers."""
    if not weights or history_df is None or history_df.empty:
        return weights
    returns = _prepare_returns_matrix(history_df, list(weights.keys()), price_col, lookback_days)
    if returns.empty:
        return weights
    corr = returns.corr().fillna(0.0)
    avg_corr = corr.abs().mean().clip(lower=0.0, upper=1.0)
    penalty = (1 - penalty_strength * avg_corr).clip(lower=0.1)
    w = pd.Series(weights, dtype=float)
    return (w * penalty.reindex(w.index).fillna(1.0)).to_dict()


def apply_liquidity_cap(
    weights: dict[str, float],
    history_df: pd.DataFrame | None = None,
    capital_base: float | None = None,
    max_adv_fraction: float = 0.1,
    price_col: str = "close",
    volume_col: str = "volume",
    lookback_days: int = 20,
) -> dict[str, float]:
    """Cap each position at a fraction of its average daily notional volume."""
    if not weights or history_df is None or history_df.empty:
        return weights
    if not capital_base or capital_base <= 0:
        logger.warning("Liquidity cap skipped: missing capital_base.")
        return weights
    adv = _prepare_adv_series(
        history_df, list(weights.keys()), price_col, volume_col, lookback_days
    )
    if adv.empty:
        return weights
    max_by_adv = (adv * float(max_adv_fraction)) / float(capital_base)
    w = pd.Series(weights, dtype=float)
    capped = []
    for symbol, weight in w.items():
        cap = max_by_adv.get(symbol)
        capped.append(
            float(np.clip(weight, -abs(float(cap)), abs(float(cap))))
            if not pd.isna(cap)
            else weight
        )
    return dict(zip(w.index, capped))
