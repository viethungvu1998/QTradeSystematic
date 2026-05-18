"""Base class and shared helpers for portfolio construction."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BasePortfolioConstructor(ABC):
    """Contract for all portfolio constructors.

    Subclasses receive the shared long/short selection params via __init__
    and implement the weighting logic in ``compute``.
    """

    def __init__(
        self,
        *,
        num_long_positions: int = 20,
        num_short_positions: int = 0,
        long_threshold: float | None = None,
        short_threshold: float | None = None,
        long_exposure: float = 1.0,
        short_exposure: float = 1.0,
    ) -> None:
        _check_counts(num_long_positions, num_short_positions)
        self.num_long_positions = num_long_positions
        self.num_short_positions = num_short_positions
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.long_exposure = long_exposure
        self.short_exposure = short_exposure

    @abstractmethod
    def compute(
        self,
        predictions: pd.Series,
        *,
        history_df: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Return symbol → weight mapping from ranked ``predictions``."""

    def __call__(
        self,
        predictions: pd.Series,
        *,
        history_df: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        return self.compute(predictions, history_df=history_df)


# ---------------------------------------------------------------------------
# Shared selection helpers
# ---------------------------------------------------------------------------


def _check_counts(num_long: int, num_short: int) -> None:
    if num_long < 0 or num_short < 0:
        raise ValueError("num_long_positions and num_short_positions must be non-negative")


def _split_long_short(
    predictions: pd.Series,
    num_long: int,
    num_short: int,
    long_threshold: float | None,
    short_threshold: float | None,
) -> tuple[list[str], list[str]]:
    sorted_preds = predictions.sort_values(ascending=False)

    long_pool = (
        sorted_preds[sorted_preds > long_threshold]
        if long_threshold is not None
        else sorted_preds[sorted_preds > 0]
    )
    long_stocks = list(long_pool.index[:num_long])

    short_stocks: list[str] = []
    if num_short > 0:
        short_pool = (
            sorted_preds[sorted_preds < short_threshold]
            if short_threshold is not None
            else sorted_preds[sorted_preds < 0]
        )
        short_stocks = list(short_pool.index[-num_short:])

    return long_stocks, short_stocks


def _apply_legs(
    long_stocks: list[str],
    short_stocks: list[str],
    long_exposure: float,
    short_exposure: float,
    weight_fn,
) -> dict[str, float]:
    weights: dict[str, float] = {}
    for stocks, exposure, sign in [
        (long_stocks, long_exposure, 1),
        (short_stocks, short_exposure, -1),
    ]:
        if not stocks:
            continue
        w = weight_fn(stocks)
        if isinstance(w, pd.Series):
            weights.update((w * sign * float(exposure)).to_dict())
        else:
            weights.update(dict(zip(stocks, w * sign * float(exposure), strict=False)))
    return weights


# ---------------------------------------------------------------------------
# Shared covariance / returns helpers
# ---------------------------------------------------------------------------


def _prepare_returns_matrix(
    history_df: pd.DataFrame,
    symbols: list[str],
    price_col: str = "close",
    lookback_days: int = 63,
) -> pd.DataFrame:
    if history_df is None or history_df.empty:
        return pd.DataFrame()
    required = {"date", "symbol", price_col}
    if not required.issubset(history_df.columns):
        return pd.DataFrame()
    hist = history_df.loc[
        history_df["symbol"].isin(symbols), ["date", "symbol", price_col]
    ].copy()
    if hist.empty:
        return pd.DataFrame()
    hist["date"] = pd.to_datetime(hist["date"])
    hist.sort_values(["symbol", "date"], inplace=True)
    hist["ret"] = hist.groupby("symbol")[price_col].pct_change()
    ret_matrix = (
        hist.pivot(index="date", columns="symbol", values="ret")
        .dropna(how="all")
        .sort_index()
    )
    if lookback_days and lookback_days > 0:
        ret_matrix = ret_matrix.tail(int(lookback_days))
    return ret_matrix


def _prepare_adv_series(
    history_df: pd.DataFrame,
    symbols: list[str],
    price_col: str = "close",
    volume_col: str = "volume",
    lookback_days: int = 20,
) -> pd.Series:
    if history_df is None or history_df.empty:
        return pd.Series(dtype=float)
    required = {"date", "symbol", price_col, volume_col}
    if not required.issubset(history_df.columns):
        return pd.Series(dtype=float)
    hist = history_df.loc[
        history_df["symbol"].isin(symbols), ["date", "symbol", price_col, volume_col]
    ].copy()
    if hist.empty:
        return pd.Series(dtype=float)
    hist["date"] = pd.to_datetime(hist["date"])
    hist.sort_values(["symbol", "date"], inplace=True)
    if lookback_days and lookback_days > 0:
        hist = hist.groupby("symbol", group_keys=False).tail(int(lookback_days))
    adv = (hist[price_col] * hist[volume_col]).groupby(hist["symbol"]).mean()
    return adv


def _shrink_covariance(
    returns: pd.DataFrame,
    method: str = "diagonal",
    shrinkage: float = 0.1,
) -> np.ndarray:
    if returns is None or returns.empty:
        return np.empty((0, 0))
    cov = returns.cov().to_numpy()
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    method = str(method or "diagonal").lower()
    if method in {"none", "sample"}:
        return cov
    if method in {"ledoit_wolf", "ledoit-wolf", "lw"}:
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(returns.fillna(0.0).to_numpy())
            return np.nan_to_num(lw.covariance_, nan=0.0, posinf=0.0, neginf=0.0)
        except ImportError:
            logger.warning("scikit-learn unavailable for Ledoit-Wolf; falling back to diagonal shrinkage.")
    shrink = float(np.clip(shrinkage, 0.0, 1.0))
    diag = np.diag(np.diag(cov))
    return (1 - shrink) * cov + shrink * diag


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    if cov.size == 0:
        return cov
    std = np.sqrt(np.diag(cov))
    std = np.where(std == 0, np.nan, std)
    corr = cov / np.outer(std, std)
    return np.clip(np.nan_to_num(corr, nan=0.0), -1.0, 1.0)


def _risk_parity_weights(cov: np.ndarray, max_iter: int = 200, tol: float = 1e-6) -> np.ndarray:
    n = cov.shape[0]
    if n == 0:
        return np.array([])
    w = np.full(n, 1.0 / n)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    for _ in range(max_iter):
        port_var = w @ cov @ w
        if port_var <= 0:
            break
        mrc = cov @ w
        rc = w * mrc
        target = port_var / n
        if np.all(np.abs(rc - target) < tol):
            break
        adj = target / np.where(rc == 0, np.nan, rc)
        adj = np.nan_to_num(adj, nan=1.0, posinf=1.0, neginf=1.0)
        w = np.clip(w * adj, 1e-8, None)
        w = w / w.sum()
    return w


def _get_quasi_diag(link: np.ndarray) -> list[int]:
    if link.size == 0:
        return []
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = int(link[-1, 3])
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix.loc[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1]).sort_index().reset_index(drop=True)
    return sort_ix.tolist()


def _hrp_allocation(cov: np.ndarray, ordered_items: list[int]) -> np.ndarray:
    if not ordered_items:
        return np.array([])
    weights = pd.Series(1.0, index=ordered_items, dtype=float)
    clusters: list[list[int]] = [ordered_items]
    while clusters:
        clusters = [c for c in clusters if len(c) > 1]
        if not clusters:
            break
        new_clusters: list[list[int]] = []
        for cluster in clusters:
            split = len(cluster) // 2
            left, right = cluster[:split], cluster[split:]

            def _cluster_var(items: list[int]) -> float:
                sl = cov[np.ix_(items, items)]
                d = np.diag(sl)
                inv = np.where(d > 0, 1.0 / d, 0.0)
                w = inv / inv.sum() if inv.sum() != 0 else np.full(len(items), 1.0 / len(items))
                return float(w @ sl @ w)

            var_l, var_r = _cluster_var(left), _cluster_var(right)
            alpha = 1 - var_l / (var_l + var_r) if (var_l + var_r) != 0 else 0.5
            weights.loc[left] *= alpha
            weights.loc[right] *= 1 - alpha
            new_clusters.extend([left, right])
        clusters = new_clusters
    return weights.values


# ---------------------------------------------------------------------------
# Transaction cost helpers
# ---------------------------------------------------------------------------


def _estimate_cost_per_dollar(
    history_df: pd.DataFrame | None,
    symbols: list[str],
    price_col: str = "close",
    transaction_costs: dict | None = None,
    cost_per_share: float | None = None,
) -> pd.Series:
    if cost_per_share is None:
        tc = transaction_costs or {}
        commission_per_share = float((tc.get("commission") or {}).get("cost") or 0.0)
        spread = float((tc.get("slippage") or {}).get("spread") or 0.0)
        cost_per_share = commission_per_share + spread / 2.0

    if cost_per_share <= 0:
        return pd.Series(0.0, index=symbols, dtype=float)

    if history_df is None or history_df.empty:
        return pd.Series(dtype=float)

    hist = history_df.loc[
        history_df["symbol"].isin(symbols), ["symbol", "date", price_col]
    ].copy()
    if hist.empty:
        return pd.Series(dtype=float)
    hist["date"] = pd.to_datetime(hist["date"])
    hist.sort_values(["symbol", "date"], inplace=True)
    latest = hist.groupby("symbol", group_keys=False).tail(1)
    prices = latest.set_index("symbol")[price_col].replace(0, np.nan)
    return (cost_per_share / prices).replace([np.inf, -np.inf], np.nan).fillna(0.0).reindex(symbols).fillna(0.0)


def _adjust_predictions_for_costs(
    predictions: pd.Series,
    history_df: pd.DataFrame | None = None,
    price_col: str = "close",
    transaction_costs: dict | None = None,
    cost_per_share: float | None = None,
    cost_per_dollar: pd.Series | dict | None = None,
    cost_penalty: float = 1.0,
) -> pd.Series:
    if predictions.empty or float(cost_penalty) <= 0:
        return predictions
    symbols = list(predictions.index)
    if cost_per_dollar is None:
        cost_series = _estimate_cost_per_dollar(
            history_df, symbols, price_col, transaction_costs, cost_per_share
        )
    else:
        cost_series = pd.Series(cost_per_dollar, dtype=float).reindex(symbols).fillna(0.0)
    if cost_series.empty:
        return predictions
    pred = pd.to_numeric(predictions, errors="coerce").fillna(0.0)
    adjusted_mag = (pred.abs() - float(cost_penalty) * cost_series).clip(lower=0.0)
    return np.sign(pred) * adjusted_mag
