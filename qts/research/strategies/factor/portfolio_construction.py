"""Portfolio construction functions for factor strategies.

All functions accept predictions as a pandas Series (symbol → score) and return
a dict[str, float] (symbol → weight). History data for covariance/volatility
estimation must be provided as a pandas DataFrame with columns [date, symbol, close].
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_counts(num_long: int, num_short: int) -> None:
    if num_long < 0 or num_short < 0:
        raise ValueError("num_long_positions and num_short_positions must be non-negative")


def _apply_legs(
    long_stocks: list[str],
    short_stocks: list[str],
    long_exposure: float,
    short_exposure: float,
    weight_fn,
) -> dict[str, float]:
    """Apply weight_fn to each leg and combine into a single weights dict."""
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
    hist = history_df.loc[history_df["symbol"].isin(symbols), ["date", "symbol", price_col]].copy()
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


def _split_long_short(
    predictions: pd.Series,
    num_long: int,
    num_short: int,
    long_threshold: float | None,
    short_threshold: float | None,
) -> tuple[list[str], list[str]]:
    sorted_preds = predictions.sort_values(ascending=False)
    pos_mask = sorted_preds > 0

    long_pool = sorted_preds[sorted_preds > long_threshold] if long_threshold is not None else sorted_preds[pos_mask]
    long_stocks = list(long_pool.index[:num_long])

    short_stocks: list[str] = []
    if num_short > 0:
        neg_mask = sorted_preds < 0
        short_pool = sorted_preds[sorted_preds < short_threshold] if short_threshold is not None else sorted_preds[neg_mask]
        short_stocks = list(short_pool.index[-num_short:])

    return long_stocks, short_stocks


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

    hist = history_df.loc[history_df["symbol"].isin(symbols), ["symbol", "date", price_col]].copy()
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
        cost_series = _estimate_cost_per_dollar(history_df, symbols, price_col, transaction_costs, cost_per_share)
    else:
        cost_series = pd.Series(cost_per_dollar, dtype=float).reindex(symbols).fillna(0.0)
    if cost_series.empty:
        return predictions
    pred = pd.to_numeric(predictions, errors="coerce").fillna(0.0)
    adjusted_mag = (pred.abs() - float(cost_penalty) * cost_series).clip(lower=0.0)
    return np.sign(pred) * adjusted_mag


# ---------------------------------------------------------------------------
# Constraint helpers — called post-construction to adjust a weights dict
# ---------------------------------------------------------------------------


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
    adv = _prepare_adv_series(history_df, list(weights.keys()), price_col, volume_col, lookback_days)
    if adv.empty:
        return weights
    max_by_adv = (adv * float(max_adv_fraction)) / float(capital_base)
    w = pd.Series(weights, dtype=float)
    capped = []
    for symbol, weight in w.items():
        cap = max_by_adv.get(symbol)
        capped.append(float(np.clip(weight, -abs(float(cap)), abs(float(cap)))) if not pd.isna(cap) else weight)
    return dict(zip(w.index, capped))


# ---------------------------------------------------------------------------
# Portfolio construction functions
# ---------------------------------------------------------------------------


def long_short_equal_weight_portfolio(
    predictions: pd.Series,
    num_long_positions: int = 20,
    num_short_positions: int = 0,
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    history_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Equal-weight long/short portfolio from ranked predictions."""
    _check_counts(num_long_positions, num_short_positions)
    long_stocks, short_stocks = _split_long_short(
        predictions, num_long_positions, num_short_positions, long_threshold, short_threshold
    )
    return _apply_legs(
        long_stocks, short_stocks, 1.0, 1.0,
        lambda stocks: pd.Series(1.0 / len(stocks), index=stocks),
    )


def long_short_exponential_weight_portfolio(
    predictions: pd.Series,
    num_long_positions: int = 20,
    num_short_positions: int = 0,
    decay: float = 0.5,
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    history_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Exponential-decay ranked long/short portfolio."""
    _check_counts(num_long_positions, num_short_positions)
    if not 0 < decay < 1:
        raise ValueError("decay must be between 0 and 1 (exclusive)")
    long_pool = predictions[predictions > long_threshold] if long_threshold is not None else predictions[predictions > 0]
    short_pool = predictions[predictions < short_threshold] if short_threshold is not None else predictions[predictions < 0]
    long_stocks = list(long_pool.sort_values(ascending=False).index[:num_long_positions])
    short_stocks = list(short_pool.drop(long_stocks, errors="ignore").sort_values().index[:num_short_positions])
    weights: dict[str, float] = {}
    if long_stocks:
        raw = [(1 - decay) * decay**i for i in range(len(long_stocks))]
        total = sum(raw)
        scale = 0.5 / total if total else 0
        weights.update({s: w * scale for s, w in zip(long_stocks, raw)})
    if short_stocks:
        raw = [-(1 - decay) * decay**i for i in range(len(short_stocks))]
        total = sum(raw)
        scale = -0.5 / total if total else 0
        weights.update({s: w * scale for s, w in zip(short_stocks, raw)})
    return weights


def long_short_inverse_volatility_portfolio(
    predictions: pd.Series,
    num_long_positions: int = 20,
    num_short_positions: int = 0,
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    history_df: pd.DataFrame | None = None,
    price_col: str = "close",
    lookback_days: int = 63,
    long_exposure: float = 1.0,
    short_exposure: float = 1.0,
) -> dict[str, float]:
    """Inverse-volatility weighted long/short portfolio."""
    _check_counts(num_long_positions, num_short_positions)
    long_stocks, short_stocks = _split_long_short(
        predictions, num_long_positions, num_short_positions, long_threshold, short_threshold
    )
    if not long_stocks and not short_stocks:
        return {}
    returns = _prepare_returns_matrix(history_df, long_stocks + short_stocks, price_col, lookback_days)
    if returns.empty:
        return long_short_equal_weight_portfolio(
            predictions, num_long_positions, num_short_positions, long_threshold, short_threshold
        )
    vol = returns.std(ddof=0).replace(0, np.nan)

    def _inv_vol_w(stocks: list[str]) -> pd.Series:
        inv = (1 / vol.reindex(stocks)).replace([np.inf, -np.inf], np.nan).dropna()
        if inv.empty:
            return pd.Series(1.0 / len(stocks), index=stocks)
        return (inv / inv.sum()).reindex(stocks).fillna(0.0)

    return _apply_legs(long_stocks, short_stocks, long_exposure, short_exposure, _inv_vol_w)


def long_short_mean_variance_portfolio(
    predictions: pd.Series,
    num_long_positions: int = 20,
    num_short_positions: int = 0,
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    history_df: pd.DataFrame | None = None,
    price_col: str = "close",
    lookback_days: int = 63,
    shrinkage: float = 0.1,
    shrinkage_method: str = "diagonal",
    ridge: float = 1e-6,
    risk_aversion: float = 1.0,
    transaction_costs: dict | None = None,
    cost_per_share: float | None = None,
    cost_per_dollar: pd.Series | dict | None = None,
    cost_penalty: float = 0.0,
    long_exposure: float = 1.0,
    short_exposure: float = 1.0,
) -> dict[str, float]:
    """Mean-variance optimal long/short portfolio."""
    _check_counts(num_long_positions, num_short_positions)
    adjusted = _adjust_predictions_for_costs(
        pd.Series(predictions).copy(), history_df, price_col, transaction_costs, cost_per_share, cost_per_dollar, cost_penalty
    )
    long_stocks, short_stocks = _split_long_short(
        adjusted, num_long_positions, num_short_positions, long_threshold, short_threshold
    )
    if not long_stocks and not short_stocks:
        return {}
    returns = _prepare_returns_matrix(history_df, long_stocks + short_stocks, price_col, lookback_days)
    if returns.empty:
        return long_short_inverse_volatility_portfolio(
            adjusted, num_long_positions, num_short_positions, long_threshold, short_threshold,
            history_df, price_col, lookback_days, long_exposure, short_exposure
        )

    def _mv_weights(stocks: list[str]) -> np.ndarray:
        mu = adjusted.reindex(stocks).fillna(0.0).abs()
        cov = _shrink_covariance(returns[stocks].dropna(how="all"), shrinkage_method, shrinkage) + np.eye(len(mu)) * ridge
        w = np.linalg.pinv(cov) @ mu.to_numpy(dtype=float)
        if risk_aversion > 0:
            w = w / risk_aversion
        w = np.clip(np.nan_to_num(w), 0.0, None)
        return w / w.sum() if w.sum() > 0 else np.full(len(stocks), 1.0 / len(stocks))

    return _apply_legs(long_stocks, short_stocks, long_exposure, short_exposure, _mv_weights)


def long_short_risk_parity_portfolio(
    predictions: pd.Series,
    num_long_positions: int = 20,
    num_short_positions: int = 0,
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    history_df: pd.DataFrame | None = None,
    price_col: str = "close",
    lookback_days: int = 63,
    shrinkage: float = 0.1,
    shrinkage_method: str = "diagonal",
    long_exposure: float = 1.0,
    short_exposure: float = 1.0,
) -> dict[str, float]:
    """Risk parity long/short portfolio."""
    _check_counts(num_long_positions, num_short_positions)
    long_stocks, short_stocks = _split_long_short(
        predictions, num_long_positions, num_short_positions, long_threshold, short_threshold
    )
    if not long_stocks and not short_stocks:
        return {}
    returns = _prepare_returns_matrix(history_df, long_stocks + short_stocks, price_col, lookback_days)
    if returns.empty:
        return long_short_inverse_volatility_portfolio(
            predictions, num_long_positions, num_short_positions, long_threshold, short_threshold
        )

    def _rp_w(stocks: list[str]) -> np.ndarray:
        cov = _shrink_covariance(returns[stocks].dropna(how="all"), shrinkage_method, shrinkage)
        w = _risk_parity_weights(cov)
        return w if w.size > 0 else np.full(len(stocks), 1.0 / len(stocks))

    return _apply_legs(long_stocks, short_stocks, long_exposure, short_exposure, _rp_w)


def long_short_min_variance_portfolio(
    predictions: pd.Series,
    num_long_positions: int = 20,
    num_short_positions: int = 0,
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    history_df: pd.DataFrame | None = None,
    price_col: str = "close",
    lookback_days: int = 63,
    shrinkage: float = 0.1,
    shrinkage_method: str = "diagonal",
    ridge: float = 1e-6,
    long_exposure: float = 1.0,
    short_exposure: float = 1.0,
) -> dict[str, float]:
    """Minimum variance long/short portfolio."""
    _check_counts(num_long_positions, num_short_positions)
    long_stocks, short_stocks = _split_long_short(
        predictions, num_long_positions, num_short_positions, long_threshold, short_threshold
    )
    if not long_stocks and not short_stocks:
        return {}
    returns = _prepare_returns_matrix(history_df, long_stocks + short_stocks, price_col, lookback_days)
    if returns.empty:
        return long_short_inverse_volatility_portfolio(
            predictions, num_long_positions, num_short_positions, long_threshold, short_threshold
        )

    def _minvar(stocks: list[str]) -> np.ndarray:
        sub = returns[stocks].dropna(how="all")
        cov = _shrink_covariance(sub, shrinkage_method, shrinkage) + np.eye(sub.shape[1]) * ridge
        inv_cov = np.linalg.pinv(cov)
        ones = np.ones(cov.shape[0])
        w = inv_cov @ ones
        return w / w.sum() if w.sum() != 0 else np.full(cov.shape[0], 1.0 / cov.shape[0])

    return _apply_legs(long_stocks, short_stocks, long_exposure, short_exposure, _minvar)


def long_short_hrp_portfolio(
    predictions: pd.Series,
    num_long_positions: int = 20,
    num_short_positions: int = 0,
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    history_df: pd.DataFrame | None = None,
    price_col: str = "close",
    lookback_days: int = 63,
    shrinkage: float = 0.1,
    shrinkage_method: str = "diagonal",
    linkage_method: str = "single",
    long_exposure: float = 1.0,
    short_exposure: float = 1.0,
) -> dict[str, float]:
    """Hierarchical risk parity long/short portfolio. Falls back to risk parity if scipy unavailable."""
    _check_counts(num_long_positions, num_short_positions)
    long_stocks, short_stocks = _split_long_short(
        predictions, num_long_positions, num_short_positions, long_threshold, short_threshold
    )
    if not long_stocks and not short_stocks:
        return {}
    returns = _prepare_returns_matrix(history_df, long_stocks + short_stocks, price_col, lookback_days)
    if returns.empty:
        return long_short_inverse_volatility_portfolio(
            predictions, num_long_positions, num_short_positions, long_threshold, short_threshold
        )

    def _hrp_w(stocks: list[str]) -> np.ndarray:
        sub = returns[stocks].dropna(how="all")
        cov = _shrink_covariance(sub, shrinkage_method, shrinkage)
        if cov.size == 0:
            return np.full(len(stocks), 1.0 / len(stocks))
        corr = _cov_to_corr(cov)
        try:
            from scipy.cluster.hierarchy import linkage
            from scipy.spatial.distance import squareform
        except ImportError:
            logger.warning("HRP requires scipy; falling back to risk parity.")
            w = _risk_parity_weights(cov)
            return w if w.size > 0 else np.full(len(stocks), 1.0 / len(stocks))
        dist = np.clip(np.sqrt(0.5 * (1 - corr)), 0.0, None)
        link = linkage(squareform(dist, checks=False), method=linkage_method)
        sort_ix = _get_quasi_diag(link)
        if sort_ix:
            w = _hrp_allocation(cov, sort_ix)
            if w.size > 0:
                return pd.Series(w, index=sort_ix).reindex(range(cov.shape[0])).fillna(0.0).to_numpy()
        w = _risk_parity_weights(cov)
        return w if w.size > 0 else np.full(len(stocks), 1.0 / len(stocks))

    return _apply_legs(long_stocks, short_stocks, long_exposure, short_exposure, _hrp_w)


def long_short_volatility_target_portfolio(
    predictions: pd.Series,
    num_long_positions: int = 20,
    num_short_positions: int = 0,
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    history_df: pd.DataFrame | None = None,
    price_col: str = "close",
    lookback_days: int = 63,
    target_vol: float = 0.15,
    max_scale: float | None = None,
    long_exposure: float = 1.0,
    short_exposure: float = 1.0,
) -> dict[str, float]:
    """Inverse-vol portfolio scaled to a target annualized volatility."""
    weights = long_short_inverse_volatility_portfolio(
        predictions, num_long_positions, num_short_positions, long_threshold, short_threshold,
        history_df, price_col, lookback_days, long_exposure, short_exposure,
    )
    if not weights or history_df is None or history_df.empty:
        return weights
    returns = _prepare_returns_matrix(history_df, list(weights.keys()), price_col, lookback_days)
    if returns.empty:
        return weights
    w_series = pd.Series(weights).reindex(returns.columns).fillna(0.0)
    realized_vol = (returns.mul(w_series, axis=1).sum(axis=1).std(ddof=0) * np.sqrt(252))
    if realized_vol == 0 or np.isnan(realized_vol):
        return weights
    scale = float(target_vol) / realized_vol
    if max_scale is not None:
        scale = min(scale, float(max_scale))
    if scale <= 0:
        return weights
    return (w_series * scale).to_dict()


def long_short_kelly_portfolio(
    predictions: pd.Series,
    num_long_positions: int = 20,
    num_short_positions: int = 0,
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    history_df: pd.DataFrame | None = None,
    price_col: str = "close",
    lookback_days: int = 63,
    kelly_fraction: float = 0.5,
    max_abs_weight: float | None = None,
    long_exposure: float = 1.0,
    short_exposure: float = 1.0,
) -> dict[str, float]:
    """Kelly-sized long/short portfolio (µ/σ² sizing)."""
    _check_counts(num_long_positions, num_short_positions)
    long_stocks, short_stocks = _split_long_short(
        predictions, num_long_positions, num_short_positions, long_threshold, short_threshold
    )
    if not long_stocks and not short_stocks:
        return {}
    returns = _prepare_returns_matrix(history_df, long_stocks + short_stocks, price_col, lookback_days)
    if returns.empty:
        return long_short_equal_weight_portfolio(
            predictions, num_long_positions, num_short_positions, long_threshold, short_threshold
        )
    var = returns.var(ddof=0).replace(0, np.nan)
    mu = predictions.copy()

    def _kelly_w(stocks: list[str]) -> pd.Series:
        raw = (mu.reindex(stocks) / var.reindex(stocks)).replace([np.inf, -np.inf], np.nan).dropna()
        if raw.empty:
            return pd.Series(1.0 / len(stocks), index=stocks)
        return (raw / raw.sum()).reindex(stocks).fillna(0.0)

    weights = _apply_legs(
        long_stocks, short_stocks,
        long_exposure * kelly_fraction, short_exposure * kelly_fraction,
        _kelly_w,
    )
    if max_abs_weight is not None:
        maw = float(max_abs_weight)
        weights = {k: float(np.clip(v, -maw, maw)) for k, v in weights.items()}
    return weights


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
    """Equal-weight portfolio after adjusting signal magnitudes for transaction costs."""
    _check_counts(num_long_positions, num_short_positions)
    adjusted = _adjust_predictions_for_costs(
        pd.Series(predictions).copy(), history_df, price_col, transaction_costs, cost_per_share, cost_per_dollar, cost_penalty
    )
    long_stocks, short_stocks = _split_long_short(
        adjusted, num_long_positions, num_short_positions, long_threshold, short_threshold
    )
    return _apply_legs(
        long_stocks, short_stocks, long_exposure, short_exposure,
        lambda stocks: pd.Series(1.0 / len(stocks), index=stocks),
    )


def long_short_mean_variance_turnover_portfolio(
    predictions: pd.Series,
    num_long_positions: int = 20,
    num_short_positions: int = 0,
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    history_df: pd.DataFrame | None = None,
    price_col: str = "close",
    lookback_days: int = 63,
    shrinkage: float = 0.1,
    shrinkage_method: str = "diagonal",
    ridge: float = 1e-6,
    risk_aversion: float = 1.0,
    prev_weights: dict[str, float] | None = None,
    turnover_penalty: float = 0.0,
    transaction_costs: dict | None = None,
    cost_per_share: float | None = None,
    cost_per_dollar: pd.Series | dict | None = None,
    cost_penalty: float = 0.0,
    long_exposure: float = 1.0,
    short_exposure: float = 1.0,
) -> dict[str, float]:
    """Mean-variance portfolio with L2 turnover penalty anchored to prev_weights."""
    _check_counts(num_long_positions, num_short_positions)
    adjusted = _adjust_predictions_for_costs(
        pd.Series(predictions).copy(), history_df, price_col, transaction_costs, cost_per_share, cost_per_dollar, cost_penalty
    )
    long_stocks, short_stocks = _split_long_short(
        adjusted, num_long_positions, num_short_positions, long_threshold, short_threshold
    )
    if not long_stocks and not short_stocks:
        return {}
    returns = _prepare_returns_matrix(history_df, long_stocks + short_stocks, price_col, lookback_days)
    if returns.empty:
        return long_short_cost_adjusted_portfolio(
            adjusted, num_long_positions, num_short_positions, long_threshold, short_threshold,
            history_df, price_col, transaction_costs, cost_per_share, cost_per_dollar, cost_penalty, long_exposure, short_exposure
        )
    tp = max(0.0, float(turnover_penalty))
    ra = float(risk_aversion) if risk_aversion > 0 else 1.0
    prev = pd.Series(prev_weights or {}, dtype=float)

    def _mvt_w(stocks: list[str]) -> np.ndarray:
        mu = adjusted.reindex(stocks).fillna(0.0).abs()
        p = prev.reindex(stocks).fillna(0.0).abs()
        sub = returns[stocks].dropna(how="all")
        cov = _shrink_covariance(sub, shrinkage_method, shrinkage) + np.eye(len(mu)) * ridge
        A = ra * cov + 2 * tp * np.eye(cov.shape[0])
        b = mu.to_numpy(dtype=float) + 2 * tp * p.to_numpy(dtype=float)
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(A) @ b
        w = np.clip(np.nan_to_num(w), 0.0, None)
        return w / w.sum() if w.sum() > 0 else np.full(len(stocks), 1.0 / len(stocks))

    return _apply_legs(long_stocks, short_stocks, long_exposure, short_exposure, _mvt_w)
