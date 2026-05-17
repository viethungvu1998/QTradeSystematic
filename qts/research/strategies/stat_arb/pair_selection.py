from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, coint
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "statsmodels is required for cointegration tests. "
        "Install via `pip install statsmodels`."
    ) from exc


@dataclass(frozen=True)
class PairCandidate:
    symbol_a: str
    symbol_b: str
    pvalue: float
    test_stat: float


def compute_adf_pvalue(series: pd.Series) -> float:
    cleaned = series.dropna()
    if cleaned.empty:
        return float("nan")
    try:
        result = adfuller(cleaned, autolag="AIC")
    except Exception:  # pragma: no cover - defensive
        return float("nan")
    return float(result[1])


def compute_half_life(series: pd.Series) -> float:
    cleaned = series.dropna()
    if len(cleaned) < 2:
        return float("nan")
    delta = cleaned.diff().dropna()
    lagged = cleaned.shift(1).dropna()
    aligned = pd.concat([delta, lagged], axis=1).dropna()
    if aligned.empty:
        return float("nan")
    y = aligned.iloc[:, 0]
    x = sm.add_constant(aligned.iloc[:, 1])
    try:
        model = sm.OLS(y, x).fit()
    except Exception:  # pragma: no cover - defensive
        return float("nan")
    beta = model.params.iloc[1]
    if beta >= 0:
        return float("inf")
    return float(-np.log(2) / beta)


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    *,
    candidate_pairs: Iterable[Sequence[str]] | None = None,
    max_pairs: int = 50,
    pvalue_threshold: float = 0.05,
    min_obs: int = 252,
) -> list[PairCandidate]:
    if prices.empty:
        logger.warning("No price data available for pair selection.")
        return []

    symbols = list(prices.columns)
    if len(symbols) < 2:
        logger.warning("Need at least two symbols for cointegration selection.")
        return []

    candidates: list[PairCandidate] = []
    if candidate_pairs is None:
        pair_iter = combinations(symbols, 2)
    else:
        pair_iter = ensure_pair_list(candidate_pairs)
    for sym_a, sym_b in pair_iter:
        if sym_a not in prices.columns or sym_b not in prices.columns:
            continue
        pair = prices[[sym_a, sym_b]].dropna()
        if len(pair) < min_obs:
            continue
        corr = pair[sym_a].corr(pair[sym_b])
        if corr is not None and np.isfinite(corr) and abs(corr) > 0.999:
            logger.debug("Skipping nearly collinear pair %s/%s (corr=%.4f).", sym_a, sym_b, corr)
            continue
        try:
            test_stat, pvalue, _ = coint(pair[sym_a], pair[sym_b])
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Cointegration test failed for %s/%s: %s", sym_a, sym_b, exc)
            continue
        if np.isnan(pvalue):
            continue
        if pvalue <= pvalue_threshold:
            candidates.append(
                PairCandidate(sym_a, sym_b, float(pvalue), float(test_stat))
            )

    candidates.sort(key=lambda item: item.pvalue)
    if max_pairs:
        candidates = candidates[:max_pairs]
    return candidates


def estimate_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    *,
    method: str = "ols",
    add_const: bool = True,
) -> float:
    df = pd.concat([y, x], axis=1).dropna()
    if df.empty:
        return float("nan")

    if method == "fixed":
        return 1.0
    if method != "ols":
        raise ValueError(f"Unsupported hedge ratio method '{method}'.")

    y_vals = df.iloc[:, 0]
    x_vals = df.iloc[:, 1]
    if add_const:
        x_vals = sm.add_constant(x_vals)
    model = sm.OLS(y_vals, x_vals).fit()
    params = model.params.to_numpy()
    slope = params[1] if add_const else params[0]
    return float(slope)


def ensure_pair_list(
    pairs: Iterable[Sequence[str]],
) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(f"Pair entries must have 2 symbols, got {pair!r}.")
        normalized.append((str(pair[0]), str(pair[1])))
    return normalized


def preselect_pairs_by_correlation(
    prices: pd.DataFrame,
    *,
    min_corr: float | None = None,
    pairs_per_symbol: int | None = 5,
    max_pairs: int | None = None,
    use_returns: bool = True,
    method: str = "pearson",
) -> list[tuple[str, str]]:
    if prices.empty:
        return []

    data = prices
    if use_returns:
        data = prices.pct_change().dropna(how="all")
    if data.empty:
        return []

    corr = data.corr(method=method)
    symbols = list(corr.columns)
    pairs: set[tuple[str, str]] = set()
    for sym in symbols:
        series = corr[sym].drop(labels=[sym], errors="ignore")
        series = series.replace([np.inf, -np.inf], np.nan).dropna()
        if min_corr is not None:
            series = series[series.abs() >= float(min_corr)]
        if series.empty:
            continue
        series = series.abs().sort_values(ascending=False)
        if pairs_per_symbol:
            series = series.iloc[: int(pairs_per_symbol)]
        for other in series.index:
            pair = tuple(sorted((sym, other)))
            pairs.add(pair)

    pair_list = list(pairs)
    if max_pairs and pair_list:
        pair_list = sorted(
            pair_list,
            key=lambda pair: abs(corr.loc[pair[0], pair[1]]),
            reverse=True,
        )[: int(max_pairs)]
    return pair_list
