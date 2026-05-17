"""Benchmark comparison: alpha, beta, and information ratio."""

from __future__ import annotations

import math

import numpy as np
import polars as pl


def compare_to_benchmark(
    strategy_returns: pl.DataFrame,
    benchmark_returns: pl.DataFrame,
    return_col: str = "portfolio_return",
    benchmark_col: str = "portfolio_return",
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Compute alpha, beta, and information ratio vs a benchmark return series.

    Parameters
    ----------
    strategy_returns:
        DataFrame with columns [date, <return_col>].
    benchmark_returns:
        DataFrame with columns [date, <benchmark_col>].
    return_col:
        Column name for strategy daily returns.
    benchmark_col:
        Column name for benchmark daily returns.
    periods_per_year:
        Trading periods per year for annualisation.

    Returns
    -------
    dict with keys: alpha, beta, information_ratio, tracking_error, correlation.
    """
    joined = (
        strategy_returns.rename({return_col: "strat"})
        .join(benchmark_returns.rename({benchmark_col: "bench"}), on="date", how="inner")
        .sort("date")
        .drop_nulls(["strat", "bench"])
    )
    if joined.height < 3:
        return {"alpha": float("nan"), "beta": float("nan"), "information_ratio": float("nan"),
                "tracking_error": float("nan"), "correlation": float("nan")}

    s = joined["strat"].to_numpy()
    b = joined["bench"].to_numpy()

    # Beta via OLS
    cov_sb = float(np.cov(s, b)[0, 1])
    var_b = float(np.var(b, ddof=1))
    beta = cov_sb / var_b if var_b != 0 else float("nan")

    # Alpha (annualised)
    excess = s - beta * b
    alpha = float(np.mean(excess)) * periods_per_year

    # Tracking error and IR
    active = s - b
    te = float(np.std(active, ddof=1)) * math.sqrt(periods_per_year)
    ir = float(np.mean(active)) * periods_per_year / te if te > 0 else float("nan")

    corr = float(np.corrcoef(s, b)[0, 1]) if len(s) >= 2 else float("nan")

    return {
        "alpha": alpha,
        "beta": beta,
        "information_ratio": ir,
        "tracking_error": te,
        "correlation": corr,
    }
