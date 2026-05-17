"""Factor analysis: information coefficient and quantile returns."""

from __future__ import annotations

import polars as pl


def compute_ic(
    df: pl.DataFrame,
    factor_cols: list[str],
    return_col: str,
    method: str = "spearman",
) -> pl.DataFrame:
    """Compute per-date information coefficient between each factor and a forward return.

    Parameters
    ----------
    df:
        Panel with columns [date, symbol] + factor_cols + [return_col].
    factor_cols:
        Factor columns to evaluate.
    return_col:
        Forward return column to correlate against.
    method:
        "spearman" (rank correlation) or "pearson".

    Returns
    -------
    pl.DataFrame with columns [date] + factor_cols, one IC per factor per date.
    """
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    rows: list[dict] = []
    for date_val, group in df.group_by("date"):
        g = group.drop_nulls(subset=[return_col] + factor_cols)
        if g.height < 3:
            continue
        ret = g[return_col].to_numpy()
        record: dict = {"date": date_val if not isinstance(date_val, tuple) else date_val[0]}
        for col in factor_cols:
            factor = g[col].to_numpy()
            try:
                if method == "spearman":
                    rho, _ = spearmanr(factor, ret, nan_policy="omit")
                else:
                    valid = ~(np.isnan(factor) | np.isnan(ret))
                    rho, _ = pearsonr(factor[valid], ret[valid]) if valid.sum() >= 3 else (np.nan, None)
                record[col] = float(rho) if np.isfinite(rho) else None
            except Exception:
                record[col] = None
        rows.append(record)

    if not rows:
        schema = {"date": pl.Date, **{c: pl.Float64 for c in factor_cols}}
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows).sort("date")


def compute_factor_quantile_returns(
    df: pl.DataFrame,
    factor_col: str,
    return_col: str,
    n_quantiles: int = 5,
) -> pl.DataFrame:
    """Compute mean forward return per factor quantile per date.

    Returns
    -------
    pl.DataFrame with columns [date, quantile, mean_return].
    """
    valid = df.drop_nulls(subset=[factor_col, return_col])
    if valid.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "quantile": pl.Int32, "mean_return": pl.Float64})

    labeled = valid.with_columns(
        pl.col(factor_col)
        .qcut(n_quantiles, labels=[str(i + 1) for i in range(n_quantiles)], allow_duplicates=True)
        .over("date")
        .cast(pl.Int32)
        .alias("quantile")
    )
    return (
        labeled.group_by(["date", "quantile"])
        .agg(pl.col(return_col).mean().alias("mean_return"))
        .sort(["date", "quantile"])
    )
