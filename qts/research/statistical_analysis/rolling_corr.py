"""Rolling cross-asset correlation utilities."""

from __future__ import annotations

import numpy as np
import polars as pl


def rolling_correlation_matrix(
    df: pl.DataFrame,
    window: int = 63,
    price_col: str = "close",
) -> dict[str, pl.DataFrame]:
    """Compute a rolling correlation matrix across all symbols.

    Parameters
    ----------
    df:
        Long-format OHLCV frame with columns [date, symbol, close].
    window:
        Rolling window in trading days.
    price_col:
        Column used to compute log returns.

    Returns
    -------
    Dict mapping each date (as ISO string) to a symmetric Polars DataFrame whose
    index column is "symbol" and remaining columns are symbol names.
    Only dates with at least `window` observations are included.
    """
    if df.is_empty():
        return {}

    # Pivot to wide format: date × symbol returns
    returns = (
        df.sort(["symbol", "date"])
        .with_columns(
            (pl.col(price_col).log() - pl.col(price_col).log().shift(1).over("symbol")).alias("log_ret")
        )
        .drop_nulls("log_ret")
        .pivot(on="symbol", index="date", values="log_ret")
        .sort("date")
    )

    dates = returns["date"].to_list()
    symbols = [c for c in returns.columns if c != "date"]
    if not symbols or len(dates) < window:
        return {}

    mat = returns.select(symbols).to_numpy()
    result: dict[str, pl.DataFrame] = {}

    for i in range(window - 1, len(dates)):
        window_mat = mat[i - window + 1 : i + 1]
        # Drop columns that are entirely nan
        valid_mask = ~np.all(np.isnan(window_mat), axis=0)
        valid_symbols = [s for s, v in zip(symbols, valid_mask) if v]
        if len(valid_symbols) < 2:
            continue
        sub = window_mat[:, valid_mask]
        # Replace nan with column mean for correlation
        col_means = np.nanmean(sub, axis=0)
        nan_mask = np.isnan(sub)
        sub_filled = np.where(nan_mask, col_means, sub)
        corr = np.corrcoef(sub_filled.T)
        corr_df = pl.DataFrame(
            {"symbol": valid_symbols, **{s: corr[:, j].tolist() for j, s in enumerate(valid_symbols)}}
        )
        result[str(dates[i])] = corr_df

    return result


def average_pairwise_correlation(
    df: pl.DataFrame,
    window: int = 63,
    price_col: str = "close",
) -> pl.DataFrame:
    """Compute the average absolute pairwise correlation over time.

    Returns a (date, avg_correlation) DataFrame suitable for regime monitoring.
    """
    matrices = rolling_correlation_matrix(df, window=window, price_col=price_col)
    rows: list[dict] = []
    for date_str, corr_df in sorted(matrices.items()):
        symbols = [c for c in corr_df.columns if c != "symbol"]
        if not symbols:
            continue
        mat = corr_df.select(symbols).to_numpy()
        n = mat.shape[0]
        # Upper triangle excluding diagonal
        upper_indices = np.triu_indices(n, k=1)
        if upper_indices[0].size == 0:
            continue
        avg_corr = float(np.abs(mat[upper_indices]).mean())
        rows.append({"date": date_str, "avg_correlation": avg_corr})

    if not rows:
        return pl.DataFrame(schema={"date": pl.String, "avg_correlation": pl.Float64})
    return pl.DataFrame(rows).with_columns(pl.col("date").str.to_date().alias("date")).sort("date")
