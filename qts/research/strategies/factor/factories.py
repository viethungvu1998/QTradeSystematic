"""Registered factor strategy factories."""

from __future__ import annotations

import pandas as pd
import polars as pl

from qts.core.registry import Registry

from .algorithms import (
    train_and_predict_ic_composite,
    train_and_predict_xgb_ranker,
    train_and_predict_xgb_regressor,
)
from qts.research.portfolio_construction import (
    long_short_cost_adjusted_portfolio,
    long_short_equal_weight_portfolio,
    long_short_exponential_weight_portfolio,
    long_short_hrp_portfolio,
    long_short_inverse_volatility_portfolio,
    long_short_kelly_portfolio,
    long_short_mean_variance_portfolio,
    long_short_mean_variance_turnover_portfolio,
    long_short_min_variance_portfolio,
    long_short_risk_parity_portfolio,
    long_short_volatility_target_portfolio,
)

Registry.register_factor_trainer("xgb_regressor")(train_and_predict_xgb_regressor)
Registry.register_factor_trainer("xgb_ranker")(train_and_predict_xgb_ranker)
Registry.register_factor_trainer("ic_composite")(train_and_predict_ic_composite)

Registry.register_portfolio_constructor("equal_weight")(long_short_equal_weight_portfolio)
Registry.register_portfolio_constructor("exponential_weight")(long_short_exponential_weight_portfolio)
Registry.register_portfolio_constructor("inverse_volatility")(long_short_inverse_volatility_portfolio)
Registry.register_portfolio_constructor("mean_variance")(long_short_mean_variance_portfolio)
Registry.register_portfolio_constructor("risk_parity")(long_short_risk_parity_portfolio)
Registry.register_portfolio_constructor("min_variance")(long_short_min_variance_portfolio)
Registry.register_portfolio_constructor("hrp")(long_short_hrp_portfolio)
Registry.register_portfolio_constructor("volatility_target")(long_short_volatility_target_portfolio)
Registry.register_portfolio_constructor("kelly")(long_short_kelly_portfolio)
Registry.register_portfolio_constructor("cost_adjusted")(long_short_cost_adjusted_portfolio)
Registry.register_portfolio_constructor("mean_variance_turnover")(long_short_mean_variance_turnover_portfolio)


@Registry.register_signal_algorithm("cross_sectional_rank")
def cross_sectional_rank(
    df: pl.DataFrame,
    *,
    predictor_cols: list[str],
    **kwargs,
) -> pd.Series:
    """Z-score each predictor cross-sectionally on the most recent date, average."""
    last_date = df["date"].max()
    last = df.filter(pl.col("date") == last_date)
    pdf = last.select(["symbol"] + predictor_cols).to_pandas().set_index("symbol")
    z_scores = (pdf - pdf.mean()) / pdf.std().replace(0, 1)
    return z_scores.mean(axis=1).rename("score")


@Registry.register_signal_algorithm("factor_as_signal")
def factor_as_signal(
    df: pl.DataFrame,
    *,
    predictor_cols: list[str],
    **kwargs,
) -> pd.Series:
    """Use predictor_cols[0] directly on the most recent date as the score."""
    col = predictor_cols[0]
    last_date = df["date"].max()
    last = df.filter(pl.col("date") == last_date)
    pdf = last.select(["symbol", col]).to_pandas().set_index("symbol")
    return pdf[col].rename("score")


@Registry.register_signal_algorithm("ic_weighted")
def ic_weighted(
    df: pl.DataFrame,
    *,
    predictor_cols: list[str],
    ic_window: int = 252,
    forward_col: str | None = None,
    **kwargs,
) -> pd.Series:
    """Weight each factor by its rolling Spearman IC against forward returns."""
    if forward_col and forward_col in df.columns:
        try:
            pdf = df.select(["symbol", "date"] + predictor_cols + [forward_col]).to_pandas()
            ic_weights: dict[str, float] = {}
            for col in predictor_cols:
                tail = pdf[[col, forward_col]].dropna().tail(ic_window)
                if len(tail) >= 20:
                    ic_weights[col] = tail[col].corr(tail[forward_col], method="spearman")
                else:
                    ic_weights[col] = 0.0
            total = sum(abs(value) for value in ic_weights.values())
            if total > 0:
                ic_weights = {key: value / total for key, value in ic_weights.items()}
            else:
                ic_weights = {col: 1.0 / len(predictor_cols) for col in predictor_cols}

            last_date = df["date"].max()
            last = df.filter(pl.col("date") == last_date)
            lpdf = last.select(["symbol"] + predictor_cols).to_pandas().set_index("symbol")
            score = sum(lpdf[col] * ic_weights[col] for col in predictor_cols)
            return score.rename("score")
        except Exception:
            pass

    return cross_sectional_rank(df, predictor_cols=predictor_cols)
