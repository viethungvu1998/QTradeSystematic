"""Walk-forward signal generation for the VN100 quantamental strategy."""

from __future__ import annotations

import math
from datetime import date

import pandas as pd
import polars as pl

from qts.research.backtest._runner import _rebalance_dates
from qts.research.portfolio_construction import long_short_equal_weight_portfolio
from qts.research.strategies.base import BaseStrategy
from qts.research.strategies.factor.algorithms import train_and_predict_xgb_regressor
from qts.research.strategies.factor.core import factor_feature_columns

from .config import MODEL_PARAMS, ExperimentConfig, FeatureConfig


def default_predictor_candidates(config: FeatureConfig) -> list[str]:
    from qts.research.features.fundamentals import FUNDAMENTAL_FACTOR_GROUPS
    from .config import qsmom_column

    return [
        qsmom_column(config),
        "technicalCompositeFactor",
        "fundamentalCompositeFactor",
        "hybridCompositeFactor",
        *FUNDAMENTAL_FACTOR_GROUPS.keys(),
        "roc_63",
        "roc_252",
        "rsi_63",
        "macd_hist",
        "hist_vol_63",
        "hist_vol_126",
        "close_to_sma_50",
        "close_to_sma_200",
        "volume_ratio_20",
        "dollar_volume_zscore_63",
    ]


def available_predictors(df: pl.DataFrame, candidates: list[str]) -> list[str]:
    return [
        col
        for col in candidates
        if col in df.columns and df.select(pl.col(col).is_not_null().sum()).item() > 0
    ]


def effective_long_threshold(value: float | None) -> float:
    return -math.inf if value is None else float(value)


def choose_predictors(
    df: pl.DataFrame,
    config: FeatureConfig,
    explicit_cols: list[str] | None = None,
) -> list[str]:
    if explicit_cols:
        return available_predictors(df, explicit_cols)
    # Default: all feature columns present in the dataframe (excludes OHLCV base
    # columns, signal/weight, and any forward_return_* target columns).
    all_feature_cols = factor_feature_columns(df.columns)
    return available_predictors(df, all_feature_cols)


def signals_from_weights(trade_date: date, weights: dict[str, float]) -> pl.DataFrame:
    rows = [
        {
            "date": trade_date,
            "symbol": symbol,
            "signal": 1 if weight > 0 else -1,
            "weight": abs(float(weight)),
        }
        for symbol, weight in weights.items()
        if float(weight) != 0.0
    ]
    if not rows:
        return BaseStrategy.empty_signal_frame()
    schema = {"date": pl.Date, "symbol": pl.Utf8, "signal": pl.Int8, "weight": pl.Float64}
    return pl.DataFrame(rows, schema=schema)


def walk_forward_ml_signals(
    df: pl.DataFrame,
    experiment: ExperimentConfig,
    predictor_cols: list[str],
    target_col: str,
) -> pl.DataFrame:
    all_dates = sorted(df["date"].unique().to_list())
    rebalance = _rebalance_dates(all_dates, experiment.rebalance_period)
    date_index = {item: idx for idx, item in enumerate(all_dates)}
    signal_frames: list[pl.DataFrame] = []
    model_params = experiment.model_params or MODEL_PARAMS

    for rebalance_date in rebalance:
        idx = date_index[rebalance_date]
        if idx < max(experiment.feature.qsmom_slow, experiment.feature.forward_period, 80):
            continue
        train_start = all_dates[max(0, idx - experiment.train_window)]
        train = df.filter(
            (pl.col("date") >= train_start)
            & (pl.col("date") < rebalance_date)
            & pl.col(target_col).is_not_null()
        ).drop_nulls(predictor_cols + [target_col])
        predict = df.filter(pl.col("date") == rebalance_date).drop_nulls(predictor_cols)
        min_predict_rows = max(
            2, min(experiment.num_long_positions, df.select("symbol").n_unique())
        )
        if train.height < 100 or predict.height < min_predict_rows:
            continue

        scores = train_and_predict_xgb_regressor(
            train_data=train,
            predict_data=predict,
            predictor_cols=predictor_cols,
            target_col=target_col,
            model_params=model_params,
        )
        predictions = pd.Series(scores, index=predict["symbol"].to_list())
        weights = long_short_equal_weight_portfolio(
            predictions=predictions,
            num_long_positions=experiment.num_long_positions,
            num_short_positions=experiment.num_short_positions,
            long_threshold=effective_long_threshold(experiment.long_threshold),
            short_threshold=experiment.short_threshold,
            history_df=train.to_pandas(),
        )
        signals = signals_from_weights(rebalance_date, weights)
        if not signals.is_empty():
            signal_frames.append(signals)

    if not signal_frames:
        return BaseStrategy.empty_signal_frame()
    return pl.concat(signal_frames, how="vertical").sort(["date", "symbol"])


__all__ = [
    "available_predictors",
    "choose_predictors",
    "default_predictor_candidates",
    "effective_long_threshold",
    "signals_from_weights",
    "walk_forward_ml_signals",
]
