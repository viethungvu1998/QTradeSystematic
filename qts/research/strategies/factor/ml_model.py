"""ML factor strategy: train on historical features, predict on current date."""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
import polars as pl

from qts.core.registry import Registry

from .base import BaseFactorStrategy

logger = logging.getLogger(__name__)


@Registry.register_strategy("ml_factor")
class MLFactorStrategy(BaseFactorStrategy):
    """Factor strategy that trains an ML model on each rebalance call.

    train_func must accept (train_pd: pd.DataFrame, predict_pd: pd.DataFrame)
    and return np.ndarray of scores aligned to predict_pd rows.
    Use functools.partial to pre-bind predictor_cols, target_col, and model params.

    portfolio_func must accept (predictions: pd.Series, history_df: pd.DataFrame)
    as keyword arg and return dict[str, float] (symbol -> signed weight).
    Use functools.partial to pre-bind num_long_positions and other kwargs.
    history_df will be the full training frame converted to pandas.
    """

    def __init__(
        self,
        predictor_cols: list[str],
        target_col: str,
        train_func: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
        portfolio_func: Callable[..., dict[str, float]],
    ) -> None:
        self.predictor_cols = predictor_cols
        self.target_col = target_col
        self.train_func = train_func
        self.portfolio_func = portfolio_func

    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        available = [c for c in self.predictor_cols if c in df.columns]
        if not available:
            return self.empty_signal_frame()

        if self.target_col not in df.columns:
            return self.empty_signal_frame()

        df_pd = df.to_pandas()

        # Rows with known target → train; last date rows → predict
        train_pd = df_pd[df_pd[self.target_col].notna()].copy()
        last_date = df_pd["date"].max()
        predict_pd = df_pd[df_pd["date"] == last_date].copy()

        if len(train_pd) < 2 or predict_pd.empty:
            return self.empty_signal_frame()

        train_pd = train_pd.dropna(subset=available)
        if len(train_pd) < 2:
            return self.empty_signal_frame()

        scores = self.train_func(train_pd, predict_pd)

        predictions = pd.Series(scores, index=predict_pd["symbol"].values)

        weights_dict = self.portfolio_func(predictions, history_df=df_pd)

        if not weights_dict:
            return self.empty_signal_frame()

        return self.signal_frame_from_weights(last_date, weights_dict)
