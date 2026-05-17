"""Tests for strategies.factor.ml_model.MLFactorStrategy."""

from __future__ import annotations

import functools
from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

pytest.importorskip("xgboost")

from qts.research.strategies.factor.algorithms import train_and_predict_xgb_regressor
from qts.research.strategies.factor.ml_model import MLFactorStrategy
from qts.research.strategies.factor.portfolio_construction import long_short_equal_weight_portfolio

_SIGNAL_COLS = {"date", "symbol", "signal", "weight"}
_PREDICTOR_COLS = ["f1", "f2", "f3"]
_TARGET_COL = "forward_return_21"


@pytest.fixture
def ml_df(rng):
    """3 symbols × 60 dates; forward_return_21 null for last 21 dates."""
    symbols = ["A", "B", "C"]
    base = date(2023, 1, 2)
    rows = []
    for i in range(60):
        for sym in symbols:
            fwd = float(rng.normal(0.001, 0.02)) if i < 39 else None
            rows.append({
                "date": base + timedelta(days=i),
                "symbol": sym,
                "f1": float(rng.normal(0, 1)),
                "f2": float(rng.normal(0, 1)),
                "f3": float(rng.normal(0, 1)),
                _TARGET_COL: fwd,
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


@pytest.fixture
def strategy():
    train_func = functools.partial(
        train_and_predict_xgb_regressor,
        predictor_cols=_PREDICTOR_COLS,
        target_col=_TARGET_COL,
        model_params={"n_estimators": 10, "random_state": 42},
    )
    portfolio_func = functools.partial(long_short_equal_weight_portfolio, num_long_positions=2)
    return MLFactorStrategy(_PREDICTOR_COLS, _TARGET_COL, train_func, portfolio_func)


def test_generate_signals_columns(strategy, ml_df):
    result = strategy.generate_signals(ml_df)
    assert set(result.columns) == _SIGNAL_COLS


def test_generate_signals_valid_signal_values(strategy, ml_df):
    result = strategy.generate_signals(ml_df)
    assert result["signal"].is_in([-1, 0, 1]).all()


def test_generate_signals_weight_in_range(strategy, ml_df):
    result = strategy.generate_signals(ml_df)
    assert (result["weight"] >= 0.0).all()
    assert (result["weight"] <= 1.0).all()


def test_generate_signals_last_date(strategy, ml_df):
    result = strategy.generate_signals(ml_df)
    last_date = ml_df["date"].max()
    assert (result["date"] == last_date).all()


def test_generate_signals_missing_target_empty(strategy, ml_df):
    df_no_target = ml_df.drop(_TARGET_COL)
    result = strategy.generate_signals(df_no_target)
    assert result.is_empty()
    assert set(result.columns) == _SIGNAL_COLS


def test_generate_signals_all_null_target_empty(strategy, ml_df):
    df_null = ml_df.with_columns(pl.lit(None).cast(pl.Float64).alias(_TARGET_COL))
    result = strategy.generate_signals(df_null)
    assert result.is_empty()


def test_generate_signals_missing_predictors_empty(ml_df):
    train_func = functools.partial(
        train_and_predict_xgb_regressor,
        predictor_cols=["missing1", "missing2"],
        target_col=_TARGET_COL,
        model_params={"n_estimators": 10, "random_state": 42},
    )
    portfolio_func = functools.partial(long_short_equal_weight_portfolio, num_long_positions=2)
    strategy_bad = MLFactorStrategy(["missing1", "missing2"], _TARGET_COL, train_func, portfolio_func)
    result = strategy_bad.generate_signals(ml_df)
    assert result.is_empty()
    assert set(result.columns) == _SIGNAL_COLS


def test_generate_signals_weight_normalization(ml_df):
    def _big_weight_portfolio(predictions, history_df=None):
        return {"A": 3.0, "B": -2.5}

    train_func = functools.partial(
        train_and_predict_xgb_regressor,
        predictor_cols=_PREDICTOR_COLS,
        target_col=_TARGET_COL,
        model_params={"n_estimators": 10, "random_state": 42},
    )
    strategy_norm = MLFactorStrategy(_PREDICTOR_COLS, _TARGET_COL, train_func, _big_weight_portfolio)
    result = strategy_norm.generate_signals(ml_df)
    assert (result["weight"] <= 1.0).all()
    assert (result["weight"] >= 0.0).all()


def test_generate_signals_direction_matches_weight_sign(ml_df):
    def _fixed_portfolio(predictions, history_df=None):
        return {"A": 0.5, "B": -0.3, "C": 0.0}

    train_func = functools.partial(
        train_and_predict_xgb_regressor,
        predictor_cols=_PREDICTOR_COLS,
        target_col=_TARGET_COL,
        model_params={"n_estimators": 10, "random_state": 42},
    )
    strategy_dir = MLFactorStrategy(_PREDICTOR_COLS, _TARGET_COL, train_func, _fixed_portfolio)
    result = strategy_dir.generate_signals(ml_df)

    row_a = result.filter(pl.col("symbol") == "A")
    if row_a.height > 0:
        assert row_a["signal"].item() == 1

    row_b = result.filter(pl.col("symbol") == "B")
    if row_b.height > 0:
        assert row_b["signal"].item() == -1


def test_ml_factor_registered():
    from qts.core.registry import Registry
    cls = Registry.get_strategy("ml_factor")
    assert cls is MLFactorStrategy
