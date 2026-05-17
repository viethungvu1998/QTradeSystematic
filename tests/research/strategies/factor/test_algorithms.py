"""Tests for strategies.factor.algorithms."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

pytest.importorskip("xgboost")

from qts.research.strategies.factor.algorithms import (
    train_and_predict_ic_composite,
    train_and_predict_linear_regression,
    train_and_predict_xgb_ranker,
    train_and_predict_xgb_regressor,
)


@pytest.fixture
def panel_pd(rng):
    """60-row panel (20 dates × 3 symbols) with features and target."""
    symbols = ["A", "B", "C"]
    base = date(2023, 1, 2)
    rows = []
    for i in range(20):
        for sym in symbols:
            rows.append({
                "date": base + timedelta(days=i),
                "symbol": sym,
                "f1": float(rng.normal(0, 1)),
                "f2": float(rng.normal(0, 1)),
                "f3": float(rng.normal(0, 1)),
                "forward_return": float(rng.normal(0, 0.01)),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def predict_pd(rng):
    """3-row predict set (1 date × 3 symbols)."""
    symbols = ["A", "B", "C"]
    return pd.DataFrame({
        "date": [date(2023, 2, 1)] * 3,
        "symbol": symbols,
        "f1": rng.normal(0, 1, 3).tolist(),
        "f2": rng.normal(0, 1, 3).tolist(),
        "f3": rng.normal(0, 1, 3).tolist(),
    })


# ---- train_and_predict_xgb_regressor ----

def test_xgb_regressor_output_length(panel_pd, predict_pd):
    result = train_and_predict_xgb_regressor(
        panel_pd, predict_pd,
        predictor_cols=["f1", "f2", "f3"],
        target_col="forward_return",
        model_params={"n_estimators": 10, "random_state": 42},
    )
    assert len(result) == len(predict_pd)


def test_xgb_regressor_finite_floats(panel_pd, predict_pd):
    result = train_and_predict_xgb_regressor(
        panel_pd, predict_pd,
        predictor_cols=["f1", "f2", "f3"],
        target_col="forward_return",
        model_params={"n_estimators": 10, "random_state": 42},
    )
    assert all(np.isfinite(v) for v in result)


def test_xgb_regressor_accepts_polars(panel_pd, predict_pd, rng):
    train_pl = pl.DataFrame(panel_pd)
    predict_pl = pl.DataFrame(predict_pd)
    result = train_and_predict_xgb_regressor(
        train_pl, predict_pl,
        predictor_cols=["f1", "f2", "f3"],
        target_col="forward_return",
        model_params={"n_estimators": 10, "random_state": 42},
    )
    assert len(result) == len(predict_pd)


# ---- train_and_predict_xgb_ranker ----

@pytest.fixture
def panel_with_rank(panel_pd):
    """panel_pd with integer rank labels for XGBRanker (rank:pairwise accepts floats too)."""
    df = panel_pd.copy()
    # rank:pairwise works with continuous floats; no change needed — use model_params to specify
    return df


def test_xgb_ranker_output_length(panel_with_rank, predict_pd):
    assert panel_with_rank["date"].nunique() >= 2
    result = train_and_predict_xgb_ranker(
        panel_with_rank, predict_pd,
        predictor_cols=["f1", "f2", "f3"],
        target_col="forward_return",
        model_params={"n_estimators": 10, "random_state": 42, "objective": "rank:pairwise"},
    )
    assert len(result) == len(predict_pd)


def test_xgb_ranker_finite_floats(panel_with_rank, predict_pd):
    result = train_and_predict_xgb_ranker(
        panel_with_rank, predict_pd,
        predictor_cols=["f1", "f2", "f3"],
        target_col="forward_return",
        model_params={"n_estimators": 10, "random_state": 42, "objective": "rank:pairwise"},
    )
    assert all(np.isfinite(v) for v in result)


# ---- train_and_predict_linear_regression ----

def test_linear_regression_ridge_output_series(panel_pd):
    result = train_and_predict_linear_regression(
        panel_pd,
        predictor_cols=["f1", "f2", "f3"],
        target_col="forward_return",
        model_params={"model": "ridge"},
    )
    assert isinstance(result, pd.Series)
    assert len(result) > 0


def test_linear_regression_zero_variance_col_handled(panel_pd):
    panel_pd = panel_pd.copy()
    panel_pd["f_const"] = 1.0
    result = train_and_predict_linear_regression(
        panel_pd,
        predictor_cols=["f1", "f_const"],
        target_col="forward_return",
    )
    assert isinstance(result, pd.Series)


def test_linear_regression_all_nan_target():
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "forward_return": [np.nan, np.nan, np.nan]})
    result = train_and_predict_linear_regression(df, predictor_cols=["f1"], target_col="forward_return")
    assert isinstance(result, pd.Series)
    assert len(result) == 0


# ---- train_and_predict_ic_composite ----

@pytest.fixture
def ic_panel(rng):
    """65 dates × 5 symbols panel for IC composite tests."""
    pytest.importorskip("scipy")
    symbols = ["A", "B", "C", "D", "E"]
    base = date(2022, 1, 3)
    rows = []
    for i in range(65):
        for sym in symbols:
            rows.append({
                "date": base + timedelta(days=i),
                "symbol": sym,
                "f1": float(rng.normal(0, 1)),
                "f2": float(rng.normal(0, 1)),
                "f3": float(rng.normal(0, 1)),
                "fwd_21": float(rng.normal(0, 0.02)),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def ic_predict(rng):
    symbols = ["A", "B", "C", "D", "E"]
    return pd.DataFrame({
        "date": [date(2022, 4, 1)] * 5,
        "symbol": symbols,
        "f1": rng.normal(0, 1, 5).tolist(),
        "f2": rng.normal(0, 1, 5).tolist(),
        "f3": rng.normal(0, 1, 5).tolist(),
    })


def test_ic_composite_output_length(ic_panel, ic_predict):
    pytest.importorskip("scipy")
    result = train_and_predict_ic_composite(
        ic_panel, ic_predict,
        factor_cols=["f1", "f2", "f3"],
        future_return_cols={21: "fwd_21"},
        composite_params={"min_dates": 60},
    )
    assert len(result) == len(ic_predict)


def test_ic_composite_output_finite(ic_panel, ic_predict):
    pytest.importorskip("scipy")
    result = train_and_predict_ic_composite(
        ic_panel, ic_predict,
        factor_cols=["f1", "f2", "f3"],
        future_return_cols={21: "fwd_21"},
        composite_params={"min_dates": 60},
    )
    assert all(np.isfinite(v) for v in result)


def test_ic_composite_too_few_dates_raises(ic_predict, rng):
    pytest.importorskip("scipy")
    symbols = ["A", "B"]
    base = date(2022, 1, 3)
    rows = []
    for i in range(5):
        for sym in symbols:
            rows.append({"date": base + timedelta(days=i), "symbol": sym,
                         "f1": float(rng.normal(0, 1)), "fwd_21": float(rng.normal(0, 0.01))})
    small_train = pd.DataFrame(rows)
    with pytest.raises(RuntimeError, match="too_few_dates"):
        train_and_predict_ic_composite(
            small_train, ic_predict,
            factor_cols=["f1"],
            future_return_cols={21: "fwd_21"},
            composite_params={"min_dates": 60},
        )


def test_ic_composite_no_factors_raises(ic_panel, ic_predict):
    pytest.importorskip("scipy")
    with pytest.raises(RuntimeError, match="no_factors"):
        train_and_predict_ic_composite(
            ic_panel, ic_predict,
            factor_cols=["missing1", "missing2"],
            future_return_cols={21: "fwd_21"},
        )
