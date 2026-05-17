"""Tests for strategies.ml_factor.model."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from qts.core.registry import Registry
from qts.research.strategies.ml_factor.model import (
    CLASS_PERCENTILES,
    ML_FACTOR_CLASS_SCORES,
    MLFactorClass,
    MLFactorClassThresholds,
    MLFactorStrategy,
    class_scores_from_probabilities,
    classification_metrics,
    train_and_predict_xgb_classifier,
)

_SIGNAL_COLS = {"date", "symbol", "signal", "weight"}
_PREDICTOR_COLS = ["f1", "f2", "f3"]
_TARGET_COL = "forward_return_21"


@pytest.fixture
def ml_df(rng):
    symbols = ["A", "B", "C"]
    base = date(2023, 1, 2)
    rows = []
    for i in range(60):
        for offset, sym in enumerate(symbols):
            target = float((i - 30 + offset) / 1000) if i < 59 else None
            rows.append(
                {
                    "date": base + timedelta(days=i),
                    "symbol": sym,
                    "f1": float(rng.normal(0, 1)),
                    "f2": float(rng.normal(0, 1)),
                    "f3": float(rng.normal(0, 1)),
                    _TARGET_COL: target,
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


@pytest.fixture
def strategy():
    def train_func(train: pd.DataFrame, predict: pd.DataFrame) -> np.ndarray:
        thresholds = MLFactorClassThresholds.fit(train[_TARGET_COL], target_col=_TARGET_COL)
        labels = thresholds.transform(train[_TARGET_COL])
        assert set(labels.tolist()) <= {label.value for label in MLFactorClass}
        return predict["f1"].to_numpy(dtype=float)

    def portfolio_func(predictions: pd.Series, history_df: pd.DataFrame | None = None):
        ranked = predictions.sort_values(ascending=False)
        return {str(ranked.index[0]): 1.0, str(ranked.index[-1]): -1.0}

    return MLFactorStrategy(_PREDICTOR_COLS, _TARGET_COL, train_func, portfolio_func)


def test_class_labels_are_assigned_from_train_fitted_thresholds():
    train_targets = pd.Series(np.linspace(-0.10, 0.10, 101))

    thresholds = MLFactorClassThresholds.fit(train_targets, target_col=_TARGET_COL)
    expected = np.quantile(train_targets.to_numpy(), CLASS_PERCENTILES)

    assert thresholds.values == pytest.approx(tuple(expected))
    assert [label.value for label in MLFactorClass] == [0, 1, 2, 3, 4]
    assert thresholds.transform(train_targets).min() == MLFactorClass.STRONG_BEAR
    assert thresholds.transform(train_targets).max() == MLFactorClass.STRONG_BULL


def test_validation_data_does_not_change_fitted_thresholds():
    train_targets = pd.Series(np.linspace(-1.0, 1.0, 100))
    validation_targets = pd.Series([10_000.0] * 20)

    thresholds = MLFactorClassThresholds.fit(train_targets, target_col=_TARGET_COL)
    leaked = MLFactorClassThresholds.fit(
        pd.concat([train_targets, validation_targets], ignore_index=True),
        target_col=_TARGET_COL,
    )

    assert thresholds.values == pytest.approx(tuple(np.quantile(train_targets, CLASS_PERCENTILES)))
    assert thresholds.weak_bull_max != pytest.approx(leaked.weak_bull_max)


def test_strategy_thresholds_ignore_prediction_date_target(ml_df):
    last_date = ml_df["date"].max()
    leaky_df = ml_df.with_columns(
        pl.when(pl.col("date") == last_date)
        .then(10_000.0)
        .otherwise(pl.col(_TARGET_COL))
        .alias(_TARGET_COL)
    )
    captured: dict[str, object] = {}

    def train_func(train: pd.DataFrame, predict: pd.DataFrame) -> np.ndarray:
        captured["train_max_date"] = train["date"].max()
        captured["thresholds"] = MLFactorClassThresholds.fit(
            train[_TARGET_COL],
            target_col=_TARGET_COL,
        )
        return np.linspace(-1.0, 1.0, len(predict))

    def portfolio_func(predictions: pd.Series, history_df: pd.DataFrame | None = None):
        captured["history_max_date"] = history_df["date"].max() if history_df is not None else None
        return {
            str(predictions.idxmax()): 1.0,
            str(predictions.idxmin()): -1.0,
        }

    strategy = MLFactorStrategy(_PREDICTOR_COLS, _TARGET_COL, train_func, portfolio_func)

    result = strategy.generate_signals(leaky_df)

    thresholds = captured["thresholds"]
    assert isinstance(thresholds, MLFactorClassThresholds)
    assert thresholds.weak_bull_max < 1.0
    assert captured["train_max_date"].date() < last_date
    assert captured["history_max_date"].date() < last_date
    assert not result.is_empty()


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
    result = strategy.generate_signals(ml_df.drop(_TARGET_COL))
    assert result.is_empty()
    assert set(result.columns) == _SIGNAL_COLS


def test_generate_signals_missing_predictors_empty(ml_df):
    strategy_bad = MLFactorStrategy(
        ["missing1", "missing2"],
        _TARGET_COL,
        lambda train, predict: np.array([], dtype=float),
        lambda predictions, history_df=None: {},
    )
    result = strategy_bad.generate_signals(ml_df)
    assert result.is_empty()
    assert set(result.columns) == _SIGNAL_COLS


def test_predictor_target_leakage_rejected():
    with pytest.raises(ValueError, match="future target"):
        MLFactorStrategy(
            ["f1", _TARGET_COL],
            _TARGET_COL,
            lambda train, predict: np.array([], dtype=float),
            lambda predictions, history_df=None: {},
        )


def test_class_scores_from_probabilities_shape_and_classes():
    probabilities = np.eye(5)

    scores = class_scores_from_probabilities(probabilities)
    metrics = classification_metrics(np.arange(5), np.arange(5))

    assert scores.shape == (5,)
    assert scores.tolist() == pytest.approx(ML_FACTOR_CLASS_SCORES.tolist())
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)


def test_xgb_classifier_output_shape_and_score_range():
    pytest.importorskip("xgboost")
    rows = []
    for i, target in enumerate(np.linspace(-0.20, 0.20, 120)):
        rows.append(
            {
                "date": date(2023, 1, 1) + timedelta(days=i),
                "symbol": f"S{i % 6}",
                "f1": float(target),
                "f2": float(target**2),
                "f3": float(i % 3),
                _TARGET_COL: float(target),
            }
        )
    train = pd.DataFrame(rows)
    predict = train.tail(5).drop(columns=[_TARGET_COL]).copy()

    scores = train_and_predict_xgb_classifier(
        train,
        predict,
        predictor_cols=_PREDICTOR_COLS,
        target_col=_TARGET_COL,
        model_params={"n_estimators": 5, "max_depth": 2, "random_state": 7, "n_jobs": 1},
    )

    assert scores.shape == (5,)
    assert np.isfinite(scores).all()
    assert (scores >= ML_FACTOR_CLASS_SCORES.min()).all()
    assert (scores <= ML_FACTOR_CLASS_SCORES.max()).all()


def test_xgb_classifier_handles_missing_training_classes():
    pytest.importorskip("xgboost")
    targets = [-0.20] * 30 + [0.20] * 30
    train = pd.DataFrame(
        {
            "date": [date(2023, 1, 1) + timedelta(days=i) for i in range(len(targets))],
            "symbol": [f"S{i % 3}" for i in range(len(targets))],
            "f1": targets,
            _TARGET_COL: targets,
        }
    )
    predict = train.tail(3).drop(columns=[_TARGET_COL]).copy()

    scores = train_and_predict_xgb_classifier(
        train,
        predict,
        predictor_cols=["f1"],
        target_col=_TARGET_COL,
        model_params={"n_estimators": 2, "random_state": 7, "n_jobs": 1},
    )

    assert scores.shape == (3,)
    assert np.isfinite(scores).all()


def test_ml_factor_registered():
    cls = Registry.get_strategy("ml_factor")
    assert cls is MLFactorStrategy
