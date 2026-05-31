"""Registered ML factor training functions."""

from __future__ import annotations

from typing import Any

import numpy as np

from .constants import (
    drop_model_nulls,
    min_int,
    non_negative_int,
    optional_positive_int,
    time_series_cv_splits,
    to_pandas_frame,
    validate_predictor_columns,
    validate_target_col,
)

_LEGACY_REGRESSION_TRAINERS = {"xgb_regressor", "xgb_ranker", "ic_composite"}


def train_and_predict_xgb_classifier(
    train_data: object,
    predict_data: object,
    predictor_cols: list[str],
    target_col: str,
    model_params: dict[str, Any] | None = None,
    cv_splits: int = 5,
    cv_gap: int = 0,
    cv_test_size: int | None = None,
    cv_max_train_size: int | None = None,
) -> np.ndarray:
    """Fit an XGBoost multiclass classifier and return signed class scores."""

    validate_target_col(target_col)
    validate_predictor_columns(predictor_cols, target_col)
    train = to_pandas_frame(train_data, label="train_data")
    predict = to_pandas_frame(predict_data, label="predict_data")
    missing = [column for column in predictor_cols if column not in train or column not in predict]
    if missing:
        raise ValueError(f"Missing ML factor predictor columns: {sorted(set(missing))}")

    train = drop_model_nulls(train, [*predictor_cols, target_col])
    predict = drop_model_nulls(predict, predictor_cols)
    if train.empty or predict.empty:
        return np.array([], dtype=float)
    splits = time_series_cv_splits(
        train,
        n_splits=min_int(cv_splits, "cv_splits", minimum=2),
        gap=non_negative_int(cv_gap, "cv_gap"),
        test_size=optional_positive_int(cv_test_size, "cv_test_size"),
        max_train_size=optional_positive_int(cv_max_train_size, "cv_max_train_size"),
    )
    if not splits:
        return np.array([], dtype=float)

    from qts.research.strategies.ml_factor.models import XGBClassifierModel

    fold_scores = []
    for train_fold, test_fold in splits:
        if train_fold.empty or test_fold.empty:
            continue
        model = XGBClassifierModel(task="classification", **(model_params or {}))
        model.fit(train_fold[predictor_cols], train_fold[target_col])
        fold_scores.append(model.predict(predict[predictor_cols]))
    if not fold_scores:
        return np.array([], dtype=float)
    return np.mean(np.vstack(fold_scores), axis=0)


__all__ = ["_LEGACY_REGRESSION_TRAINERS", "train_and_predict_xgb_classifier"]
