"""XGBoost ML factor models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from qts.core.registry import Registry
from qts.research.strategies.ml_factor.constants import (
    CLASS_LABEL_COLUMN,
    FORBIDDEN_PREDICTOR_COLUMNS,
    FORBIDDEN_PREDICTOR_PREFIXES,
    TARGET_PREFIXES,
)
from qts.research.strategies.ml_factor.labels import (
    MLFactorClass,
    MLFactorClassThresholds,
    align_class_probabilities,
    class_scores_from_probabilities,
)
from qts.utils.dataframe import drop_non_numeric_nulls, to_pandas_frame
from qts.utils.time_series import time_series_cv_splits
from qts.utils.validation import (
    min_int,
    non_negative_int,
    optional_positive_int,
    validate_predictor_columns,
    validate_target_col,
)

from .base import BaseMLFactorModel


def fit_xgb_ranker(
    features: pd.DataFrame,
    target: pd.Series,
    qid: np.ndarray,
    model_params: dict | None = None,
):
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("xgboost is required for XGBoost ranker models.") from exc
    model = xgb.XGBRanker(**(model_params or {}))
    model.fit(features, target, qid=qid)
    return model


def fit_xgb_regressor(
    features: pd.DataFrame,
    target: pd.Series,
    model_params: dict | None = None,
):
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("xgboost is required for XGBoost regression models.") from exc
    model = xgb.XGBRegressor(**(model_params or {}))
    model.fit(features, target)
    return model


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

    validate_target_col(
        target_col,
        TARGET_PREFIXES,
        message="MLFactorStrategy target_col must be a future percentage-change column",
    )
    validate_predictor_columns(
        predictor_cols,
        target_col,
        forbidden_columns=FORBIDDEN_PREDICTOR_COLUMNS,
        forbidden_prefixes=FORBIDDEN_PREDICTOR_PREFIXES,
        message="ML factor predictors cannot include future target columns",
    )
    train = to_pandas_frame(train_data, label="train_data")
    predict = to_pandas_frame(predict_data, label="predict_data")
    missing = [column for column in predictor_cols if column not in train or column not in predict]
    if missing:
        raise ValueError(f"Missing ML factor predictor columns: {sorted(set(missing))}")

    train = drop_non_numeric_nulls(train, [*predictor_cols, target_col])
    predict = drop_non_numeric_nulls(predict, predictor_cols)
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


@Registry.register_model("xgb_classifier")
class XGBClassifierModel(BaseMLFactorModel):
    """XGBoost classifier for ML factor scoring."""

    def __init__(self, task: str = "classification", **xgb_params) -> None:
        self.task = task
        model_params = xgb_params.pop("model_params", None)
        if isinstance(model_params, dict):
            xgb_params.update(model_params)
        self.xgb_params = xgb_params
        self._model = None
        self._feature_names: list[str] = []
        self._class_labels: np.ndarray | None = None
        self._constant_score: float | None = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for XGBClassifierModel. Install with: pip install xgboost"
            ) from exc

        target_col = str(target.name or "target")
        thresholds = MLFactorClassThresholds.fit(target, target_col=target_col)
        labelled = thresholds.append_labels(pd.DataFrame({target_col: target}))
        features_fit = features.loc[labelled.index]
        y_labelled = labelled[CLASS_LABEL_COLUMN].to_numpy(dtype=int)

        self._feature_names = list(features.columns)
        self._class_labels = np.array(sorted(np.unique(y_labelled)), dtype=int)
        self._constant_score = None
        if self._class_labels.size < 2:
            self._model = None
            self._constant_score = MLFactorClass(int(y_labelled[0])).score
            return

        encoded_y = np.searchsorted(self._class_labels, y_labelled)
        params = dict(self.xgb_params)
        params["objective"] = "multi:softprob"
        params["num_class"] = int(self._class_labels.size)
        self._model = xgb.XGBClassifier(**params)
        self._model.fit(features_fit, encoded_y)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self._constant_score is not None:
            return np.full(len(features), self._constant_score, dtype=float)
        if self._model is None or self._class_labels is None:
            raise RuntimeError("XGBClassifierModel.fit() must be called before predict().")

        probabilities = self._model.predict_proba(features)
        aligned = align_class_probabilities(probabilities, self._class_labels)
        return class_scores_from_probabilities(aligned)

    def feature_importances(self) -> dict[str, float] | None:
        if self._model is None:
            return None
        return dict(zip(self._feature_names, self._model.feature_importances_, strict=True))


@Registry.register_model("xgb_regressor")
class XGBRegressorModel(BaseMLFactorModel):
    """XGBoost regressor for continuous return prediction."""

    def __init__(self, task: str = "regression", **xgb_params) -> None:
        self.task = task
        model_params = xgb_params.pop("model_params", None)
        if isinstance(model_params, dict):
            xgb_params.update(model_params)
        self.xgb_params = xgb_params
        self._model = None
        self._feature_names: list[str] = []

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        self._feature_names = list(features.columns)
        self._model = fit_xgb_regressor(features, target, self.xgb_params)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("XGBRegressorModel.fit() must be called before predict().")
        return np.asarray(self._model.predict(features), dtype=float)

    def feature_importances(self) -> dict[str, float] | None:
        if self._model is None:
            return None
        return dict(zip(self._feature_names, self._model.feature_importances_, strict=True))


__all__ = [
    "XGBClassifierModel",
    "XGBRegressorModel",
    "fit_xgb_ranker",
    "fit_xgb_regressor",
    "train_and_predict_xgb_classifier",
]
