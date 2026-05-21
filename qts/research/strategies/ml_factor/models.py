"""Concrete ML model implementations — all registered with the model registry."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qts.core.registry import Registry
from qts.research.strategies.ml_factor.base_model import BaseModel


def fit_xgb_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    model_params: dict | None = None,
):
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for XGBoost regression models."
        ) from exc
    model = xgb.XGBRegressor(**(model_params or {}))
    model.fit(X, y)
    return model


def fit_xgb_ranker(
    X: pd.DataFrame,
    y: pd.Series,
    qid: np.ndarray,
    model_params: dict | None = None,
):
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("xgboost is required for XGBoost ranker models.") from exc
    model = xgb.XGBRanker(**(model_params or {}))
    model.fit(X, y, qid=qid)
    return model


@Registry.register_model("xgb_classifier")
class XGBClassifierModel(BaseModel):
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

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for XGBClassifierModel. "
                "Install with: pip install xgboost"
            ) from exc

        from qts.research.strategies.ml_factor.model import (
            CLASS_LABEL_COLUMN,
            MLFactorClass,
            MLFactorClassThresholds,
        )

        target_col = str(y.name or "target")
        thresholds = MLFactorClassThresholds.fit(y, target_col=target_col)
        labelled = thresholds.append_labels(pd.DataFrame({target_col: y}))
        X_fit = X.loc[labelled.index]
        y_labelled = labelled[CLASS_LABEL_COLUMN].to_numpy(dtype=int)

        self._feature_names = list(X.columns)
        self._class_labels = np.array(sorted(np.unique(y_labelled)), dtype=int)
        self._constant_score = None
        if self._class_labels.size < 2:
            self._model = None
            self._constant_score = MLFactorClass(int(y_labelled[0])).score
            return

        encoded_y = np.searchsorted(self._class_labels, y_labelled)
        params = {
            "objective": "multi:softprob",
            "num_class": int(self._class_labels.size),
            "eval_metric": "mlogloss",
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": 42,
            "n_jobs": -1,
        }
        params.update(self.xgb_params)
        params["objective"] = "multi:softprob"
        params["num_class"] = int(self._class_labels.size)
        self._model = xgb.XGBClassifier(**params)
        self._model.fit(X_fit, encoded_y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._constant_score is not None:
            return np.full(len(X), self._constant_score, dtype=float)
        if self._model is None or self._class_labels is None:
            raise RuntimeError("XGBClassifierModel.fit() must be called before predict().")

        from qts.research.strategies.ml_factor.model import class_scores_from_probabilities

        probabilities = self._model.predict_proba(X)
        aligned = np.zeros((len(X), 5), dtype=float)
        for index, label in enumerate(self._class_labels):
            if 0 <= int(label) < aligned.shape[1] and index < probabilities.shape[1]:
                aligned[:, int(label)] = probabilities[:, index]
        return class_scores_from_probabilities(aligned)

    def feature_importances(self) -> dict[str, float] | None:
        if self._model is None:
            return None
        return dict(zip(self._feature_names, self._model.feature_importances_))


@Registry.register_model("xgb_regressor")
class XGBRegressorModel(BaseModel):
    """XGBoost regressor for continuous return prediction."""

    def __init__(self, task: str = "regression", **xgb_params) -> None:
        self.task = task
        model_params = xgb_params.pop("model_params", None)
        if isinstance(model_params, dict):
            xgb_params.update(model_params)
        self.xgb_params = xgb_params
        self._model = None
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("xgboost is required for XGBRegressorModel.") from exc
        self._feature_names = list(X.columns)
        self._model = xgb.XGBRegressor(**self.xgb_params)
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("XGBRegressorModel.fit() must be called before predict().")
        return self._model.predict(X)

    def feature_importances(self) -> dict[str, float] | None:
        if self._model is None:
            return None
        return dict(zip(self._feature_names, self._model.feature_importances_))


@Registry.register_model("linear")
class LinearRegressionModel(BaseModel):
    """Ridge regression model for factor scoring."""

    def __init__(self, task: str = "regression", alpha: float = 1.0, **kwargs) -> None:
        self.task = task
        self.alpha = alpha
        self._model = None
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        from sklearn.linear_model import Ridge  # type: ignore[import]

        self._feature_names = list(X.columns)
        self._model = Ridge(alpha=self.alpha)
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("LinearRegressionModel.fit() must be called before predict().")
        return self._model.predict(X)

    def feature_importances(self) -> dict[str, float] | None:
        if self._model is None:
            return None
        return dict(zip(self._feature_names, self._model.coef_.tolist()))


@Registry.register_model("ic_composite")
class ICCompositeModel(BaseModel):
    """IC-weighted composite scorer."""

    def __init__(self, task: str = "regression", ic_window: int = 0, **kwargs) -> None:
        self.task = task
        self.ic_window = ic_window
        self._weights: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        tail = X.tail(self.ic_window) if self.ic_window > 0 else X
        y_tail = y.iloc[-len(tail):]
        ic_values = {
            col: tail[col].corr(y_tail, method="spearman")
            for col in tail.columns
        }
        weights = pd.Series(ic_values).fillna(0.0)
        total = weights.abs().sum()
        self._weights = weights / total if total > 0 else weights * 0 + 1.0 / len(weights)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("ICCompositeModel.fit() must be called before predict().")
        aligned = X.reindex(columns=self._weights.index, fill_value=0.0)
        return (aligned * self._weights).sum(axis=1).to_numpy()
