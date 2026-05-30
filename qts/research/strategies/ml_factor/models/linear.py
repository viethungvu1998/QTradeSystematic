"""Linear ML factor models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qts.core.registry import Registry

from .base import BaseMLFactorModel


@Registry.register_model("linear")
class LinearRegressionModel(BaseMLFactorModel):
    """Ridge regression model for factor scoring."""

    def __init__(self, task: str = "regression", alpha: float = 1.0, **kwargs) -> None:
        self.task = task
        self.alpha = alpha
        self._model = None
        self._feature_names: list[str] = []

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        from sklearn.linear_model import Ridge

        self._feature_names = list(features.columns)
        self._model = Ridge(alpha=self.alpha)
        self._model.fit(features, target)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("LinearRegressionModel.fit() must be called before predict().")
        return np.asarray(self._model.predict(features), dtype=float)

    def feature_importances(self) -> dict[str, float] | None:
        if self._model is None:
            return None
        return dict(zip(self._feature_names, self._model.coef_.tolist(), strict=True))


__all__ = ["LinearRegressionModel"]
