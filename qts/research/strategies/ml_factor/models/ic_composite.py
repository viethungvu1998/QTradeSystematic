"""IC-weighted composite ML factor model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qts.core.registry import Registry

from .base import BaseMLFactorModel


@Registry.register_model("ic_composite")
class ICCompositeModel(BaseMLFactorModel):
    """IC-weighted composite scorer."""

    def __init__(self, task: str = "regression", ic_window: int = 0, **kwargs) -> None:
        self.task = task
        self.ic_window = ic_window
        self._weights: pd.Series | None = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        tail = features.tail(self.ic_window) if self.ic_window > 0 else features
        target_tail = target.iloc[-len(tail) :]
        ic_values = {
            column: tail[column].corr(target_tail, method="spearman") for column in tail.columns
        }
        weights = pd.Series(ic_values).fillna(0.0)
        total = weights.abs().sum()
        self._weights = weights / total if total > 0 else weights * 0 + 1.0 / len(weights)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("ICCompositeModel.fit() must be called before predict().")
        aligned = features.reindex(columns=self._weights.index, fill_value=0.0)
        return aligned.mul(self._weights, axis=1).sum(axis=1).to_numpy(dtype=float)


__all__ = ["ICCompositeModel"]
