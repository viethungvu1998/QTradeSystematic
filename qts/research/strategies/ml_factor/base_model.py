"""Abstract base class for all pluggable ML models in QTS."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Uniform fit/predict contract."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train on in-sample data. Mutates self in place."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return per-row float scores."""

    def feature_importances(self) -> dict[str, float] | None:
        """Optional feature importances."""
        return None
