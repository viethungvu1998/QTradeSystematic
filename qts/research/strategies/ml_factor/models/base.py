"""Base interface for pluggable ML factor models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseMLFactorModel(ABC):
    """Uniform fit/predict contract for ML factor scoring models."""

    @abstractmethod
    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Train on in-sample data. Mutates self in place."""

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return per-row float scores."""

    def feature_importances(self) -> dict[str, float] | None:
        """Optional feature importances."""
        return None


__all__ = ["BaseMLFactorModel"]
