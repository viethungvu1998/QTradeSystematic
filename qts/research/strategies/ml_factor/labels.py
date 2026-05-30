"""ML factor class labels and probability scoring."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd

from qts.utils.classification import (
    QuantileClassThresholds,
)
from qts.utils.classification import (
    align_class_probabilities as align_probabilities_to_classes,
)
from qts.utils.classification import (
    class_scores_from_probabilities as probability_weighted_class_scores,
)
from qts.utils.metrics import normalize_class_probabilities

from .constants import (
    CLASS_LABEL_COLUMN,
    CLASS_PERCENTILES,
    ML_FACTOR_CLASS_COUNT,
    ML_FACTOR_CLASS_SCORES,
)


class MLFactorClass(IntEnum):
    """Percentile class labels for future percentage change."""

    STRONG_BEAR = 0
    WEAK_BEAR = 1
    NO_CHANGE = 2
    WEAK_BULL = 3
    STRONG_BULL = 4

    @property
    def score(self) -> float:
        return float(ML_FACTOR_CLASS_SCORES[int(self)])


@dataclass(frozen=True, slots=True)
class MLFactorClassThresholds:
    """Training-window percentile thresholds for ML factor classes."""

    target_col: str
    strong_bear_max: float
    weak_bear_max: float
    no_change_max: float
    weak_bull_max: float

    @classmethod
    def fit(
        cls,
        values: Sequence[float] | pd.Series | np.ndarray,
        *,
        target_col: str,
        percentiles: Sequence[float] = CLASS_PERCENTILES,
    ) -> MLFactorClassThresholds:
        pct = tuple(float(value) for value in percentiles)
        if pct != CLASS_PERCENTILES:
            raise ValueError(f"ML factor class percentiles must be {CLASS_PERCENTILES}")
        thresholds = QuantileClassThresholds.fit(
            values,
            target_col=target_col,
            percentiles=pct,
        )
        q15, q40, q60, q85 = thresholds.thresholds
        return cls(
            target_col=target_col,
            strong_bear_max=float(q15),
            weak_bear_max=float(q40),
            no_change_max=float(q60),
            weak_bull_max=float(q85),
        )

    @property
    def values(self) -> tuple[float, float, float, float]:
        return (
            self.strong_bear_max,
            self.weak_bear_max,
            self.no_change_max,
            self.weak_bull_max,
        )

    def transform(self, values: Sequence[float] | pd.Series | np.ndarray) -> np.ndarray:
        thresholds = QuantileClassThresholds(
            target_col=self.target_col,
            thresholds=self.values,
        )
        return thresholds.transform(values)

    def append_labels(
        self,
        frame: pd.DataFrame,
        *,
        label_col: str = CLASS_LABEL_COLUMN,
    ) -> pd.DataFrame:
        thresholds = QuantileClassThresholds(
            target_col=self.target_col,
            thresholds=self.values,
        )
        return thresholds.append_labels(frame, label_col=label_col)


def align_class_probabilities(probabilities: np.ndarray, classes: Sequence[int]) -> np.ndarray:
    return align_probabilities_to_classes(
        probabilities,
        classes,
        num_classes=ML_FACTOR_CLASS_COUNT,
    )


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return normalize_class_probabilities(
        probabilities,
        num_classes=ML_FACTOR_CLASS_COUNT,
    )


def class_scores_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Convert five-class probabilities to signed portfolio ranking scores."""

    return probability_weighted_class_scores(probabilities, ML_FACTOR_CLASS_SCORES)


__all__ = [
    "CLASS_LABEL_COLUMN",
    "CLASS_PERCENTILES",
    "ML_FACTOR_CLASS_COUNT",
    "ML_FACTOR_CLASS_SCORES",
    "MLFactorClass",
    "MLFactorClassThresholds",
    "align_class_probabilities",
    "class_scores_from_probabilities",
    "normalize_probabilities",
]
