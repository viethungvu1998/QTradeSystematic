"""Reusable label schemes and probability scoring helpers."""

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

CLASS_PERCENTILES = (0.15, 0.45, 0.55, 0.85)
CLASS_LABEL_COLUMN = "ml_factor_class"
ML_FACTOR_CLASS_COUNT = 5
ML_FACTOR_CLASS_SCORES = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)


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
        q1, q2, q3, q4 = thresholds.thresholds
        return cls(
            target_col=target_col,
            strong_bear_max=float(q1),
            weak_bear_max=float(q2),
            no_change_max=float(q3),
            weak_bull_max=float(q4),
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


def class_labels_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Decode five-class probabilities into ordinal class labels.

    This uses the probability-weighted class score instead of nominal argmax,
    which reduces the tendency to collapse adjacent uncertainty into classes 1
    and 3. Exact midpoint ties break toward the less-extreme class.
    """

    scores = class_scores_from_probabilities(probabilities)
    distances = np.abs(scores[:, None] - ML_FACTOR_CLASS_SCORES[None, :])
    centrality_penalty = np.abs(ML_FACTOR_CLASS_SCORES)[None, :] * 1e-12
    return np.argmin(distances + centrality_penalty, axis=1).astype(np.int16)


__all__ = [
    "CLASS_LABEL_COLUMN",
    "CLASS_PERCENTILES",
    "ML_FACTOR_CLASS_COUNT",
    "ML_FACTOR_CLASS_SCORES",
    "MLFactorClass",
    "MLFactorClassThresholds",
    "align_class_probabilities",
    "class_labels_from_probabilities",
    "class_scores_from_probabilities",
    "normalize_probabilities",
]
