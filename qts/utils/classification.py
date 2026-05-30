"""Shared classification helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qts.utils.metrics import normalize_class_probabilities
from qts.utils.numeric import finite_numeric


@dataclass(frozen=True, slots=True)
class QuantileClassThresholds:
    """Quantile thresholds for ordered numeric class labels."""

    target_col: str
    thresholds: tuple[float, ...]

    @classmethod
    def fit(
        cls,
        values: Sequence[float] | pd.Series | np.ndarray,
        *,
        target_col: str,
        percentiles: Sequence[float],
    ) -> QuantileClassThresholds:
        clean = finite_numeric(values)
        if clean.size == 0:
            raise ValueError("Cannot fit class thresholds without finite target values")
        percentile_values = tuple(float(value) for value in percentiles)
        thresholds = tuple(float(value) for value in np.quantile(clean, percentile_values))
        return cls(target_col=target_col, thresholds=thresholds)

    def transform(self, values: Sequence[float] | pd.Series | np.ndarray) -> np.ndarray:
        numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        labels = np.full(numeric.shape[0], -1, dtype=np.int16)
        finite = np.isfinite(numeric)
        labels[finite] = np.searchsorted(np.array(self.thresholds), numeric[finite], side="left")
        return labels

    def append_labels(self, frame: pd.DataFrame, *, label_col: str) -> pd.DataFrame:
        result = frame.copy()
        result[label_col] = self.transform(result[self.target_col])
        return result[result[label_col] >= 0].copy()


def align_class_probabilities(
    probabilities: np.ndarray,
    classes: Sequence[int],
    *,
    num_classes: int,
) -> np.ndarray:
    raw = np.asarray(probabilities, dtype=float)
    if raw.ndim != 2:
        raise ValueError("Classifier probabilities must be a 2D array")
    if raw.shape[1] == num_classes:
        return normalize_class_probabilities(raw, num_classes=num_classes)
    aligned = np.zeros((raw.shape[0], num_classes), dtype=float)
    for index, label in enumerate(classes):
        class_index = int(label)
        if 0 <= class_index < num_classes and index < raw.shape[1]:
            aligned[:, class_index] = raw[:, index]
    return normalize_class_probabilities(aligned, num_classes=num_classes)


def class_scores_from_probabilities(
    probabilities: np.ndarray,
    class_scores: Sequence[float] | np.ndarray,
) -> np.ndarray:
    scores = np.asarray(class_scores, dtype=float)
    normalized = normalize_class_probabilities(probabilities, num_classes=len(scores))
    return normalized @ scores


__all__ = [
    "QuantileClassThresholds",
    "align_class_probabilities",
    "class_scores_from_probabilities",
]
