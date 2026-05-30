"""Constants for ML factor strategies."""

from __future__ import annotations

import numpy as np

from qts.research.strategies.factor.core import FORWARD_RETURN_PREFIX

CLASS_PERCENTILES = (0.15, 0.40, 0.60, 0.85)
CLASS_LABEL_COLUMN = "ml_factor_class"
ML_FACTOR_CLASS_COUNT = 5
ML_FACTOR_CLASS_SCORES = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)

LEGACY_REGRESSION_TRAINERS = frozenset({"xgb_regressor", "xgb_ranker", "ic_composite"})
TARGET_PREFIXES = (FORWARD_RETURN_PREFIX, "future_change_pct")
FORBIDDEN_PREDICTOR_PREFIXES = (FORWARD_RETURN_PREFIX, "future_", "target_")
FORBIDDEN_PREDICTOR_COLUMNS = frozenset(
    {
        CLASS_LABEL_COLUMN,
        "class_label",
        "label",
        "target",
        "future_change_pct",
    }
)

__all__ = [
    "CLASS_LABEL_COLUMN",
    "CLASS_PERCENTILES",
    "FORBIDDEN_PREDICTOR_COLUMNS",
    "FORBIDDEN_PREDICTOR_PREFIXES",
    "LEGACY_REGRESSION_TRAINERS",
    "ML_FACTOR_CLASS_COUNT",
    "ML_FACTOR_CLASS_SCORES",
    "TARGET_PREFIXES",
]
