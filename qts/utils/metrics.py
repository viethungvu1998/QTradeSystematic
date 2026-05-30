"""Shared metric helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def multiclass_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    num_classes: int,
) -> dict[str, float]:
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    valid = (
        (true >= 0)
        & (true < num_classes)
        & (pred >= 0)
        & (pred < num_classes)
    )
    if not valid.any():
        return {}
    true = true[valid]
    pred = pred[valid]
    per_class: list[tuple[float, float, float]] = []
    for label in range(num_classes):
        true_positive = float(((pred == label) & (true == label)).sum())
        false_positive = float(((pred == label) & (true != label)).sum())
        false_negative = float(((pred != label) & (true == label)).sum())
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append((precision, recall, f1))
    return {
        "accuracy": float((true == pred).mean()),
        "macro_precision": float(np.mean([item[0] for item in per_class])),
        "macro_recall": float(np.mean([item[1] for item in per_class])),
        "macro_f1": float(np.mean([item[2] for item in per_class])),
    }


def multiclass_classification_metrics_from_probabilities(
    y_true: Sequence[int] | np.ndarray,
    probabilities: np.ndarray,
    *,
    num_classes: int,
) -> dict[str, float]:
    normalized = normalize_class_probabilities(probabilities, num_classes=num_classes)
    y_pred = normalized.argmax(axis=1)
    metrics = multiclass_classification_metrics(
        y_true,
        y_pred,
        num_classes=num_classes,
    )
    y = np.asarray(y_true, dtype=int)
    valid = (y >= 0) & (y < num_classes)
    if valid.any():
        clipped = np.clip(normalized[valid], 1e-15, 1.0)
        metrics["log_loss"] = float(-np.log(clipped[np.arange(valid.sum()), y[valid]]).mean())
    return metrics


def normalize_class_probabilities(
    probabilities: np.ndarray,
    *,
    num_classes: int,
) -> np.ndarray:
    raw = np.asarray(probabilities, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != num_classes:
        raise ValueError(f"Expected probabilities with shape (n, {num_classes})")
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = raw.sum(axis=1, keepdims=True)
    fallback = np.full_like(raw, 1.0 / num_classes)
    return np.divide(raw, row_sums, out=fallback, where=row_sums != 0.0)


__all__ = [
    "multiclass_classification_metrics",
    "multiclass_classification_metrics_from_probabilities",
    "normalize_class_probabilities",
]
