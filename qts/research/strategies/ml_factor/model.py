"""Classification-based ML factor strategy."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from qts.core.registry import Registry
from qts.research.strategies.factor.base import BaseFactorStrategy
from qts.research.strategies.factor.core import FORWARD_RETURN_PREFIX

CLASS_PERCENTILES = (0.15, 0.40, 0.60, 0.85)
CLASS_LABEL_COLUMN = "ml_factor_class"
ML_FACTOR_CLASS_COUNT = 5
ML_FACTOR_CLASS_SCORES = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)
_TARGET_PREFIXES = (FORWARD_RETURN_PREFIX, "future_change_pct")
_FORBIDDEN_PREDICTOR_PREFIXES = (FORWARD_RETURN_PREFIX, "future_", "target_")
_FORBIDDEN_PREDICTOR_COLUMNS = {
    CLASS_LABEL_COLUMN,
    "class_label",
    "label",
    "target",
    "future_change_pct",
}
_LEGACY_REGRESSION_TRAINERS = {"xgb_regressor", "xgb_ranker", "ic_composite"}


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
        clean = _finite_numeric(values)
        if clean.size == 0:
            raise ValueError("Cannot fit ML factor class thresholds without finite target values")
        pct = tuple(float(value) for value in percentiles)
        if pct != CLASS_PERCENTILES:
            raise ValueError(f"ML factor class percentiles must be {CLASS_PERCENTILES}")
        q15, q40, q60, q85 = np.quantile(clean, pct)
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
        numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        labels = np.full(numeric.shape[0], -1, dtype=np.int16)
        finite = np.isfinite(numeric)
        labels[finite] = np.searchsorted(np.array(self.values), numeric[finite], side="left")
        return labels

    def append_labels(
        self,
        frame: pd.DataFrame,
        *,
        label_col: str = CLASS_LABEL_COLUMN,
    ) -> pd.DataFrame:
        result = frame.copy()
        result[label_col] = self.transform(result[self.target_col])
        return result[result[label_col] >= 0].copy()


@Registry.register_strategy("ml_factor")
class MLFactorStrategy(BaseFactorStrategy):
    """Factor strategy that converts future returns into five classification labels."""

    def __init__(
        self,
        predictor_cols: list[str],
        target_col: str,
        train_func: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
        portfolio_func: Callable[..., dict[str, float]],
        rebalance_period: int = 10,
        min_train_rows: int = 2,
    ) -> None:
        _validate_target_col(target_col)
        _validate_predictor_columns(predictor_cols, target_col)
        self.predictor_cols = list(predictor_cols)
        self.target_col = target_col
        self.train_func = train_func
        self.portfolio_func = portfolio_func
        self.rebalance_period = _positive_int(rebalance_period, "rebalance_period")
        self.min_train_rows = int(min_train_rows)
        self.last_thresholds: MLFactorClassThresholds | None = None

    @classmethod
    def from_config_params(cls, params: Mapping[str, object]) -> MLFactorStrategy:
        payload = dict(params)
        predictor_cols = [str(item) for item in payload.pop("predictor_cols")]
        target_col = str(payload.pop("target_col"))
        rebalance_period = _positive_int(
            payload.pop("rebalance_period", payload.pop("rebalance_frequency", 10)),
            "rebalance_period",
        )
        min_train_rows = int(payload.pop("min_train_rows", 2))
        trainer = _named_section(payload.pop("trainer", {"name": "xgb_classifier"}), "trainer")
        portfolio = _named_section(payload.pop("portfolio"), "portfolio")

        trainer_name = str(trainer["name"])
        if trainer_name in _LEGACY_REGRESSION_TRAINERS:
            raise ValueError(
                f"MLFactorStrategy requires a classification trainer, got {trainer_name}"
            )
        trainer_params = dict(trainer.get("params", {}))
        trainer_params.setdefault("predictor_cols", predictor_cols)
        trainer_params.setdefault("target_col", target_col)
        train_func = partial(Registry.get_factor_trainer(trainer_name), **trainer_params)

        portfolio_func = partial(
            Registry.get_portfolio_constructor(str(portfolio["name"])),
            **dict(portfolio.get("params", {})),
        )
        return cls(
            predictor_cols,
            target_col,
            train_func,
            portfolio_func,
            rebalance_period,
            min_train_rows,
        )

    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return self.empty_signal_frame()
        if self.target_col not in df.columns:
            return self.empty_signal_frame()
        if any(column not in df.columns for column in self.predictor_cols):
            return self.empty_signal_frame()

        df_pd = df.sort(["date", "symbol"]).to_pandas()
        last_date = df_pd["date"].max()
        train_pd = df_pd[(df_pd["date"] < last_date) & df_pd[self.target_col].notna()].copy()
        predict_pd = df_pd[df_pd["date"] == last_date].copy()
        train_pd = _drop_model_nulls(train_pd, [*self.predictor_cols, self.target_col])
        predict_pd = _drop_model_nulls(predict_pd, self.predictor_cols)
        if len(train_pd) < self.min_train_rows or predict_pd.empty:
            return self.empty_signal_frame()

        self.last_thresholds = MLFactorClassThresholds.fit(
            train_pd[self.target_col],
            target_col=self.target_col,
        )
        scores = np.asarray(self.train_func(train_pd, predict_pd), dtype=float)
        if scores.shape[0] != len(predict_pd):
            raise ValueError("ML factor trainer output must align to predict rows")

        predictions = pd.Series(scores, index=predict_pd["symbol"].values)
        weights_dict = self.portfolio_func(predictions, history_df=train_pd)
        if not weights_dict:
            return self.empty_signal_frame()
        return self.signal_frame_from_weights(last_date, weights_dict)


def train_and_predict_xgb_classifier(
    train_data: object,
    predict_data: object,
    predictor_cols: list[str],
    target_col: str,
    model_params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Fit an XGBoost multiclass classifier and return signed class scores."""

    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("xgboost is not installed: pip install xgboost") from exc

    _validate_target_col(target_col)
    _validate_predictor_columns(predictor_cols, target_col)
    train = _to_pandas(train_data, label="train_data")
    predict = _to_pandas(predict_data, label="predict_data")
    missing = [column for column in predictor_cols if column not in train or column not in predict]
    if missing:
        raise ValueError(f"Missing ML factor predictor columns: {sorted(set(missing))}")

    train = _drop_model_nulls(train, [*predictor_cols, target_col])
    predict = _drop_model_nulls(predict, predictor_cols)
    if train.empty or predict.empty:
        return np.array([], dtype=float)

    thresholds = MLFactorClassThresholds.fit(train[target_col], target_col=target_col)
    labelled = thresholds.append_labels(train)
    y_train = labelled[CLASS_LABEL_COLUMN].to_numpy(dtype=int)
    class_labels = np.array(sorted(np.unique(y_train)), dtype=int)
    if class_labels.size < 2:
        return np.full(len(predict), MLFactorClass(int(y_train[0])).score, dtype=float)
    encoded_y = np.searchsorted(class_labels, y_train)

    params = _classifier_model_params(model_params, num_class=int(class_labels.size))
    model = xgb.XGBClassifier(**params)
    model.fit(labelled[predictor_cols], encoded_y)
    probabilities = _align_class_probabilities(
        model.predict_proba(predict[predictor_cols]),
        class_labels,
    )
    return class_scores_from_probabilities(probabilities)


def class_scores_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Convert five-class probabilities to signed portfolio ranking scores."""

    normalized = _normalize_probabilities(probabilities)
    return normalized @ ML_FACTOR_CLASS_SCORES


def classification_metrics_from_probabilities(
    y_true: Sequence[int] | np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Return accuracy, macro averages, and multiclass log loss."""

    normalized = _normalize_probabilities(probabilities)
    y_pred = normalized.argmax(axis=1)
    metrics = classification_metrics(y_true, y_pred)
    y = np.asarray(y_true, dtype=int)
    valid = (y >= 0) & (y < ML_FACTOR_CLASS_COUNT)
    if valid.any():
        clipped = np.clip(normalized[valid], 1e-15, 1.0)
        metrics["log_loss"] = float(-np.log(clipped[np.arange(valid.sum()), y[valid]]).mean())
    return metrics


def classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
) -> dict[str, float]:
    """Return leakage-free multiclass metrics for already-labelled rows."""

    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    valid = (
        (true >= 0) & (true < ML_FACTOR_CLASS_COUNT) & (pred >= 0) & (pred < ML_FACTOR_CLASS_COUNT)
    )
    if not valid.any():
        return {}
    true = true[valid]
    pred = pred[valid]
    per_class: list[tuple[float, float, float]] = []
    for label in range(ML_FACTOR_CLASS_COUNT):
        tp = float(((pred == label) & (true == label)).sum())
        fp = float(((pred == label) & (true != label)).sum())
        fn = float(((pred != label) & (true == label)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append((precision, recall, f1))
    return {
        "accuracy": float((true == pred).mean()),
        "macro_precision": float(np.mean([item[0] for item in per_class])),
        "macro_recall": float(np.mean([item[1] for item in per_class])),
        "macro_f1": float(np.mean([item[2] for item in per_class])),
    }


def _classifier_model_params(
    model_params: dict[str, Any] | None,
    *,
    num_class: int,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "objective": "multi:softprob",
        "num_class": num_class,
        "eval_metric": "mlogloss",
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "n_jobs": -1,
    }
    params.update(model_params or {})
    params["objective"] = "multi:softprob"
    params["num_class"] = num_class
    return params


def _align_class_probabilities(probabilities: np.ndarray, classes: Sequence[int]) -> np.ndarray:
    raw = np.asarray(probabilities, dtype=float)
    if raw.ndim != 2:
        raise ValueError("Classifier probabilities must be a 2D array")
    if raw.shape[1] == ML_FACTOR_CLASS_COUNT:
        return _normalize_probabilities(raw)
    aligned = np.zeros((raw.shape[0], ML_FACTOR_CLASS_COUNT), dtype=float)
    for index, label in enumerate(classes):
        class_index = int(label)
        if 0 <= class_index < ML_FACTOR_CLASS_COUNT and index < raw.shape[1]:
            aligned[:, class_index] = raw[:, index]
    return _normalize_probabilities(aligned)


def _normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    raw = np.asarray(probabilities, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != ML_FACTOR_CLASS_COUNT:
        raise ValueError(f"Expected probabilities with shape (n, {ML_FACTOR_CLASS_COUNT})")
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = raw.sum(axis=1, keepdims=True)
    fallback = np.full_like(raw, 1.0 / ML_FACTOR_CLASS_COUNT)
    return np.divide(raw, row_sums, out=fallback, where=row_sums != 0.0)


def _drop_model_nulls(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.dropna(subset=list(columns)).copy()
    for column in columns:
        result = result[pd.to_numeric(result[column], errors="coerce").notna()]
    return result.copy()


def _finite_numeric(values: Sequence[float] | pd.Series | np.ndarray) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return numeric[np.isfinite(numeric)]


def _to_pandas(frame: object, *, label: str = "frame") -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame
    if hasattr(frame, "to_pandas"):
        result = frame.to_pandas()
        if isinstance(result, pd.DataFrame):
            return result
    raise TypeError(f"{label} must be a pandas or polars DataFrame; got {type(frame)!r}")


def _validate_target_col(target_col: str) -> None:
    if not any(
        target_col == prefix or target_col.startswith(prefix) for prefix in _TARGET_PREFIXES
    ):
        raise ValueError("MLFactorStrategy target_col must be a future percentage-change column")


def _validate_predictor_columns(predictor_cols: Sequence[str], target_col: str) -> None:
    forbidden = [
        column
        for column in predictor_cols
        if column == target_col
        or column in _FORBIDDEN_PREDICTOR_COLUMNS
        or any(column.startswith(prefix) for prefix in _FORBIDDEN_PREDICTOR_PREFIXES)
    ]
    if forbidden:
        raise ValueError(f"ML factor predictors cannot include future target columns: {forbidden}")


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be a positive integer")
    if value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _named_section(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or "name" not in value:
        raise ValueError(f"{label} must be a mapping with a name")
    return dict(value)


__all__ = [
    "CLASS_LABEL_COLUMN",
    "CLASS_PERCENTILES",
    "ML_FACTOR_CLASS_COUNT",
    "ML_FACTOR_CLASS_SCORES",
    "MLFactorClass",
    "MLFactorClassThresholds",
    "MLFactorStrategy",
    "class_scores_from_probabilities",
    "classification_metrics",
    "classification_metrics_from_probabilities",
    "train_and_predict_xgb_classifier",
]
