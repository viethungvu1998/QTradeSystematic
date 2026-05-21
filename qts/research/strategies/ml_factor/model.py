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
from qts.research.strategies.ml_factor.base_model import BaseModel

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
        train_func: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray] | None = None,
        portfolio_func: Callable[..., dict[str, float]] | None = None,
        rebalance_period: int = 10,
        min_train_rows: int = 2,
        model: BaseModel | None = None,
        task: str = "classification",
        cv_splits: int = 5,
        cv_gap: int = 0,
        cv_test_size: int | None = None,
        cv_max_train_size: int | None = None,
    ) -> None:
        _validate_target_col(target_col)
        _validate_predictor_columns(predictor_cols, target_col)
        self.predictor_cols = list(predictor_cols)
        self.target_col = target_col
        self.model = model
        self.train_func = train_func
        self.portfolio_func = portfolio_func
        self.task = task
        self.rebalance_period = _positive_int(rebalance_period, "rebalance_period")
        self.min_train_rows = int(min_train_rows)
        self.cv_splits = _min_int(cv_splits, "cv_splits", minimum=2)
        self.cv_gap = _non_negative_int(cv_gap, "cv_gap")
        self.cv_test_size = _optional_positive_int(cv_test_size, "cv_test_size")
        self.cv_max_train_size = _optional_positive_int(cv_max_train_size, "cv_max_train_size")
        self.last_thresholds: MLFactorClassThresholds | None = None
        self.last_cv_train_dates: list[pd.Timestamp] = []
        self.last_cv_test_dates: list[pd.Timestamp] = []

    @classmethod
    def from_config_params(
        cls,
        params: Mapping[str, object],
        *,
        portfolio_func: Callable | None = None,
    ) -> MLFactorStrategy:
        payload = dict(params)
        predictor_cols = [str(item) for item in payload.pop("predictor_cols")]
        target_col = str(payload.pop("target_col"))
        task = str(payload.pop("task", "classification"))
        rebalance_period = _positive_int(
            payload.pop("rebalance_period", payload.pop("rebalance_frequency", 10)),
            "rebalance_period",
        )
        min_train_rows = int(payload.pop("min_train_rows", 2))
        cv_splits = _min_int(payload.pop("cv_splits", 5), "cv_splits", minimum=2)
        cv_gap = _non_negative_int(payload.pop("cv_gap", 0), "cv_gap")
        cv_test_size = _optional_positive_int(payload.pop("cv_test_size", None), "cv_test_size")
        cv_max_train_size = _optional_positive_int(
            payload.pop("cv_max_train_size", None),
            "cv_max_train_size",
        )
        model_raw = payload.pop("model", None)
        trainer_raw = payload.pop("trainer", None)
        resolved_model: BaseModel | None = None
        train_func = None

        if model_raw is not None:
            model_cfg = _named_section(model_raw, "model")
            model_name = str(model_cfg["name"])
            model_params = dict(model_cfg.get("params", {}))
            if "classifier" in model_name and task != "classification":
                import warnings

                warnings.warn(
                    f"model '{model_name}' implies classification but task='{task}' was specified.",
                    stacklevel=2,
                )
            if "regressor" in model_name and task != "regression":
                import warnings

                warnings.warn(
                    f"model '{model_name}' implies regression but task='{task}' was specified.",
                    stacklevel=2,
                )
            model_cls = Registry.get_model(model_name)
            resolved_model = model_cls(task=task, **model_params)
        elif trainer_raw is not None:
            trainer = _named_section(trainer_raw, "trainer")
            trainer_name = str(trainer["name"])
            if trainer_name in _LEGACY_REGRESSION_TRAINERS:
                raise ValueError(
                    f"MLFactorStrategy requires a classification trainer, got {trainer_name}"
                )
            trainer_params = dict(trainer.get("params", {}))
            trainer_params.setdefault("predictor_cols", predictor_cols)
            trainer_params.setdefault("target_col", target_col)
            train_func = partial(Registry.get_factor_trainer(trainer_name), **trainer_params)
        else:
            model_cls = Registry.get_model("xgb_classifier")
            resolved_model = model_cls(task=task)

        if portfolio_func is None and "portfolio" in payload:
            portfolio = _named_section(payload.pop("portfolio"), "portfolio")
            portfolio_func = partial(
                Registry.get_portfolio_constructor(str(portfolio["name"])),
                **dict(portfolio.get("params", {})),
            )
        else:
            payload.pop("portfolio", None)
        payload.pop("rebalance_period", None)

        return cls(
            predictor_cols=predictor_cols,
            target_col=target_col,
            train_func=train_func,
            portfolio_func=portfolio_func,
            rebalance_period=rebalance_period,
            min_train_rows=min_train_rows,
            model=resolved_model,
            task=task,
            cv_splits=cv_splits,
            cv_gap=cv_gap,
            cv_test_size=cv_test_size,
            cv_max_train_size=cv_max_train_size,
        )

    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        self.last_cv_train_dates = []
        self.last_cv_test_dates = []
        if df.is_empty():
            return self.empty_signal_frame()
        if self.target_col not in df.columns:
            return self.empty_signal_frame()
        if any(column not in df.columns for column in self.predictor_cols):
            return self.empty_signal_frame()

        df_pd = df.sort(["date", "symbol"]).to_pandas()
        last_date = df_pd["date"].max()
        trainable_pd = df_pd[(df_pd["date"] < last_date) & df_pd[self.target_col].notna()].copy()
        predict_pd = df_pd[df_pd["date"] == last_date].copy()
        trainable_pd = _drop_model_nulls(trainable_pd, [*self.predictor_cols, self.target_col])
        predict_pd = _drop_model_nulls(predict_pd, self.predictor_cols)
        if len(trainable_pd) < self.min_train_rows or predict_pd.empty:
            return self.empty_signal_frame()

        splits = _time_series_cv_splits(
            trainable_pd,
            n_splits=self.cv_splits,
            gap=self.cv_gap,
            test_size=self.cv_test_size,
            max_train_size=self.cv_max_train_size,
        )
        fold_scores: list[np.ndarray] = []
        last_train_pd: pd.DataFrame | None = None
        last_test_pd: pd.DataFrame | None = None
        for train_pd, test_pd in splits:
            if len(train_pd) < self.min_train_rows or test_pd.empty:
                continue
            if self.model is not None:
                self.model.fit(train_pd[self.predictor_cols], train_pd[self.target_col])
                fold_scores.append(
                    np.asarray(self.model.predict(predict_pd[self.predictor_cols]), dtype=float)
                )
            elif self.train_func is not None:
                self.last_thresholds = MLFactorClassThresholds.fit(
                    train_pd[self.target_col],
                    target_col=self.target_col,
                )
                fold_scores.append(np.asarray(self.train_func(train_pd, predict_pd), dtype=float))
            else:
                raise RuntimeError("MLFactorStrategy requires either model= or train_func=.")
            last_train_pd = train_pd
            last_test_pd = test_pd
        if not fold_scores or last_train_pd is None or last_test_pd is None:
            return self.empty_signal_frame()
        self.last_cv_train_dates = _unique_sorted_timestamps(last_train_pd["date"])
        self.last_cv_test_dates = _unique_sorted_timestamps(last_test_pd["date"])
        scores = np.mean(np.vstack(fold_scores), axis=0)
        if scores.shape[0] != len(predict_pd):
            raise ValueError("ML factor trainer output must align to predict rows")

        predictions = pd.Series(scores, index=predict_pd["symbol"].values)
        if self.portfolio_func is None:
            raise RuntimeError("MLFactorStrategy requires portfolio_func=.")
        weights_dict = self.portfolio_func(predictions, history_df=last_train_pd)
        if not weights_dict:
            return self.empty_signal_frame()
        return self.signal_frame_from_weights(last_date, weights_dict)


def train_and_predict_xgb_classifier(
    train_data: object,
    predict_data: object,
    predictor_cols: list[str],
    target_col: str,
    model_params: dict[str, Any] | None = None,
    cv_splits: int = 5,
    cv_gap: int = 0,
    cv_test_size: int | None = None,
    cv_max_train_size: int | None = None,
) -> np.ndarray:
    """Fit an XGBoost multiclass classifier and return signed class scores."""
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
    splits = _time_series_cv_splits(
        train,
        n_splits=_min_int(cv_splits, "cv_splits", minimum=2),
        gap=_non_negative_int(cv_gap, "cv_gap"),
        test_size=_optional_positive_int(cv_test_size, "cv_test_size"),
        max_train_size=_optional_positive_int(cv_max_train_size, "cv_max_train_size"),
    )
    if not splits:
        return np.array([], dtype=float)

    from qts.research.strategies.ml_factor.models import XGBClassifierModel

    fold_scores = []
    for train_fold, test_fold in splits:
        if train_fold.empty or test_fold.empty:
            continue
        model = XGBClassifierModel(task="classification", **(model_params or {}))
        model.fit(train_fold[predictor_cols], train_fold[target_col])
        fold_scores.append(model.predict(predict[predictor_cols]))
    if not fold_scores:
        return np.array([], dtype=float)
    return np.mean(np.vstack(fold_scores), axis=0)


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


def _time_series_cv_splits(
    frame: pd.DataFrame,
    *,
    n_splits: int,
    gap: int,
    test_size: int | None,
    max_train_size: int | None,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    if "date" not in frame.columns:
        raise ValueError("TimeSeriesSplit requires a date column")

    from sklearn.model_selection import TimeSeriesSplit

    sorted_frame = frame.sort_values(["date", "symbol"]).copy()
    sorted_frame["_cv_date"] = pd.to_datetime(sorted_frame["date"])
    dates = sorted_frame["_cv_date"].drop_duplicates().sort_values().to_numpy()
    if len(dates) <= n_splits:
        return []
    splitter = TimeSeriesSplit(
        n_splits=n_splits,
        gap=gap,
        test_size=test_size,
        max_train_size=max_train_size,
    )
    try:
        index_splits = list(splitter.split(dates))
    except ValueError:
        return []

    frames = []
    for train_idx, test_idx in index_splits:
        train_dates = set(dates[train_idx])
        test_dates = set(dates[test_idx])
        train = sorted_frame[sorted_frame["_cv_date"].isin(train_dates)].drop(columns="_cv_date")
        test = sorted_frame[sorted_frame["_cv_date"].isin(test_dates)].drop(columns="_cv_date")
        frames.append((train.copy(), test.copy()))
    return frames


def _unique_sorted_timestamps(values: pd.Series) -> list[pd.Timestamp]:
    return list(pd.DatetimeIndex(pd.to_datetime(values).drop_duplicates()).sort_values())


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


def _min_int(value: object, label: str, *, minimum: int) -> int:
    result = _non_negative_int(value, label)
    if result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return result


def _non_negative_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative integer") from exc
    if result < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return result


def _optional_positive_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, label)


def _named_section(value: object, label: str) -> dict[str, object]:
    if isinstance(value, str):
        return {"name": value, "params": {}}
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
