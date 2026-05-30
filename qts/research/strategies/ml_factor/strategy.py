"""Classification-based ML factor strategy."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial

import numpy as np
import pandas as pd
import polars as pl

from qts.core.registry import Registry
from qts.research.strategies._config import resolve_named_section
from qts.research.strategies.factor.base import BaseFactorStrategy
from qts.utils.dataframe import drop_non_numeric_nulls
from qts.utils.time_series import time_series_cv_splits, unique_sorted_timestamps
from qts.utils.validation import (
    min_int,
    non_negative_int,
    optional_positive_int,
    positive_int,
    validate_predictor_columns,
    validate_target_col,
)

from .constants import (
    FORBIDDEN_PREDICTOR_COLUMNS,
    FORBIDDEN_PREDICTOR_PREFIXES,
    LEGACY_REGRESSION_TRAINERS,
    TARGET_PREFIXES,
)
from .labels import MLFactorClassThresholds
from .models.base import BaseMLFactorModel


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
        model: BaseMLFactorModel | None = None,
        task: str = "classification",
        cv_splits: int = 5,
        cv_gap: int = 0,
        cv_test_size: int | None = None,
        cv_max_train_size: int | None = None,
    ) -> None:
        validate_target_col(
            target_col,
            TARGET_PREFIXES,
            message="MLFactorStrategy target_col must be a future percentage-change column",
        )
        validate_predictor_columns(
            predictor_cols,
            target_col,
            forbidden_columns=FORBIDDEN_PREDICTOR_COLUMNS,
            forbidden_prefixes=FORBIDDEN_PREDICTOR_PREFIXES,
            message="ML factor predictors cannot include future target columns",
        )
        self.predictor_cols = list(predictor_cols)
        self.target_col = target_col
        self.model = model
        self.train_func = train_func
        self.portfolio_func = portfolio_func
        self.task = task
        self.rebalance_period = positive_int(rebalance_period, "rebalance_period")
        self.min_train_rows = int(min_train_rows)
        self.cv_splits = min_int(cv_splits, "cv_splits", minimum=2)
        self.cv_gap = non_negative_int(cv_gap, "cv_gap")
        self.cv_test_size = optional_positive_int(cv_test_size, "cv_test_size")
        self.cv_max_train_size = optional_positive_int(cv_max_train_size, "cv_max_train_size")
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
        rebalance_period = positive_int(
            payload.pop("rebalance_period", payload.pop("rebalance_frequency", 10)),
            "rebalance_period",
        )
        min_train_rows = int(payload.pop("min_train_rows", 2))
        cv_splits = min_int(payload.pop("cv_splits", 5), "cv_splits", minimum=2)
        cv_gap = non_negative_int(payload.pop("cv_gap", 0), "cv_gap")
        cv_test_size = optional_positive_int(payload.pop("cv_test_size", None), "cv_test_size")
        cv_max_train_size = optional_positive_int(
            payload.pop("cv_max_train_size", None),
            "cv_max_train_size",
        )
        model_raw = payload.pop("model", None)
        trainer_raw = payload.pop("trainer", None)
        resolved_model: BaseMLFactorModel | None = None
        train_func = None

        if model_raw is not None:
            model_cfg = resolve_named_section(model_raw, "model")
            model_name = str(model_cfg["name"])
            model_params = dict(model_cfg["params"])
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
            trainer = resolve_named_section(trainer_raw, "trainer")
            trainer_name = str(trainer["name"])
            if trainer_name in LEGACY_REGRESSION_TRAINERS:
                raise ValueError(
                    f"MLFactorStrategy requires a classification trainer, got {trainer_name}"
                )
            trainer_params = dict(trainer["params"])
            trainer_params.setdefault("predictor_cols", predictor_cols)
            trainer_params.setdefault("target_col", target_col)
            train_func = partial(Registry.get_factor_trainer(trainer_name), **trainer_params)
        else:
            model_cls = Registry.get_model("xgb_classifier")
            resolved_model = model_cls(task=task)

        if portfolio_func is None and "portfolio" in payload:
            portfolio = resolve_named_section(payload.pop("portfolio"), "portfolio")
            portfolio_func = partial(
                Registry.get_portfolio_constructor(str(portfolio["name"])),
                **dict(portfolio["params"]),
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
        trainable_pd = drop_non_numeric_nulls(
            trainable_pd,
            [*self.predictor_cols, self.target_col],
        )
        predict_pd = drop_non_numeric_nulls(predict_pd, self.predictor_cols)
        if len(trainable_pd) < self.min_train_rows or predict_pd.empty:
            return self.empty_signal_frame()

        splits = time_series_cv_splits(
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
        self.last_cv_train_dates = unique_sorted_timestamps(last_train_pd["date"])
        self.last_cv_test_dates = unique_sorted_timestamps(last_test_pd["date"])
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


__all__ = ["MLFactorStrategy"]
