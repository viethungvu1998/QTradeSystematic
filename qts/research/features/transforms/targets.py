"""Target-label transforms for research pipelines."""

from __future__ import annotations

import polars as pl

from qts.core.registry import Registry
from qts.research.features.forward_returns import ForwardReturns
from qts.utils.labels import MLFactorClassThresholds

_FORWARD_RETURN_PREFIX = "forward_return_"


def _ensure_target_column(df: pl.DataFrame, *, target_column: str) -> pl.DataFrame:
    if target_column in df.columns:
        return df
    if not target_column.startswith(_FORWARD_RETURN_PREFIX):
        raise ValueError(
            f"Transform target column '{target_column}' is missing and cannot be derived."
        )
    period_text = target_column.removeprefix(_FORWARD_RETURN_PREFIX)
    try:
        period = int(period_text)
    except ValueError as exc:
        raise ValueError(
            f"Transform target column '{target_column}' must end with an integer period."
        ) from exc
    return ForwardReturns(periods=[period]).fit_transform(df)


@Registry.register_transform("ml_factor_target_class")
def ml_factor_target_class_transform(
    df: pl.DataFrame,
    *,
    target_column: str,
    output_column: str | None = None,
) -> pl.DataFrame:
    """Append five quantile classes for an ML factor forward-return target."""

    frame = _ensure_target_column(df, target_column=target_column)
    label_column = output_column or f"{target_column}_class"
    thresholds = MLFactorClassThresholds.fit(
        frame[target_column].to_list(),
        target_col=target_column,
    )
    q15, q40, q60, q85 = thresholds.values
    labels = (
        pl.when(pl.col(target_column).is_null() | pl.col(target_column).is_nan())
        .then(pl.lit(None, dtype=pl.Int16))
        .when(pl.col(target_column) <= q15)
        .then(pl.lit(0, dtype=pl.Int16))
        .when(pl.col(target_column) <= q40)
        .then(pl.lit(1, dtype=pl.Int16))
        .when(pl.col(target_column) <= q60)
        .then(pl.lit(2, dtype=pl.Int16))
        .when(pl.col(target_column) <= q85)
        .then(pl.lit(3, dtype=pl.Int16))
        .otherwise(pl.lit(4, dtype=pl.Int16))
        .alias(label_column)
    )
    return frame.with_columns(labels)

