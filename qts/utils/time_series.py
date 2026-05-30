"""Shared time-series helpers."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def time_series_cv_splits(
    frame: pd.DataFrame,
    *,
    n_splits: int,
    gap: int,
    test_size: int | None,
    max_train_size: int | None,
    date_col: str = "date",
    sort_cols: Sequence[str] = ("date", "symbol"),
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    if date_col not in frame.columns:
        raise ValueError("TimeSeriesSplit requires a date column")

    from sklearn.model_selection import TimeSeriesSplit

    order_cols = [column for column in sort_cols if column in frame.columns]
    sorted_frame = frame.sort_values(order_cols).copy() if order_cols else frame.copy()
    sorted_frame["_cv_date"] = pd.to_datetime(sorted_frame[date_col])
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


def unique_sorted_timestamps(values: pd.Series) -> list[pd.Timestamp]:
    return list(pd.DatetimeIndex(pd.to_datetime(values).drop_duplicates()).sort_values())


__all__ = ["time_series_cv_splits", "unique_sorted_timestamps"]
