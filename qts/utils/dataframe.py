"""Shared DataFrame helpers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pandas as pd
import polars as pl


def drop_non_numeric_nulls(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.dropna(subset=list(columns)).copy()
    for column in columns:
        result = result[pd.to_numeric(result[column], errors="coerce").notna()]
    return result.copy()


def serialise_list_columns(frame: pl.DataFrame) -> pl.DataFrame:
    csv_frame = frame
    for name, dtype in frame.schema.items():
        if isinstance(dtype, pl.List):
            values = [to_json_text(value) for value in frame[name].to_list()]
            csv_frame = csv_frame.with_columns(pl.Series(name, values, dtype=pl.String))
    return csv_frame


def to_pandas_frame(frame: object, *, label: str = "frame") -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame
    if hasattr(frame, "to_pandas"):
        result = frame.to_pandas()
        if isinstance(result, pd.DataFrame):
            return result
    raise TypeError(f"{label} must be a pandas or polars DataFrame; got {type(frame)!r}")


def to_json_text(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


__all__ = [
    "drop_non_numeric_nulls",
    "serialise_list_columns",
    "to_json_text",
    "to_pandas_frame",
]
