"""Date interval helpers for out-of-sample windows."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date, datetime
from typing import Any

DateInterval = tuple[date, date | None]


def to_date(value: Any) -> date:
    """Normalize common date-like values without depending on pandas."""

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if hasattr(value, "date") and callable(value.date):
        return value.date()
    return date.fromisoformat(str(value))


def normalize_date_intervals(intervals: Iterable[Any] | None) -> list[DateInterval]:
    """Return sorted, non-overlapping inclusive intervals."""

    normalized: list[DateInterval] = []
    for item in intervals or []:
        if isinstance(item, tuple) and len(item) == 2:
            start_raw, end_raw = item
        elif isinstance(item, list) and len(item) == 2:
            start_raw, end_raw = item
        else:
            start_raw = getattr(item, "start_date")
            end_raw = getattr(item, "end_date")
        start = to_date(start_raw)
        end = None if end_raw is None else to_date(end_raw)
        if end is not None and start > end:
            raise ValueError("test interval start_date must be on or before end_date")
        normalized.append((start, end))

    normalized.sort(key=lambda interval: interval[0])
    previous_end: date | None = None
    for start, end in normalized:
        if previous_end is not None and start <= previous_end:
            raise ValueError("test intervals must not overlap")
        if end is None:
            previous_end = date.max
        else:
            previous_end = end
    return normalized


def date_in_intervals(value: Any, intervals: Sequence[DateInterval]) -> bool:
    """Return True when value is inside any inclusive interval."""

    current = to_date(value)
    return any(start <= current and (end is None or current <= end) for start, end in intervals)


def filter_dates_in_intervals(values: Iterable[Any], intervals: Sequence[DateInterval]) -> list[Any]:
    """Keep values inside at least one configured interval."""

    if not intervals:
        return list(values)
    return [value for value in values if date_in_intervals(value, intervals)]


def select_training_dates(
    values: Iterable[Any],
    *,
    before: Any,
    train_window: int,
    test_intervals: Sequence[DateInterval],
) -> list[Any]:
    """Select the last train_window dates before before, excluding test intervals."""

    before_date = to_date(before)
    eligible = [
        value
        for value in values
        if to_date(value) < before_date and not date_in_intervals(value, test_intervals)
    ]
    if train_window < 1:
        return []
    return eligible[-train_window:]


def test_intervals_from_config(config: Any) -> list[DateInterval]:
    """Resolve multi-interval validation with legacy test_start_date fallback."""

    validation = getattr(config, "validation", None)
    validation_intervals = getattr(validation, "test_intervals", None) if validation else None
    if validation_intervals:
        return normalize_date_intervals(validation_intervals)

    start = getattr(validation, "test_start_date", None) if validation else None
    if start is None:
        start = getattr(config, "test_start_date", None)
    if start is None:
        return []
    return normalize_date_intervals([(start, None)])


def format_date_intervals(intervals: Sequence[DateInterval]) -> str:
    """Format intervals for concise reports."""

    if not intervals:
        return "full period"
    parts = []
    for start, end in intervals:
        end_text = "open" if end is None else end.isoformat()
        parts.append(f"{start.isoformat()}..{end_text}")
    return ", ".join(parts)


__all__ = [
    "DateInterval",
    "date_in_intervals",
    "filter_dates_in_intervals",
    "format_date_intervals",
    "normalize_date_intervals",
    "select_training_dates",
    "test_intervals_from_config",
    "to_date",
]
