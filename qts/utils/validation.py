"""Shared validation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def names_matching_any(
    names: Sequence[str],
    *,
    exact: Iterable[str] = (),
    prefixes: Iterable[str] = (),
) -> list[str]:
    exact_set = set(exact)
    prefix_tuple = tuple(prefixes)
    return [
        name
        for name in names
        if name in exact_set or any(name.startswith(prefix) for prefix in prefix_tuple)
    ]


def name_matches_any_prefix(name: str, prefixes: Iterable[str]) -> bool:
    return any(name == prefix or name.startswith(prefix) for prefix in prefixes)


def validate_target_col(
    target_col: str,
    target_prefixes: Iterable[str],
    *,
    message: str = "target_col must match an allowed prefix",
) -> None:
    if not name_matches_any_prefix(target_col, target_prefixes):
        raise ValueError(message)


def validate_predictor_columns(
    predictor_cols: Sequence[str],
    target_col: str,
    *,
    forbidden_columns: Iterable[str] = (),
    forbidden_prefixes: Iterable[str] = (),
    message: str = "predictor columns cannot include target columns",
) -> None:
    forbidden = names_matching_any(
        predictor_cols,
        exact=(*tuple(forbidden_columns), target_col),
        prefixes=forbidden_prefixes,
    )
    if forbidden:
        raise ValueError(f"{message}: {forbidden}")


def positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be a positive integer")
    if value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def min_int(value: object, label: str, *, minimum: int) -> int:
    result = non_negative_int(value, label)
    if result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return result


def non_negative_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative integer") from exc
    if result < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return result


def optional_positive_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    return positive_int(value, label)


__all__ = [
    "min_int",
    "name_matches_any_prefix",
    "names_matching_any",
    "non_negative_int",
    "optional_positive_int",
    "positive_int",
    "validate_predictor_columns",
    "validate_target_col",
]
