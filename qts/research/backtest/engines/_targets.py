"""Helpers for turning standard signal frames into engine order schedules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl


@dataclass(frozen=True, slots=True)
class TargetSchedule:
    """Dense held targets plus sparse change events derived from a signal frame."""

    targets: pd.DataFrame
    events: pd.DataFrame


def _targets_from_signals(signals: pl.DataFrame) -> pd.DataFrame:
    if signals.is_empty():
        return pd.DataFrame()

    sig_pdf = (
        signals.select(["date", "symbol", "signal", "weight"])
        .to_pandas()
        .copy()
    )
    sig_pdf["date"] = pd.to_datetime(sig_pdf["date"])
    sig_pdf["target"] = np.where(sig_pdf["signal"] != 0, sig_pdf["signal"] * sig_pdf["weight"], 0.0)
    return (
        sig_pdf.pivot_table(index="date", columns="symbol", values="target", aggfunc="last")
        .rename_axis(index=None, columns=None)
        .sort_index()
        .sort_index(axis=1)
    )


def build_target_schedule(
    signals: pl.DataFrame,
    sessions: pd.DatetimeIndex,
    symbols: list[str],
    *,
    shift_by_one_bar: bool = False,
) -> TargetSchedule:
    """Build held targets and sparse target-change events for engine execution.

    `targets` is a dense matrix of held target weights after applying optional
    next-session execution timing. `events` contains only dates where the held
    target changes; NaN means "no action / keep holding".
    """

    idx = pd.DatetimeIndex(pd.to_datetime(sessions)).sort_values()
    cols = pd.Index(sorted(symbols))
    event_targets = _targets_from_signals(signals).reindex(index=idx, columns=cols)
    if shift_by_one_bar:
        event_targets = event_targets.shift(1)

    held_targets = event_targets.ffill().fillna(0.0)
    previous_targets = held_targets.shift(1).fillna(0.0)
    changed_mask = ~np.isclose(
        held_targets.to_numpy(dtype=float),
        previous_targets.to_numpy(dtype=float),
        atol=1e-12,
        rtol=0.0,
    )
    change_events = held_targets.where(pd.DataFrame(changed_mask, index=idx, columns=cols))
    return TargetSchedule(targets=held_targets, events=change_events)


def schedule_to_lookup(events: pd.DataFrame) -> dict[object, dict[str, float]]:
    """Convert sparse event schedule to {date: {symbol: target}} lookup."""

    lookup: dict[object, dict[str, float]] = {}
    for timestamp, row in events.iterrows():
        targets = {
            str(symbol): float(value)
            for symbol, value in row.items()
            if pd.notna(value)
        }
        if targets:
            lookup[pd.Timestamp(timestamp).date()] = targets
    return lookup
