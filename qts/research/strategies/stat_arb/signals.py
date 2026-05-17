from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

SideType = Literal["long_only", "long_short"]


def compute_spread(series_a: pd.Series, series_b: pd.Series, hedge_ratio: float) -> pd.Series:
    return series_a - hedge_ratio * series_b


def compute_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling = series.rolling(window)
    mean = rolling.mean()
    std = rolling.std().replace(0, np.nan)
    return (series - mean) / std


def generate_zscore_signals(
    zscore: pd.Series,
    *,
    entry_z: float,
    exit_z: float,
    side: SideType = "long_short",
    stop_z: float | None = None,
    max_holding_bars: int | None = None,
) -> dict[str, pd.Series | None]:
    long_entries = _crossed_below(zscore, -entry_z)
    long_exits = _crossed_above(zscore, exit_z)
    long_entries, long_exits = _clean_entry_exit(long_entries, long_exits)

    short_entries = None
    short_exits = None
    if side == "long_short":
        short_entries = _crossed_above(zscore, entry_z)
        short_exits = _crossed_below(zscore, -exit_z)
        short_entries, short_exits = _clean_entry_exit(short_entries, short_exits)
    elif side != "long_only":
        raise ValueError(f"Unsupported side '{side}'.")

    if stop_z is not None:
        stop_hits = zscore.abs() >= float(stop_z)
        long_exits = long_exits | stop_hits
        if short_exits is not None:
            short_exits = short_exits | stop_hits

    if max_holding_bars and max_holding_bars > 0:
        long_exits = _apply_max_holding(long_entries, long_exits, max_holding_bars)
        if short_entries is not None and short_exits is not None:
            short_exits = _apply_max_holding(short_entries, short_exits, max_holding_bars)

    long_entries, long_exits = _clean_entry_exit(long_entries, long_exits)
    if short_entries is not None and short_exits is not None:
        short_entries, short_exits = _clean_entry_exit(short_entries, short_exits)

    return {
        "long_entries": long_entries,
        "long_exits": long_exits,
        "short_entries": short_entries,
        "short_exits": short_exits,
    }


def _apply_max_holding(
    entries: pd.Series,
    exits: pd.Series,
    max_bars: int,
) -> pd.Series:
    entries_arr = entries.to_numpy(dtype=bool)
    exits_arr = exits.to_numpy(dtype=bool)
    forced_exits = exits_arr.copy()
    in_position = False
    entry_idx = -1

    for idx in range(len(entries_arr)):
        if entries_arr[idx]:
            in_position = True
            entry_idx = idx
        if in_position:
            if exits_arr[idx]:
                in_position = False
                entry_idx = -1
                continue
            if idx - entry_idx >= max_bars:
                forced_exits[idx] = True
                in_position = False
                entry_idx = -1

    return pd.Series(forced_exits, index=entries.index)


def _crossed_above(series: pd.Series, threshold: float) -> pd.Series:
    previous = series.shift(1)
    current = series
    return ((previous <= threshold) & (current > threshold)).fillna(False)


def _crossed_below(series: pd.Series, threshold: float) -> pd.Series:
    previous = series.shift(1)
    current = series
    return ((previous >= threshold) & (current < threshold)).fillna(False)


def _clean_entry_exit(entries: pd.Series, exits: pd.Series) -> tuple[pd.Series, pd.Series]:
    entries_arr = entries.fillna(False).to_numpy(dtype=bool)
    exits_arr = exits.fillna(False).to_numpy(dtype=bool)
    cleaned_entries = np.zeros(len(entries_arr), dtype=bool)
    cleaned_exits = np.zeros(len(exits_arr), dtype=bool)
    in_position = False

    for idx, (entry, exit_) in enumerate(zip(entries_arr, exits_arr, strict=True)):
        if not in_position and entry:
            cleaned_entries[idx] = True
            in_position = True
        elif in_position and exit_:
            cleaned_exits[idx] = True
            in_position = False

    return pd.Series(cleaned_entries, index=entries.index), pd.Series(cleaned_exits, index=exits.index)
