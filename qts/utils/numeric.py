"""Shared numeric helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def finite_numeric(values: Sequence[float] | pd.Series | np.ndarray) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return numeric[np.isfinite(numeric)]


__all__ = ["finite_numeric"]
