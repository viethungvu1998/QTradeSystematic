"""Bundle abstractions."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl


class BaseBundleAdapter:
    """Contract for local bundle adapters."""

    def ingest(self, name: str, data: pl.DataFrame, start: date, end: date) -> Path:
        raise NotImplementedError

    def load(self, name: str) -> pl.DataFrame:
        raise NotImplementedError

    def exists(self, name: str) -> bool:
        raise NotImplementedError
