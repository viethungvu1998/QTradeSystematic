"""DuckDB-to-local-bundle ingest helpers."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from qts.data.bundles.base import BaseBundleAdapter
from qts.data.storage.duckdb import DuckDBStorage


def ingest_duckdb_to_bundle(
    storage: DuckDBStorage,
    adapter: BaseBundleAdapter,
    table: str,
    bundle_name: str,
    start: date,
    end: date,
) -> Path:
    """Persist stock data into the local bundle."""

    frame = storage.read(table).filter(pl.col("date").is_between(start, end))
    return adapter.ingest(bundle_name, frame, start, end)
