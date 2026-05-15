"""Local filesystem bundle adapter."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from qts.data.bundles.base import BaseBundleAdapter


class LocalBundleAdapter(BaseBundleAdapter):
    """Stores bundle data on disk without external zipline dependency."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def ingest(self, name: str, data: pl.DataFrame, start: date, end: date) -> Path:
        bundle_dir = self.root / name
        bundle_dir.mkdir(parents=True, exist_ok=True)
        data.write_parquet(bundle_dir / "ohlcv.parquet")
        manifest = pl.DataFrame(
            {
                "name": [name],
                "start_date": [start],
                "end_date": [end],
                "rows": [data.height],
            }
        )
        manifest.write_parquet(bundle_dir / "manifest.parquet")
        return bundle_dir

    def load(self, name: str) -> pl.DataFrame:
        return pl.read_parquet(self.root / name / "ohlcv.parquet")

    def exists(self, name: str) -> bool:
        return (self.root / name / "ohlcv.parquet").exists()
