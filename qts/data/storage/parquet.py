"""Parquet storage."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from qts.core.registry import Registry
from qts.data.storage.base import BaseStorage


@Registry.register_storage("parquet")
class ParquetStorage(BaseStorage):
    """Filesystem-backed parquet cache."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        return self.root / f"{key}.parquet"

    def write(self, key: str, df: pl.DataFrame) -> None:
        df.write_parquet(self._path_for(key))

    def read(self, key: str) -> pl.DataFrame:
        return pl.read_parquet(self._path_for(key))

    def append(self, key: str, df: pl.DataFrame) -> None:
        if not self.exists(key):
            self.write(key, df)
            return
        combined = pl.concat([self.read(key), df], how="vertical").unique(maintain_order=True)
        self.write(key, combined)

    def query(self, sql: str) -> pl.DataFrame:
        raise NotImplementedError("ParquetStorage does not support SQL queries.")

    def exists(self, key: str) -> bool:
        return self._path_for(key).exists()

    def list_keys(self) -> list[str]:
        return sorted(path.stem for path in self.root.glob("*.parquet"))
