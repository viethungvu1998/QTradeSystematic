"""Storage abstractions."""

from __future__ import annotations

import polars as pl


class BaseStorage:
    """Contract for storage backends."""

    def write(self, key: str, df: pl.DataFrame) -> None:
        raise NotImplementedError

    def read(self, key: str) -> pl.DataFrame:
        raise NotImplementedError

    def append(self, key: str, df: pl.DataFrame) -> None:
        raise NotImplementedError

    def query(self, sql: str) -> pl.DataFrame:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def list_keys(self) -> list[str]:
        raise NotImplementedError
