"""DuckDB storage."""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

from qts.core.registry import Registry
from qts.data.storage.base import BaseStorage


@Registry.register_storage("duckdb")
class DuckDBStorage(BaseStorage):
    """DuckDB-backed storage."""

    def __init__(self, database: str = ":memory:") -> None:
        self.database = database
        if database != ":memory:":
            Path(database).parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(database=database)

    def write(self, key: str, df: pl.DataFrame) -> None:
        self.connection.register("incoming_frame", df.to_arrow())
        self.connection.execute(f"create or replace table {key} as select * from incoming_frame")

    def read(self, key: str) -> pl.DataFrame:
        return self.connection.sql(f"select * from {key}").pl()

    def append(self, key: str, df: pl.DataFrame) -> None:
        if not self.exists(key):
            self.write(key, df)
            return
        self.connection.register("incoming_frame", df.to_arrow())
        self.connection.execute(
            f"""
            insert into {key}
            select * from incoming_frame
            except
            select * from {key}
            """
        )

    def query(self, sql: str) -> pl.DataFrame:
        return self.connection.sql(sql).pl()

    def exists(self, key: str) -> bool:
        result = self.connection.execute(
            "select count(*) from information_schema.tables where table_name = ?",
            [key],
        ).fetchone()
        return bool(result and result[0])

    def list_keys(self) -> list[str]:
        rows = self.connection.execute(
            "select table_name from information_schema.tables where table_schema = 'main'"
        ).fetchall()
        return sorted(row[0] for row in rows)
