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
        existing_columns = self._columns(key)
        if self._uses_bar_identity(existing_columns, df.columns):
            self._append_by_bar_identity(key, existing_columns)
            return
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

    def _columns(self, key: str) -> list[str]:
        rows = self.connection.execute(
            """
            select column_name
            from information_schema.columns
            where table_schema = 'main' and table_name = ?
            order by ordinal_position
            """,
            [key],
        ).fetchall()
        return [row[0] for row in rows]

    def _append_by_bar_identity(self, key: str, columns: list[str]) -> None:
        # Intraday bars can be revised; identity-based replacement avoids duplicate bar keys.
        table = _quote_identifier(key)
        self.connection.execute(
            f"""
            delete from {table}
            where (symbol, interval, bar_time) in (
                select symbol, interval, bar_time from incoming_frame
            )
            """
        )
        column_sql = ", ".join(_quote_identifier(column) for column in columns)
        self.connection.execute(
            f"""
            insert into {table} ({column_sql})
            select {column_sql}
            from incoming_frame
            """
        )

    @staticmethod
    def _uses_bar_identity(existing_columns: list[str], incoming_columns: list[str]) -> bool:
        identity_columns = {"symbol", "interval", "bar_time"}
        return identity_columns <= set(existing_columns) and identity_columns <= set(
            incoming_columns
        )


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'
