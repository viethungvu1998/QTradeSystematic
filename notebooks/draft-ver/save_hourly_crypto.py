"""Export hourly crypto futures bars from DuckDB to per-symbol CSV files."""

from __future__ import annotations

import os
import re
from pathlib import Path

import duckdb


TABLE = "crypto_futures_intraday_prices"
INTERVAL = "1h"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "csv"
EXPORT_COLUMNS = [
    "bar_time",
    "date",
    "symbol",
    "interval",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def database_path() -> Path:
    qts_root = Path(os.environ.get("QTS_ROOT", Path.home() / ".qts"))
    return qts_root / "database" / "qts.duckdb"


def quote_identifier(value: str) -> str:
    """Quote a DuckDB identifier."""
    return '"' + value.replace('"', '""') + '"'


def csv_name(symbol: str) -> str:
    safe_symbol = re.sub(r"[^A-Za-z0-9._-]+", "_", symbol).strip("_")
    return f"{safe_symbol}_{INTERVAL}.csv"


def table_exists(connection: duckdb.DuckDBPyConnection, table: str) -> bool:
    row = connection.execute(
        """
        select count(*)
        from information_schema.tables
        where table_schema = 'main' and table_name = ?
        """,
        [table],
    ).fetchone()
    return bool(row and row[0])


def open_database(db_path: Path) -> duckdb.DuckDBPyConnection:
    try:
        return duckdb.connect(str(db_path), read_only=True)
    except duckdb.IOException as exc:
        if "Conflicting lock" in str(exc):
            raise SystemExit(
                f"DuckDB database is locked by another process: {db_path}\n"
                "Close any notebook/kernel with an open DuckDB connection, then rerun this script."
            ) from None
        raise


def symbols_for_interval(connection: duckdb.DuckDBPyConnection) -> list[str]:
    table = quote_identifier(TABLE)
    rows = connection.execute(
        f"""
        select distinct symbol
        from {table}
        where interval = ?
        order by symbol
        """,
        [INTERVAL],
    ).fetchall()
    return [row[0] for row in rows]


def export_symbol(connection: duckdb.DuckDBPyConnection, symbol: str) -> tuple[Path, int]:
    table = quote_identifier(TABLE)
    columns = ", ".join(quote_identifier(column) for column in EXPORT_COLUMNS)
    frame = connection.execute(
        f"""
        select {columns}
        from {table}
        where interval = ? and symbol = ?
        order by bar_time
        """,
        [INTERVAL, symbol],
    ).pl()

    path = OUTPUT_DIR / csv_name(symbol)
    frame.write_csv(path)
    return path, frame.height


def main() -> None:
    db_path = database_path()
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB database not found: {db_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open_database(db_path) as connection:
        if not table_exists(connection, TABLE):
            raise RuntimeError(f"DuckDB table not found: {TABLE}")

        symbols = symbols_for_interval(connection)
        if not symbols:
            print(f"No rows found in {TABLE} for interval={INTERVAL!r}")
            return

        total_rows = 0
        for symbol in symbols:
            path, row_count = export_symbol(connection, symbol)
            total_rows += row_count
            print(f"{symbol}: {row_count:,} rows -> {path}")

    print(f"Exported {len(symbols)} CSV files and {total_rows:,} rows to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
