from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from qts.utils.paths import database_path

logger = logging.getLogger(__name__)


def _connect_duckdb(db_path: Path):
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("DuckDB is required for the stat-arb universe screener.") from exc
    return duckdb.connect(str(db_path), read_only=True)


def _resolve_exchange_column(con, profiles_table: str) -> str | None:
    try:
        cols = con.execute(f"PRAGMA table_info('{profiles_table}')").fetchdf()
    except Exception:  # pragma: no cover
        return None
    col_names = set(cols["name"].astype(str))
    if "exchangeShortName" in col_names:
        return "exchangeShortName"
    if "exchange" in col_names:
        return "exchange"
    return None


def stat_arb_universe_screener(
    *,
    db_path: Path | str | None = None,
    prices_table: str,
    profiles_table: str | None = "bulk_company_profiles_fmp",
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    min_history_days: int = 504,
    max_symbols: int = 250,
    min_avg_volume: float | None = None,
    min_price: float | None = None,
    sector_filter: Iterable[str] | None = None,
    exchange_filter: Iterable[str] | None = None,
    active_only: bool = True,
) -> list[str]:
    resolved_path = Path(db_path) if db_path else database_path()
    sector_filter = [s for s in (sector_filter or []) if s]
    exchange_filter = [e for e in (exchange_filter or []) if e]

    params: list[object] = [start_date.to_pydatetime(), end_date.to_pydatetime()]
    having = ["COUNT(*) >= ?"]
    params.append(int(min_history_days))

    if min_avg_volume is not None:
        having.append("AVG(volume) >= ?")
        params.append(float(min_avg_volume))
    if min_price is not None:
        having.append("AVG(close) >= ?")
        params.append(float(min_price))

    base_query = (
        "SELECT symbol, COUNT(*) AS n, AVG(volume) AS avg_volume, AVG(close) AS avg_close "
        f"FROM {prices_table} "
        "WHERE date >= ? AND date <= ? "
        "GROUP BY symbol "
        f"HAVING {' AND '.join(having)}"
    )

    with _connect_duckdb(resolved_path) as con:
        try:
            df = con.execute(base_query, params).fetchdf()
        except Exception as exc:  # pragma: no cover
            logger.warning("Universe screener failed on %s: %s", prices_table, exc)
            return []

        if df.empty:
            return []

        if profiles_table:
            try:
                exchange_col = _resolve_exchange_column(con, profiles_table)
                exchange_select = f"{exchange_col} AS exchange" if exchange_col else "NULL AS exchange"
                profiles = con.execute(
                    f"SELECT symbol, sector, {exchange_select}, isActivelyTrading "
                    f"FROM {profiles_table}"
                ).fetchdf()
                df = df.merge(profiles, on="symbol", how="left")
            except Exception as exc:  # pragma: no cover
                logger.warning("Universe screener profiles join failed: %s", exc)

        if sector_filter and "sector" in df.columns:
            df = df[df["sector"].isin(sector_filter)]
        if exchange_filter and "exchange" in df.columns:
            df = df[df["exchange"].isin(exchange_filter)]
        if active_only and "isActivelyTrading" in df.columns:
            df = df[df["isActivelyTrading"] == True]  # noqa: E712

        df = df.sort_values("avg_volume", ascending=False).head(int(max_symbols))

    return df["symbol"].dropna().astype(str).tolist()


__all__ = ["stat_arb_universe_screener"]
