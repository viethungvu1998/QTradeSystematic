"""Data loading for VN100 quantamental workflows."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import yaml

from qts.data.manager import DataManager
from qts.data.sources.vnstock import VnstockDataSource
from qts.data.storage.duckdb import DuckDBStorage
from qts.data.storage.parquet import ParquetStorage
from qts.utils.paths import cache_dir


def normalize_vn_symbol(symbol: str, prefix: str = "VN:") -> str:
    raw = str(symbol).strip().upper()
    normalized_prefix = prefix.upper()
    return raw if raw.startswith(normalized_prefix) else f"{normalized_prefix}{raw}"


def load_vn100_symbols(path: Path, max_symbols: int | None) -> tuple[list[str], str]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    symbols = payload.get("symbols") or []
    if not isinstance(symbols, list) or not symbols:
        raise ValueError(f"{path} must define a non-empty symbols list")
    prefix = str(payload.get("symbol_prefix", "VN:"))
    normalized = [normalize_vn_symbol(item, prefix=prefix) for item in symbols]
    normalized = list(dict.fromkeys(normalized))
    benchmark = normalize_vn_symbol(payload.get("benchmark_symbol", "VN100"), prefix=prefix)
    if max_symbols is not None:
        normalized = normalized[:max_symbols]
    return normalized, benchmark


def make_vn_manager(runtime_root: Path) -> DataManager:
    storage = DuckDBStorage(database=str(runtime_root / "vn100.duckdb"))
    cache = ParquetStorage(root=runtime_root / "cache")
    return DataManager(
        stock_source=None,
        crypto_source=None,
        vn_stock_source=VnstockDataSource.from_env(),
        storage=storage,
        cache=cache,
        bundle_adapter=None,
    )


async def fetch_ohlcv_resilient(
    manager: DataManager,
    symbols: list[str],
    start_date: date,
    end_date: date,
    interval: str,
    batch_size: int,
) -> tuple[pl.DataFrame, list[tuple[str, str]]]:
    frames: list[pl.DataFrame] = []
    failures: list[tuple[str, str]] = []
    for start_idx in range(0, len(symbols), batch_size):
        batch = symbols[start_idx : start_idx + batch_size]
        try:
            frame = await manager.get_ohlcv(batch, start_date, end_date, interval=interval)
            if not frame.is_empty():
                frames.append(frame)
            continue
        except Exception as exc:
            print(f"Batch failed, retrying symbol-by-symbol: {batch} ({exc})")

        for symbol in batch:
            try:
                frame = await manager.get_ohlcv([symbol], start_date, end_date, interval=interval)
                if not frame.is_empty():
                    frames.append(frame)
            except Exception as exc:
                failures.append((symbol, str(exc)))

    if not frames:
        return pl.DataFrame(), failures
    combined = pl.concat(frames, how="vertical").unique(subset=["date", "symbol"], keep="last")
    return combined.sort(["symbol", "date"]), failures


async def fetch_prices_and_fundamentals(
    symbols: list[str],
    benchmark_symbol: str,
    runtime_root: Path,
    *,
    start_date: date,
    end_date: date,
    interval: str,
    batch_size: int,
    fetch_fundamentals: bool,
    fundamental_termtype: int,
    fundamental_pages: int,
    force_refresh_fundamentals: bool,
) -> tuple[pl.DataFrame, pl.DataFrame, list[tuple[str, str]]]:
    manager = make_vn_manager(runtime_root)
    ohlcv_all, failures = await fetch_ohlcv_resilient(
        manager, symbols, start_date, end_date, interval, batch_size
    )
    equity_symbols = [symbol for symbol in symbols if symbol != benchmark_symbol]
    if fetch_fundamentals:
        await manager.bulk_fetch_vn_fundamentals(
            equity_symbols,
            termtype=fundamental_termtype,
            pages=fundamental_pages,
            force_refresh=force_refresh_fundamentals,
        )
    benchmark = ohlcv_all.filter(pl.col("symbol") == benchmark_symbol)
    ohlcv = ohlcv_all.filter(pl.col("symbol") != benchmark_symbol)
    return ohlcv.sort(["symbol", "date"]), benchmark.sort(["symbol", "date"]), failures


def fundamental_cache_report(symbols: list[str], termtype: int = 1) -> pl.DataFrame:
    label = "annual" if termtype == 1 else "quarterly"
    rows = []
    for symbol in symbols:
        ticker = symbol.split(":", 1)[1] if ":" in symbol else symbol
        path = cache_dir() / "vn_fundamentals" / f"{ticker}_{label}.parquet"
        rows.append(
            {
                "symbol": symbol,
                "cached": path.exists(),
                "path": str(path),
                "rows": pl.read_parquet(path).height if path.exists() else 0,
            }
        )
    return pl.DataFrame(rows)


__all__ = [
    "fetch_ohlcv_resilient",
    "fetch_prices_and_fundamentals",
    "fundamental_cache_report",
    "load_vn100_symbols",
    "make_vn_manager",
    "normalize_vn_symbol",
]
