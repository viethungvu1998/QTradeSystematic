#!/usr/bin/env python3
"""Download, train, infer, and backtest a VN100 stock factor classifier end to end."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
from calendar import monthrange
from dataclasses import dataclass, replace
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Callable, Literal, Protocol

import matplotlib
import numpy as np
import pandas as pd
import polars as pl
from dateutil.relativedelta import relativedelta
from matplotlib.backends.backend_pdf import PdfPages

try:
    import optuna
except ImportError:  # pragma: no cover - local environment dependent
    optuna = None

from qts.data.sources.vnstock import VnstockDataSource
from qts.data.vn100_quantamental import load_vn100_symbols
from qts.research.backtest.base import (
    BacktestConfig,
    BacktestResult as QTSBacktestResult,
    CommissionConfig,
    UniverseConfig as QTSUniverseConfig,
)
from qts.research.backtest.engines.vectorbtpro_engine import VectorBTProEngine
from qts.research.backtest.metrics import build_metrics
from qts.research.backtest.pyfolio_adapter import positions_frame, returns_series, transactions_frame
from qts.research.backtest.tearsheet import save_tearsheet
from qts.research.features.fundamentals import FMP_ALIASES, KBS_FUNDAMENTAL_ITEMS
from qts.research.strategies.base import BaseStrategy

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLASS_LABELS = {
    0: "Strong Bear",
    1: "Weak Bear",
    2: "No Change",
    3: "Weak Bull",
    4: "Strong Bull",
}
FUNDAMENTAL_AUX_COLUMNS = {
    "symbol",
    "fiscal_year",
    "quarter",
    "period_end",
    "report_date",
    "available_from",
}
OHLCV_COLUMNS = {"open", "high", "low", "close", "volume"}
MIN_FUNDAMENTAL_YEAR = 2018
VCI_STATEMENT_ITEMS = {
    "KQKD": {
        "revenue": ["Sales"],
        "netRevenue": ["Net sales"],
        "grossProfit": ["Gross Profit", "Gross profit"],
        "profitBeforeTax": ["Net accounting profit/(loss) before tax"],
        "netIncome": ["Net profit/(loss) after tax"],
    },
    "CDKT": {
        "totalAssets": ["Total Assets", "TOTAL ASSETS"],
        "totalLiabilities": ["Liabilities", "TOTAL LIABILITIES"],
        "totalEquity": ["Owner's Equity", "Owner's equity", "OWNER'S EQUITY"],
        "cashAndCashEquivalents": ["Cash and cash equivalents"],
        "shortTermAssets": ["CURRENT ASSETS", "Current assets"],
        "longTermAssets": ["LONG-TERM ASSETS", "Long-term assets"],
        "shortTermLiabilities": ["Current liabilities", "Short-term liabilities"],
        "longTermLiabilities": ["Long-term liabilities"],
    },
    "LCTT": {
        "operatingCashFlow": ["Net cash flows from operating activities"],
        "investingCashFlow": ["Net cash flows from investing activities"],
        "financingCashFlow": ["Net cash flows from financing activities"],
        "netCashFlow": ["Net increase in cash and cash equivalents", "Net cash flows during the period"],
        "cashEndPeriod": ["Cash and cash equivalents at the end of period"],
    },
}


def _build_canonical_item_lookup() -> dict[tuple[str, str], str]:
    lookup = {
        (report_type, item_name): column
        for report_type, feature_map in KBS_FUNDAMENTAL_ITEMS.items()
        for column, item_names in feature_map.items()
        for item_name in item_names
    }
    for report_type, feature_map in VCI_STATEMENT_ITEMS.items():
        for column, item_names in feature_map.items():
            for item_name in item_names:
                lookup[(report_type, item_name)] = column
    for source_name, target_name in FMP_ALIASES.items():
        lookup[("CSTC", source_name)] = target_name
        lookup[("CSTC", source_name.lower())] = target_name
        lookup[("CSTC", target_name)] = target_name
    lookup.update(
        {
            ("CSTC", "debtToEquity"): "debtToEquityRatio",
            ("CSTC", "debtPerEquity"): "debtToEquityRatio",
            ("CSTC", "grossMargin"): "grossProfitMargin",
            ("CSTC", "preTaxProfitMargin"): "preTaxProfitMargin",
            ("CSTC", "afterTaxProfitMargin"): "netProfitMargin",
            ("CSTC", "currentRatio"): "currentRatio",
            ("CSTC", "quickRatio"): "quickRatio",
            ("CSTC", "cashRatio"): "cashRatio",
        }
    )
    return lookup


CANONICAL_ITEM_LOOKUP = _build_canonical_item_lookup()
SEARCH_SPACE: dict[str, list[object]] = {
    "train_window": [60, 100],
    "target_horizon": [15],
    "max_positions": [ 5],
    "max_depth": [8],
    "learning_rate": [0.03, 0.01, 0.1],
    "signal_class": [4],
}

_LOG = logging.getLogger("end2end_factor")
PREPARED_DATA_SCHEMA_VERSION = 1


def setup_logging(output_dir: Path | None = None) -> None:
    """Configure console handler (INFO) and optional file handler (DEBUG)."""
    _LOG.setLevel(logging.DEBUG)
    if _LOG.handlers:
        return
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    _LOG.addHandler(ch)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(output_dir / "pipeline.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        _LOG.addHandler(fh)


def _save_debug(df: pd.DataFrame | None, path: Path, label: str) -> None:
    """Save a DataFrame to CSV for post-run inspection; silently skips if empty."""
    if df is None or df.empty:
        _LOG.debug("_save_debug: skipping '%s' — empty", label)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    _LOG.debug("Saved debug artifact '%s' → %s (%d rows)", label, path.name, len(df))


class MissingFundamentalsError(FileNotFoundError):
    """Raised when annual or quarterly fundamentals are incomplete after download."""

    def __init__(self, freq: str, missing_files: list[str]) -> None:
        self.freq = freq
        self.missing_files = missing_files
        preview = ", ".join(missing_files[:5])
        suffix = "" if len(missing_files) <= 5 else f", ... (+{len(missing_files) - 5} more)"
        super().__init__(f"Missing {freq} fundamentals cache files: {preview}{suffix}")


@dataclass(frozen=True)
class UniverseConfig:
    name: str
    symbols_path: Path
    runtime_root: Path

    @property
    def db_path(self) -> Path:
        return self.runtime_root / "vn100.duckdb"

    @property
    def fundamental_cache_dir(self) -> Path:
        return self.runtime_root / "cache" / "vn_fundamentals"


@dataclass(frozen=True)
class RunConfig:
    universe: UniverseConfig
    download_start_date: date
    optimize_end_date: date
    inference_start_date: date
    inference_end_date: date
    target_horizon: int
    train_window: int
    rebalance_period: int
    max_positions: int
    signal_class: int
    commission: float
    initial_capital: float
    max_drawdown_limit: float
    n_trials: int
    seed: int
    price_interval: str
    batch_size: int
    fundamental_pages: int
    force_refresh_fundamentals: bool
    output_dir: Path
    prepared_data_dir: Path | None
    reuse_prepared_data: bool


@dataclass(frozen=True)
class BacktestResult:
    cagr: float
    max_drawdown: float
    sharpe: float
    calmar: float
    equity_curve: pd.Series
    weights: pd.DataFrame
    n_rebalances: int
    avg_positions: float
    avg_cash_pct: float
    qts_result: QTSBacktestResult


@dataclass(frozen=True)
class OptunaResult:
    study: object
    best_result: BacktestResult
    best_params: dict[str, object]


class FeatureBuilder(Protocol):
    def build(self, df: pd.DataFrame) -> pd.DataFrame: ...


class FundamentalTransformer(Protocol):
    def transform(self, wide_df: pd.DataFrame) -> pd.DataFrame: ...


class LabelScheme(Protocol):
    def encode(self, forward_returns: pd.DataFrame) -> pd.DataFrame: ...


class FundamentalFilter(Protocol):
    def mask(self, df: pd.DataFrame) -> pd.Series: ...


class Predictor(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None: ...

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame: ...


class Weighter(Protocol):
    def compute(
        self,
        selected: list[str],
        max_positions: int,
        prices: pd.DataFrame,
        scores: pd.Series,
    ) -> dict[str, float]: ...


ObjectiveFn = Callable[[BacktestResult], float]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def default_universe_config(repo_root: Path) -> UniverseConfig:
    return UniverseConfig(
        name="vn100",
        symbols_path=repo_root / "configs" / "assets" / "vn100.yml",
        runtime_root=repo_root / ".qts_notebook_runtime" / "live_vn100",
    )


def universe_symbols(universe: UniverseConfig) -> tuple[list[str], str]:
    return load_vn100_symbols(universe.symbols_path, max_symbols=None)


def default_prepared_data_dir(universe: UniverseConfig) -> Path:
    return universe.runtime_root / "prepared_end2end_stock_factor"


def prepared_data_key(config: RunConfig) -> str:
    return (
        f"{config.universe.name}"
        f"_start-{config.download_start_date.isoformat()}"
        f"_opt-{config.optimize_end_date.isoformat()}"
        f"_infer-{config.inference_start_date.isoformat()}-{config.inference_end_date.isoformat()}"
        f"_interval-{config.price_interval}"
    )


def resolve_prepared_data_dir(config: RunConfig) -> Path:
    base_dir = config.prepared_data_dir or default_prepared_data_dir(config.universe)
    return base_dir / prepared_data_key(config)


def strip_symbol_prefix(symbol: str) -> str:
    return symbol.split(":", 1)[1] if ":" in symbol else symbol


def with_vn_prefix(symbol: str) -> str:
    return symbol if ":" in symbol else f"VN:{symbol}"


def _ohlcv_cache_candidates(cache_root: Path, symbol: str, interval: str) -> list[Path]:
    return sorted(cache_root.glob(f"ohlcv_{symbol}_end-*_interval-{interval}_start-*.parquet"))


def _parse_ohlcv_cache_bounds(path: Path, symbol: str, interval: str) -> tuple[date, date] | None:
    pattern = re.compile(
        rf"^ohlcv_{re.escape(symbol)}_end-(\d{{4}}-\d{{2}}-\d{{2}})_interval-{re.escape(interval)}_start-(\d{{4}}-\d{{2}}-\d{{2}})$"
    )
    match = pattern.match(path.stem)
    if match is None:
        return None
    end_str, start_str = match.groups()
    return date.fromisoformat(start_str), date.fromisoformat(end_str)


def _load_cached_ohlcv(
    cache_root: Path,
    symbol: str,
    start_date: date,
    end_date: date,
    interval: str,
) -> pl.DataFrame | None:
    covering_paths: list[tuple[Path, date, date]] = []
    for path in _ohlcv_cache_candidates(cache_root, symbol, interval):
        bounds = _parse_ohlcv_cache_bounds(path, symbol, interval)
        if bounds is None:
            continue
        cached_start, cached_end = bounds
        if cached_start <= start_date and cached_end >= end_date:
            covering_paths.append((path, cached_start, cached_end))
    if not covering_paths:
        return None

    selected_path, _, _ = min(
        covering_paths,
        key=lambda item: ((item[2] - item[1]).days, item[1]),
    )
    frame = pl.read_parquet(selected_path)
    return frame.filter(pl.col("date").is_between(start_date, end_date))


def _ohlcv_cache_path(cache_root: Path, symbol: str, start_date: date, end_date: date, interval: str) -> Path:
    return cache_root / f"ohlcv_{symbol}_end-{end_date.isoformat()}_interval-{interval}_start-{start_date.isoformat()}.parquet"


def _write_ohlcv_cache(
    cache_root: Path,
    frame: pl.DataFrame,
    *,
    symbol: str,
    start_date: date,
    end_date: date,
    interval: str,
) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(_ohlcv_cache_path(cache_root, symbol, start_date, end_date, interval))


def _load_ohlcv_from_db(
    runtime_root: Path,
    symbol: str,
    start_date: date,
    end_date: date,
) -> pl.DataFrame | None:
    db_path = runtime_root / "vn100.duckdb"
    if not db_path.exists():
        return None

    try:
        import duckdb  # noqa: PLC0415

        connection = duckdb.connect(str(db_path), read_only=True)
    except Exception:  # pragma: no cover - environment/locking dependent
        return None

    try:
        table_exists = connection.execute(
            "select count(*) from information_schema.tables where table_name = 'vn_stock_prices'"
        ).fetchone()
        if not table_exists or not table_exists[0]:
            return None
        frame = connection.execute(
            """
            select *
            from vn_stock_prices
            where symbol = ?
              and date >= ?
              and date <= ?
            order by symbol, date
            """,
            [symbol, start_date, end_date],
        ).pl()
    finally:
        connection.close()

    return frame if frame.height > 0 else None


def _missing_fundamental_symbols(
    cache_dir: Path,
    symbols: list[str],
    *,
    freq: Literal["annual", "quarterly"],
) -> list[str]:
    missing: list[str] = []
    for symbol in symbols:
        path = _fundamental_cache_path(cache_dir, symbol, freq)
        if not path.exists() or not _fundamental_cache_covers_year(path, freq=freq, min_year=MIN_FUNDAMENTAL_YEAR):
            missing.append(symbol)
    return missing


def _fundamental_cache_path(
    cache_dir: Path,
    symbol: str,
    freq: Literal["annual", "quarterly"],
) -> Path:
    return cache_dir / f"{strip_symbol_prefix(symbol)}_{freq}.parquet"


def _fundamental_cache_covers_year(
    path: Path,
    *,
    freq: Literal["annual", "quarterly"],
    min_year: int,
) -> bool:
    try:
        columns = ["fiscal_year"]
        if freq == "quarterly":
            columns.append("quarter")
        frame = pl.read_parquet(path, columns=columns)
    except Exception:  # pragma: no cover - corrupt cache should trigger refetch
        return False
    if frame.is_empty() or "fiscal_year" not in frame.columns:
        return False
    min_fiscal_year = frame.select(pl.col("fiscal_year").cast(pl.Int64).min()).item()
    if min_fiscal_year is None or int(min_fiscal_year) > min_year:
        return False
    if freq == "annual":
        return True
    covered = frame.filter(
        (pl.col("fiscal_year").cast(pl.Int64) == min_year)
        & pl.col("quarter").cast(pl.Int64, strict=False).is_in([1, 2, 3, 4])
    )
    return not covered.is_empty()


async def _fetch_ohlcv_for_missing_symbols(
    symbols: list[str],
    *,
    start_date: date,
    end_date: date,
    interval: str,
) -> tuple[pl.DataFrame, list[tuple[str, str]]]:
    source = VnstockDataSource.from_env()
    frames: list[pl.DataFrame] = []
    failures: list[tuple[str, str]] = []
    for symbol in symbols:
        try:
            frame = await source.get_ohlcv(symbol, start_date, end_date, interval)
            if not frame.is_empty():
                frames.append(frame)
        except Exception as exc:  # pragma: no cover - network/source dependent
            failures.append((symbol, str(exc)))
    if not frames:
        return pl.DataFrame(), failures
    combined = pl.concat(frames, how="vertical_relaxed").unique(subset=["date", "symbol"], keep="last")
    return combined.sort(["symbol", "date"]), failures


async def _fetch_missing_fundamentals(
    symbols: list[str],
    *,
    termtype: int,
    pages: int,
    force_refresh: bool,
) -> None:
    source = VnstockDataSource.from_env(fundamentals_source="vci")
    for symbol in symbols:
        await source.get_fundamentals(
            symbol,
            termtype=termtype,
            pages=pages,
            force_refresh=force_refresh,
        )


async def download_market_data(config: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    os.environ["QTS_ROOT"] = str(config.universe.runtime_root)
    symbols, benchmark_symbol = universe_symbols(config.universe)
    requested_symbols = list(dict.fromkeys([*symbols, benchmark_symbol]))
    config.universe.runtime_root.mkdir(parents=True, exist_ok=True)
    cache_root = config.universe.runtime_root / "cache"

    cached_frames: list[pl.DataFrame] = []
    missing_price_symbols: list[str] = []
    for symbol in requested_symbols:
        cached = _load_cached_ohlcv(
            cache_root,
            symbol,
            config.download_start_date,
            config.inference_end_date,
            config.price_interval,
        )
        if cached is not None and not cached.is_empty():
            cached_frames.append(cached)
            continue

        db_frame = _load_ohlcv_from_db(
            config.universe.runtime_root,
            symbol,
            config.download_start_date,
            config.inference_end_date,
        )
        if db_frame is not None and not db_frame.is_empty():
            cached_frames.append(db_frame)
            _write_ohlcv_cache(
                cache_root,
                db_frame,
                symbol=symbol,
                start_date=config.download_start_date,
                end_date=config.inference_end_date,
                interval=config.price_interval,
            )
            continue

        missing_price_symbols.append(symbol)

    if missing_price_symbols:
        _LOG.info(
            "Price cache hit for %d/%d symbols; fetching %d missing symbols",
            len(requested_symbols) - len(missing_price_symbols),
            len(requested_symbols),
            len(missing_price_symbols),
        )
        fetched_ohlcv, failures = await _fetch_ohlcv_for_missing_symbols(
            symbols=missing_price_symbols,
            start_date=config.download_start_date,
            end_date=config.inference_end_date,
            interval=config.price_interval,
        )
        if failures:
            preview = ", ".join(f"{symbol}: {error}" for symbol, error in failures[:5])
            raise RuntimeError(f"Price download failed for {len(failures)} symbols: {preview}")
        if not fetched_ohlcv.is_empty():
            for symbol in missing_price_symbols:
                symbol_frame = fetched_ohlcv.filter(pl.col("symbol") == symbol)
                if symbol_frame.is_empty():
                    continue
                _write_ohlcv_cache(
                    cache_root,
                    symbol_frame,
                    symbol=symbol,
                    start_date=config.download_start_date,
                    end_date=config.inference_end_date,
                    interval=config.price_interval,
                )
            cached_frames.append(fetched_ohlcv)
    else:
        _LOG.info("Price cache hit for all %d requested symbols; skipping price download", len(requested_symbols))

    if not cached_frames:
        ohlcv_all = pl.DataFrame()
    else:
        ohlcv_all = (
            pl.concat(cached_frames, how="vertical_relaxed")
            .unique(subset=["date", "symbol"], keep="last")
            .sort(["symbol", "date"])
        )

    equity_symbols = [symbol for symbol in symbols if symbol != benchmark_symbol]
    annual_missing = (
        equity_symbols
        if config.force_refresh_fundamentals
        else _missing_fundamental_symbols(config.universe.fundamental_cache_dir, equity_symbols, freq="annual")
    )
    quarterly_missing = (
        equity_symbols
        if config.force_refresh_fundamentals
        else _missing_fundamental_symbols(config.universe.fundamental_cache_dir, equity_symbols, freq="quarterly")
    )

    if annual_missing:
        _LOG.info(
            "Fetching VCI annual fundamentals for %d symbols missing 2018+ cache coverage",
            len(annual_missing),
        )
        await _fetch_missing_fundamentals(
            annual_missing,
            termtype=1,
            pages=config.fundamental_pages,
            force_refresh=config.force_refresh_fundamentals,
        )
    else:
        _LOG.info(
            "Annual fundamentals cache covers %d+ for all %d equity symbols; skipping download",
            MIN_FUNDAMENTAL_YEAR,
            len(equity_symbols),
        )

    if quarterly_missing:
        _LOG.info(
            "Fetching VCI quarterly fundamentals for %d symbols missing 2018+ cache coverage",
            len(quarterly_missing),
        )
        await _fetch_missing_fundamentals(
            quarterly_missing,
            termtype=2,
            pages=config.fundamental_pages,
            force_refresh=config.force_refresh_fundamentals,
        )
    else:
        _LOG.info(
            "Quarterly fundamentals cache covers %d+ for all %d equity symbols; skipping download",
            MIN_FUNDAMENTAL_YEAR,
            len(equity_symbols),
        )

    if ohlcv_all.is_empty():
        raise ValueError("No OHLCV rows were downloaded.")

    ohlcv_pdf = (
        ohlcv_all.filter(pl.col("symbol") != benchmark_symbol)
        .sort(["date", "symbol"])
        .to_pandas()
    )
    benchmark_pdf = (
        ohlcv_all.filter(pl.col("symbol") == benchmark_symbol)
        .sort("date")
        .to_pandas()
    )
    if ohlcv_pdf.empty:
        raise ValueError("No equity OHLCV rows remain after removing the benchmark.")

    ohlcv_pdf["date"] = pd.to_datetime(ohlcv_pdf["date"])
    if not benchmark_pdf.empty:
        benchmark_pdf["date"] = pd.to_datetime(benchmark_pdf["date"])
    prices = ohlcv_pdf.set_index(["date", "symbol"]).sort_index()
    return prices, benchmark_pdf


def _period_end_from_row(freq: Literal["annual", "quarterly"], fiscal_year: int, quarter: float) -> date:
    if freq == "annual":
        return date(int(fiscal_year), 12, 31)
    month = {1: 3, 2: 6, 3: 9, 4: 12}[int(quarter)]
    return date(int(fiscal_year), month, monthrange(int(fiscal_year), month)[1])


def load_fundamentals(
    universe: UniverseConfig,
    freq: Literal["annual", "quarterly"],
) -> pd.DataFrame:
    symbols, benchmark_symbol = universe_symbols(universe)
    label = "annual" if freq == "annual" else "quarterly"
    paths = [
        universe.fundamental_cache_dir / f"{strip_symbol_prefix(symbol)}_{label}.parquet"
        for symbol in symbols
        if symbol != benchmark_symbol
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise MissingFundamentalsError(freq, missing)

    frames = [pl.read_parquet(path) for path in paths]
    combined = pl.concat(frames, how="vertical_relaxed")
    pdf = combined.to_pandas()
    pdf["symbol"] = pdf["symbol"].map(with_vn_prefix)
    pdf["report_date"] = pd.to_datetime(pdf["report_date"])
    quarter_series = pdf["quarter"] if "quarter" in pdf.columns else pd.Series(np.nan, index=pdf.index)
    pdf["period_end"] = [
        _period_end_from_row(freq, fiscal_year, quarter if pd.notna(quarter) else math.nan)
        for fiscal_year, quarter in zip(pdf["fiscal_year"], quarter_series, strict=True)
    ]
    columns = ["symbol", "fiscal_year", "period_end", "report_date", "report_type", "item_en", "value"]
    if freq == "quarterly":
        columns.insert(2, "quarter")
    return pdf[columns].copy()


def compute_available_from(period_end: pd.Series) -> pd.Series:
    return (
        pd.to_datetime(period_end)
        .add(pd.DateOffset(months=5))
        .dt.to_period("M")
        .dt.to_timestamp()
    )


def pivot_fundamentals(
    df_long: pd.DataFrame,
    freq: Literal["annual", "quarterly"],
) -> pd.DataFrame:
    long_df = df_long.copy()
    long_df["metric"] = [
        CANONICAL_ITEM_LOOKUP.get((report_type, item_en))
        for report_type, item_en in zip(long_df["report_type"], long_df["item_en"], strict=True)
    ]
    long_df = long_df.dropna(subset=["metric"])
    if long_df.empty:
        return pd.DataFrame()

    long_df["available_from"] = compute_available_from(long_df["period_end"])
    index_columns = ["symbol", "fiscal_year", "period_end", "available_from"]
    sort_columns = ["symbol", "fiscal_year", "period_end", "report_date", "metric"]
    if freq == "quarterly":
        index_columns.insert(2, "quarter")
        sort_columns.insert(2, "quarter")
    deduped = (
        long_df.sort_values(sort_columns)
        .groupby([*index_columns, "metric"], as_index=False)["value"]
        .last()
    )
    wide = (
        deduped.pivot_table(
            index=index_columns,
            columns="metric",
            values="value",
            aggfunc="last",
        )
        .reset_index()
        .sort_values(["symbol", "available_from"])
        .reset_index(drop=True)
    )
    wide.columns = [str(column) for column in wide.columns]
    return wide


def _pct_change_over_abs(series: pd.Series, lag: int) -> pd.Series:
    previous = series.shift(lag)
    change = (series - previous) / previous.abs()
    return change.where(previous.abs() > 1e-12)


class YoYAnnualTransformer:
    def __init__(self, metrics: list[str]) -> None:
        self.metrics = metrics

    def transform(self, wide_df: pd.DataFrame) -> pd.DataFrame:
        result = wide_df.copy()
        sorted_df = result.sort_values(["symbol", "available_from"])
        for metric in self.metrics:
            if metric not in sorted_df.columns:
                continue
            result[f"{metric}_yoy"] = (
                sorted_df.groupby("symbol", group_keys=False)[metric]
                .apply(lambda series: _pct_change_over_abs(series, 1))
                .sort_index()
            )
        return result


class QuarterlyQoQTransformer:
    def __init__(self, metrics: list[str]) -> None:
        self.metrics = metrics

    def transform(self, wide_df: pd.DataFrame) -> pd.DataFrame:
        result = wide_df.copy()
        sorted_df = result.sort_values(["symbol", "available_from"])
        for metric in self.metrics:
            if metric not in sorted_df.columns:
                continue
            result[f"{metric}_qoq"] = (
                sorted_df.groupby("symbol", group_keys=False)[metric]
                .apply(lambda series: _pct_change_over_abs(series, 1))
                .sort_index()
            )
        return result


class SameQuarterYoYTransformer:
    def __init__(self, metrics: list[str]) -> None:
        self.metrics = metrics

    def transform(self, wide_df: pd.DataFrame) -> pd.DataFrame:
        result = wide_df.copy()
        sorted_df = result.sort_values(["symbol", "available_from"])
        for metric in self.metrics:
            if metric not in sorted_df.columns:
                continue
            result[f"{metric}_q_yoy"] = (
                sorted_df.groupby("symbol", group_keys=False)[metric]
                .apply(lambda series: _pct_change_over_abs(series, 4))
                .sort_index()
            )
        return result


class TTMTransformer:
    def __init__(self, metrics: list[str]) -> None:
        self.metrics = metrics

    def transform(self, wide_df: pd.DataFrame) -> pd.DataFrame:
        result = wide_df.copy()
        sorted_df = result.sort_values(["symbol", "available_from"])
        for metric in self.metrics:
            if metric not in sorted_df.columns:
                continue
            result[f"{metric}_ttm"] = (
                sorted_df.groupby("symbol", group_keys=False)[metric]
                .apply(lambda series: series.rolling(4, min_periods=4).sum())
                .sort_index()
            )
        return result


def build_fundamental_features(
    annual_wide: pd.DataFrame,
    quarterly_wide: pd.DataFrame,
    annual_transformers: list[FundamentalTransformer],
    quarterly_transformers: list[FundamentalTransformer],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual_features = _add_derived_fundamental_metrics(annual_wide)
    quarterly_features = _add_derived_fundamental_metrics(quarterly_wide)
    for transformer in annual_transformers:
        annual_features = transformer.transform(annual_features)
    for transformer in quarterly_transformers:
        quarterly_features = transformer.transform(quarterly_features)
    return (
        annual_features.drop(columns=["period_end"], errors="ignore"),
        quarterly_features.drop(columns=["period_end"], errors="ignore"),
    )


def _add_derived_fundamental_metrics(wide_df: pd.DataFrame) -> pd.DataFrame:
    if wide_df.empty:
        return wide_df.copy()
    result = wide_df.copy()
    if (
        "debtToAssetsRatio" not in result.columns
        and {"totalLiabilities", "totalAssets"}.issubset(result.columns)
    ):
        denominator = result["totalAssets"].replace(0.0, np.nan)
        result["debtToAssetsRatio"] = (result["totalLiabilities"] / denominator) * 100.0
    return result


def _rename_block(frame: pd.DataFrame, suffix: str) -> tuple[pd.DataFrame, list[str]]:
    metric_columns = [column for column in frame.columns if column not in FUNDAMENTAL_AUX_COLUMNS]
    renamed = frame.rename(columns={column: f"{column}{suffix}" for column in metric_columns})
    return renamed, [f"{column}{suffix}" for column in metric_columns]


def merge_fundamentals_asof(
    prices: pd.DataFrame,
    annual_features: pd.DataFrame,
    quarterly_features: pd.DataFrame,
) -> pd.DataFrame:
    annual_block, annual_metric_columns = _rename_block(annual_features, "_a")
    quarterly_block, quarterly_metric_columns = _rename_block(quarterly_features, "_q")
    price_df = prices.reset_index().sort_values(["symbol", "date"])
    price_df["date"] = pd.to_datetime(price_df["date"]).astype("datetime64[ns]")
    if not annual_block.empty:
        annual_block = annual_block.copy()
        annual_block["available_from"] = pd.to_datetime(annual_block["available_from"]).astype("datetime64[ns]")
    if not quarterly_block.empty:
        quarterly_block = quarterly_block.copy()
        quarterly_block["available_from"] = pd.to_datetime(quarterly_block["available_from"]).astype("datetime64[ns]")
    merged_frames: list[pd.DataFrame] = []

    for symbol, group in price_df.groupby("symbol", sort=False):
        merged = group.sort_values("date").copy()
        annual_symbol = annual_block.loc[annual_block["symbol"] == symbol].sort_values("available_from")
        if annual_symbol.empty:
            for column in annual_metric_columns:
                merged[column] = np.nan
        else:
            annual_right = annual_symbol.drop(columns=["symbol", "fiscal_year"], errors="ignore")
            merged = pd.merge_asof(
                merged,
                annual_right.sort_values("available_from"),
                left_on="date",
                right_on="available_from",
                direction="backward",
            ).drop(columns=["available_from"], errors="ignore")

        quarterly_symbol = quarterly_block.loc[quarterly_block["symbol"] == symbol].sort_values("available_from")
        if quarterly_symbol.empty:
            for column in quarterly_metric_columns:
                merged[column] = np.nan
        else:
            quarterly_right = quarterly_symbol.drop(
                columns=["symbol", "fiscal_year", "quarter"],
                errors="ignore",
            )
            merged = pd.merge_asof(
                merged,
                quarterly_right.sort_values("available_from"),
                left_on="date",
                right_on="available_from",
                direction="backward",
            ).drop(columns=["available_from"], errors="ignore")

        merged_frames.append(merged)

    merged = pd.concat(merged_frames, ignore_index=True).sort_values(["date", "symbol"])
    annual_available = (
        merged[annual_metric_columns].notna().any(axis=1)
        if annual_metric_columns
        else pd.Series(False, index=merged.index)
    )
    quarterly_available = (
        merged[quarterly_metric_columns].notna().any(axis=1)
        if quarterly_metric_columns
        else pd.Series(False, index=merged.index)
    )
    # Keep rows when either annual or quarterly fundamentals are available.
    merged = merged.loc[annual_available | quarterly_available]
    return merged.set_index(["date", "symbol"]).sort_index()


class MomentumFeatures:
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        def _per_symbol(group: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "mom_21": np.log(group["close"] / group["close"].shift(21)),
                    "mom_63": np.log(group["close"] / group["close"].shift(63)),
                    "mom_126": np.log(group["close"] / group["close"].shift(126)),
                    "mom_252": np.log(group["close"] / group["close"].shift(252)),
                },
                index=group.index,
            )

        return (
            df.groupby(level="symbol", group_keys=False, sort=False)
            .apply(_per_symbol)
            .sort_index()
        )


class VolatilityFeatures:
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        def _per_symbol(group: pd.DataFrame) -> pd.DataFrame:
            close = group["close"]
            high = group["high"]
            low = group["low"]
            log_ret_1 = np.log(close / close.shift(1))
            prev_close = close.shift(1)
            true_range = pd.concat(
                [
                    (high - low),
                    (high - prev_close).abs(),
                    (low - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            vol_21 = log_ret_1.rolling(21, min_periods=21).std()
            vol_63 = log_ret_1.rolling(63, min_periods=63).std()
            atr_14 = true_range.rolling(14, min_periods=14).mean()
            return pd.DataFrame(
                {
                    "vol_21": vol_21,
                    "vol_63": vol_63,
                    "atr_14_norm": atr_14 / close.replace(0.0, np.nan),
                    "vol_ratio_21_63": vol_21 / vol_63.replace(0.0, np.nan),
                },
                index=group.index,
            )

        return (
            df.groupby(level="symbol", group_keys=False, sort=False)
            .apply(_per_symbol)
            .sort_index()
        )


class TrendFeatures:
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        def _per_symbol(group: pd.DataFrame) -> pd.DataFrame:
            close = group["close"]
            ma_50 = close.rolling(50, min_periods=50).mean()
            ma_200 = close.rolling(200, min_periods=200).mean()
            return pd.DataFrame(
                {
                    "log_close_to_ma_50": np.log(close / ma_50),
                    "log_close_to_ma_200": np.log(close / ma_200),
                    "ma_cross": np.log(ma_50 / ma_200),
                },
                index=group.index,
            )

        return (
            df.groupby(level="symbol", group_keys=False, sort=False)
            .apply(_per_symbol)
            .sort_index()
        )


class LiquidityFeatures:
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        def _per_symbol(group: pd.DataFrame) -> pd.DataFrame:
            close = group["close"]
            volume = group["volume"]
            log_ret_1 = np.log(close / close.shift(1)).abs()
            traded_value_proxy = volume.replace(0.0, np.nan)
            amihud_daily = (log_ret_1 / traded_value_proxy) * close
            return pd.DataFrame(
                {
                    "relative_volume_20": volume / volume.rolling(20, min_periods=20).mean(),
                    "amihud_illiq_21": amihud_daily.rolling(21, min_periods=21).mean(),
                },
                index=group.index,
            )

        return (
            df.groupby(level="symbol", group_keys=False, sort=False)
            .apply(_per_symbol)
            .sort_index()
        )


class FundamentalFeatures:
    """Thin wrapper that renames/selects merged fundamental columns into model-ready features.

    Column naming convention:
    - Source columns arrive with _a (annual) or _q (quarterly) suffix from merge_fundamentals_asof.
    - Output column names are source-agnostic (e.g. ``roe``, not ``roe_q``).
    - CONTRACT: do NOT recompute anything here — only rename/combine/divide.
    """

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        # Valuation & profitability ratios — prefer quarterly CSTC, fall back to annual
        features["roe"] = _combine_first(df, "returnOnEquity_q", "returnOnEquity_a")
        features["roa"] = _combine_first(df, "returnOnAssets_q", "returnOnAssets_a")
        features["pe"] = _combine_first(df, "priceToEarningsRatio_q", "priceToEarningsRatio_a")
        features["pb"] = _combine_first(df, "priceToBookRatio_q", "priceToBookRatio_a")
        features["debt_ratio"] = _combine_first(df, "debtToAssetsRatio_q", "debtToAssetsRatio_a")
        # Equity ratio = owner equity / total assets (both from annual CDKT)
        total_equity = _combine_first(df, "totalEquity_a")
        total_assets = _combine_first(df, "totalAssets_a").replace(0.0, np.nan)
        features["equity_ratio"] = total_equity / total_assets
        # TTM P&L (quarterly rolling 4-period sum)
        features["revenue_ttm"] = _combine_first(df, "revenue_ttm_q")
        features["net_income_ttm"] = _combine_first(df, "netIncome_ttm_q")
        features["profit_before_tax_ttm"] = _combine_first(df, "profitBeforeTax_ttm_q")
        # Annual YoY growth rates
        features["revenue_yoy"] = _combine_first(df, "revenue_yoy_a")
        features["net_income_yoy"] = _combine_first(df, "netIncome_yoy_a")
        features["profit_before_tax_yoy"] = _combine_first(df, "profitBeforeTax_yoy_a")
        # Quarterly same-quarter YoY (shift 4) and QoQ (shift 1)
        features["revenue_q_yoy"] = _combine_first(df, "revenue_q_yoy_q")
        features["net_income_q_yoy"] = _combine_first(df, "netIncome_q_yoy_q")
        features["profit_before_tax_q_yoy"] = _combine_first(df, "profitBeforeTax_q_yoy_q")
        features["revenue_qoq"] = _combine_first(df, "revenue_qoq_q")
        features["net_income_qoq"] = _combine_first(df, "netIncome_qoq_q")
        features["profit_before_tax_qoq"] = _combine_first(df, "profitBeforeTax_qoq_q")
        return features


def _combine_first(df: pd.DataFrame, primary: str, fallback: str | None = None) -> pd.Series:
    series = df[primary] if primary in df.columns else pd.Series(np.nan, index=df.index)
    if fallback and fallback in df.columns:
        return series.combine_first(df[fallback])
    return series


def _percent_point_threshold_in_series_units(series: pd.Series, threshold_pct_points: float) -> float:
    """Map a percentage-point threshold onto either percent or fractional ratio inputs.

    Some current CSTC payloads provide ROE/ROA-like metrics as fractions (0.15 for 15%),
    while older assumptions in this notebook treated them as percentage points (15.0).
    We detect the scale from the observed values and convert the threshold accordingly.
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return threshold_pct_points
    upper_bound = float(numeric.abs().quantile(0.99))
    if upper_bound <= 1.5:
        return threshold_pct_points / 100.0
    return threshold_pct_points


def _min_feature_observations(feature_cols: list[str]) -> int:
    if not feature_cols:
        return 0
    return max(1, math.ceil(len(feature_cols) * 0.30))


def _feature_coverage_mask(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    min_non_null_features: int | None = None,
) -> pd.Series:
    if not feature_cols:
        return pd.Series(True, index=df.index)
    threshold = min_non_null_features or _min_feature_observations(feature_cols)
    threshold = min(threshold, len(feature_cols))
    return df[feature_cols].notna().sum(axis=1) >= threshold


def build_features(df: pd.DataFrame, builders: list[FeatureBuilder]) -> pd.DataFrame:
    result = df.copy()
    seen = set(result.columns)
    for builder in builders:
        built = builder.build(df)
        duplicates = seen.intersection(built.columns)
        if duplicates:
            raise ValueError(f"Duplicate feature columns detected: {sorted(duplicates)}")
        result = pd.concat([result, built], axis=1)
        seen.update(built.columns)
    return result


class CrossSectionalQuantileLabeler:
    def __init__(self, n_classes: int = 5) -> None:
        self.n_classes = n_classes

    def encode(self, forward_returns: pd.DataFrame) -> pd.DataFrame:
        labels = pd.DataFrame(index=forward_returns.index, columns=forward_returns.columns, dtype=float)
        for index, row in forward_returns.iterrows():
            valid = row.dropna()
            if len(valid) < self.n_classes:
                continue
            ranked = valid.rank(method="first")
            labels.loc[index, valid.index] = pd.qcut(ranked, q=self.n_classes, labels=False)
        return labels


class ThresholdLabeler:
    def __init__(self, thresholds: list[float] | None = None) -> None:
        self.thresholds = thresholds or [-0.05, -0.01, 0.01, 0.05]

    def encode(self, forward_returns: pd.DataFrame) -> pd.DataFrame:
        labels = pd.DataFrame(index=forward_returns.index, columns=forward_returns.columns, dtype=float)
        for index, row in forward_returns.iterrows():
            valid = row.dropna()
            if valid.empty:
                continue
            labels.loc[index, valid.index] = np.digitize(valid.to_numpy(dtype=float), self.thresholds)
        return labels


def make_training_frame(
    df: pd.DataFrame,
    target_horizon: int,
    feature_cols: list[str],
    labeler: LabelScheme,
) -> pd.DataFrame:
    frame = df.copy().sort_index()
    frame[f"forward_log_return_{target_horizon}"] = frame["close"].groupby(level="symbol", sort=False).transform(
        lambda series: np.log(series.shift(-target_horizon) / series)
    )
    min_features = _min_feature_observations(feature_cols)
    coverage_mask = _feature_coverage_mask(frame, feature_cols, min_non_null_features=min_features)
    clean = frame.loc[coverage_mask].dropna(subset=[f"forward_log_return_{target_horizon}"]).copy()
    forward_wide = clean[f"forward_log_return_{target_horizon}"].unstack("symbol")
    label_wide = labeler.encode(forward_wide)
    # future_stack=True silences the pandas 2.1 deprecation warning
    label_long = label_wide.stack(future_stack=True).dropna().rename("label").reset_index()
    labeled = clean.reset_index().merge(label_long, on=["date", "symbol"], how="inner")
    labeled["label"] = labeled["label"].astype(int)
    _LOG.debug(
        "make_training_frame: %d labeled rows | %d symbols | horizon=%d | dropped=%d | label_dist=%s",
        len(labeled),
        labeled["symbol"].nunique(),
        target_horizon,
        len(frame) - len(labeled),
        labeled["label"].value_counts().sort_index().to_dict() if len(labeled) > 0 else "EMPTY",
    )
    if labeled.empty:
        _LOG.warning(
            "make_training_frame returned EMPTY frame — check row-level feature coverage "
            "(min_non_null_features=%d) and factor availability by date. feature_cols sample: %s",
            min_features,
            feature_cols[:10],
        )
    return labeled.set_index(["date", "symbol"]).sort_index()


class XGBPredictor:
    def __init__(
        self,
        *,
        max_depth: int = 3,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        eval_metric: str = "mlogloss",
        seed: int = 42,
    ) -> None:
        import xgboost as xgb

        self._xgb = xgb
        self.params = {
            "objective": "multi:softprob",
            "num_class": 5,
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "eval_metric": eval_metric,
            "random_state": seed,
            "n_jobs": -1,
        }
        self.model: object | None = None
        self.constant_class: int | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        y_int = pd.Series(y, copy=False).astype(int)
        unique = sorted(y_int.unique().tolist())
        if len(unique) == 1:
            self.constant_class = int(unique[0])
            self.model = None
            return
        self.constant_class = None
        self.model = self._xgb.XGBClassifier(**self.params)
        self.model.fit(X, y_int)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.constant_class is not None:
            data = np.zeros((len(X), 5), dtype=float)
            data[:, self.constant_class] = 1.0
            return pd.DataFrame(data, index=X.index, columns=list(range(5)))
        if self.model is None:
            raise RuntimeError("XGBPredictor.fit must be called before predict_proba.")
        probabilities = np.asarray(self.model.predict_proba(X), dtype=float)
        frame = pd.DataFrame(0.0, index=X.index, columns=list(range(5)))
        for index, label in enumerate(self.model.classes_):
            frame[int(label)] = probabilities[:, index]
        return frame


class ProfitabilityFilter:
    """Pass symbols with sufficient ROE.

    NaN-tolerance rules (applied per condition):
    - ``roe``: sourced from annual CSTC data.
      Current VCI/CSTC caches may provide this either as percentage points
      (12.5 means 12.5%) or as fractions (0.125 means 12.5%), so the threshold
      is converted to match the observed scale. NaN → *reject* (fillna False)
      because we want real profitability confirmation.
    - ``net_income_qoq``: sourced from quarterly data. NaN → *pass* (fillna True)
      so the filter does not block warm-up periods before four quarters are available.
    """

    def __init__(self, min_roe: float = 5.0) -> None:
        self.min_roe = min_roe

    def mask(self, df: pd.DataFrame) -> pd.Series:
        # Annual CSTC — strict: reject when below threshold or unavailable.
        roe = df.get("roe", pd.Series(np.nan, index=df.index))
        min_roe_in_series_units = _percent_point_threshold_in_series_units(roe, self.min_roe)
        roe_ok = (roe >= min_roe_in_series_units).fillna(False)
        # Quarterly — lenient: pass when data not yet available
        net_income_qoq = df.get("net_income_qoq", pd.Series(np.nan, index=df.index))
        income_ok = net_income_qoq.isna() | (net_income_qoq > -1.0)
        return roe_ok & income_ok


class QualityFilter:
    """Pass symbols with manageable leverage and positive equity.

    NaN-tolerance rules:
    - ``debt_ratio``: annual CSTC, expressed in percentage points → strict (fillna False).
    - ``equity_ratio``: annual CDKT ratio → lenient (fillna True) since it
      requires both totalEquity_a and totalAssets_a; missing during warm-up.
    """

    def __init__(self, max_debt_ratio: float = 70.0) -> None:
        self.max_debt_ratio = max_debt_ratio

    def mask(self, df: pd.DataFrame) -> pd.Series:
        debt_ok = (df.get("debt_ratio", pd.Series(np.nan, index=df.index)) <= self.max_debt_ratio).fillna(False)
        equity_ratio = df.get("equity_ratio", pd.Series(np.nan, index=df.index))
        equity_ok = equity_ratio.isna() | (equity_ratio > 0.0)
        return debt_ok & equity_ok


class GrowthFilter:
    """Pass symbols not in severe revenue decline.

    NaN-tolerance: ``revenue_q_yoy`` is quarterly-derived and remains NaN during
    warm-up until four quarters are available → fillna True to avoid blocking
    all signals early in the sample.
    """

    def __init__(self, min_revenue_q_yoy: float = -0.1) -> None:
        self.min_revenue_q_yoy = min_revenue_q_yoy

    def mask(self, df: pd.DataFrame) -> pd.Series:
        revenue_q_yoy = df.get("revenue_q_yoy", pd.Series(np.nan, index=df.index))
        growth = revenue_q_yoy.isna() | (revenue_q_yoy >= self.min_revenue_q_yoy)
        return growth


class CompositeFilter:
    def __init__(self, filters: list[FundamentalFilter]) -> None:
        self.filters = filters

    def mask(self, df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(True, index=df.index)
        for filter_item in self.filters:
            mask &= filter_item.mask(df)
        return mask.fillna(False)


def apply_filter(df: pd.DataFrame, filter_set: FundamentalFilter) -> pd.DataFrame:
    return df.loc[filter_set.mask(df)].copy()


class FixedSlotEqualWeighter:
    def compute(
        self,
        selected: list[str],
        max_positions: int,
        prices: pd.DataFrame,
        scores: pd.Series,
    ) -> dict[str, float]:
        if max_positions <= 0:
            return {}
        return {symbol: 1.0 / max_positions for symbol in selected}


def walk_forward_predict(
    df: pd.DataFrame,
    predictor: Predictor,
    config: RunConfig,
    feature_cols: list[str],
    prediction_start_date: date,
    prediction_end_date: date,
) -> pd.DataFrame:
    labeled = df.reset_index().sort_values(["date", "symbol"]).copy()
    all_dates = sorted(pd.to_datetime(labeled["date"]).drop_duplicates().tolist())
    rebalance_dates = [
        rebalance_date
        for index, rebalance_date in enumerate(all_dates)
        if index % config.rebalance_period == 0
        and prediction_start_date <= rebalance_date.date() <= prediction_end_date
    ]
    rows: list[pd.DataFrame] = []
    _LOG.info(
        "walk_forward_predict: %d labeled rows | %d rebalance dates in [%s, %s]",
        len(labeled),
        len(rebalance_dates),
        prediction_start_date,
        prediction_end_date,
    )
    skipped_train = skipped_predict = 0
    min_features = _min_feature_observations(feature_cols)

    for rebalance_date in rebalance_dates:
        idx = all_dates.index(rebalance_date)
        latest_known_label_index = idx - config.target_horizon
        if latest_known_label_index < 0:
            continue
        trainable_dates = all_dates[: latest_known_label_index + 1]
        if len(trainable_dates) < config.train_window:
            skipped_train += 1
            continue
        train_dates = trainable_dates[-config.train_window :]
        train = labeled.loc[labeled["date"].isin(train_dates)].dropna(subset=["label"])
        train = train.loc[_feature_coverage_mask(train, feature_cols, min_non_null_features=min_features)]
        predict_slice = labeled.loc[labeled["date"] == rebalance_date]
        predict_slice = predict_slice.loc[
            _feature_coverage_mask(predict_slice, feature_cols, min_non_null_features=min_features)
        ]
        if train.empty or predict_slice.empty:
            skipped_predict += 1
            _LOG.debug(
                "  skip %s — train=%d predict=%d",
                rebalance_date.date(),
                len(train),
                len(predict_slice),
            )
            continue
        predictor.fit(train[feature_cols], train["label"])
        probabilities = predictor.predict_proba(predict_slice[feature_cols])
        pred_class = probabilities.idxmax(axis=1)
        class_dist = pred_class.value_counts().sort_index().to_dict()
        n_class4 = int((pred_class == 4).sum())
        _LOG.debug(
            "  rebalance %s | train=%d | predict=%d symbols | class_dist=%s | class4=%d",
            rebalance_date.date(),
            len(train),
            len(predict_slice),
            class_dist,
            n_class4,
        )
        rows.append(
            pd.DataFrame(
                {
                    "date": predict_slice["date"].to_numpy(),
                    "symbol": predict_slice["symbol"].to_numpy(),
                    "pred_class": pred_class.to_numpy(dtype=int),
                    "prob_strong_bull": probabilities[4].to_numpy(dtype=float),
                }
            )
        )

    if skipped_train:
        _LOG.warning("walk_forward_predict: %d dates skipped — insufficient train window (%d required)", skipped_train, config.train_window)
    if skipped_predict:
        _LOG.warning("walk_forward_predict: %d dates skipped — empty train or predict slice", skipped_predict)
    if not rows:
        _LOG.warning(
            "walk_forward_predict: NO predictions generated. "
            "labeled_rows=%d, rebalance_dates=%d, skipped_train=%d, skipped_predict=%d",
            len(labeled),
            len(rebalance_dates),
            skipped_train,
            skipped_predict,
        )
        return pd.DataFrame(columns=["date", "symbol", "pred_class", "prob_strong_bull"])
    result = pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    class4_pct = (result["pred_class"] == 4).mean() * 100
    _LOG.info(
        "walk_forward_predict: %d rows | %d dates | %.1f%% class-4 signals",
        len(result),
        result["date"].nunique(),
        class4_pct,
    )
    return result


def build_portfolio_weights(
    predictions: pd.DataFrame,
    feature_frame: pd.DataFrame,
    prices: pd.DataFrame,
    filter_set: FundamentalFilter,
    weighter: Weighter,
    config: RunConfig,
) -> pd.DataFrame:
    if predictions.empty:
        empty = pd.DataFrame(columns=["date", "symbol", "weight"])
        empty.attrs["rebalance_dates"] = []
        return empty

    # Keep feature_frame in the signature because filters must evaluate the exact
    # point-in-time snapshot that produced the predictions for each rebalance date.
    feature_pdf = feature_frame.reset_index()
    price_dates = sorted(prices.index.get_level_values("date").unique().tolist())
    next_date_map = {
        current: price_dates[index + 1]
        for index, current in enumerate(price_dates[:-1])
    }
    execution_dates: list[pd.Timestamp] = []
    rows: list[dict[str, object]] = []
    n_dates_total = predictions["date"].nunique()
    n_dates_no_feature = n_dates_filter_zero = n_dates_class4_zero = 0
    _LOG.info("build_portfolio_weights: %d signal dates to process", n_dates_total)

    for signal_date, group in predictions.groupby("date", sort=True):
        signal_timestamp = pd.Timestamp(signal_date)
        execution_date = next_date_map.get(signal_timestamp)
        if execution_date is None:
            continue
        execution_dates.append(pd.Timestamp(execution_date))
        df_at_t = feature_pdf.loc[feature_pdf["date"] == signal_timestamp].copy()
        if df_at_t.empty:
            n_dates_no_feature += 1
            _LOG.debug("  %s — no feature rows at this date", signal_date)
            continue
        eligible = apply_filter(df_at_t, filter_set)
        eligible_symbols = set(eligible["symbol"].tolist())
        if not eligible_symbols:
            n_dates_filter_zero += 1
            _LOG.debug("  %s — 0/%d symbols passed filter", signal_date, len(df_at_t))
            continue
        signals = group.loc[group["symbol"].isin(eligible_symbols)]
        signals = signals.loc[signals["pred_class"] == config.signal_class]
        if signals.empty:
            n_dates_class4_zero += 1
            _LOG.debug(
                "  %s — %d eligible, %d in signal group, 0 class-%d",
                signal_date,
                len(eligible_symbols),
                len(group.loc[group["symbol"].isin(eligible_symbols)]),
                config.signal_class,
            )
            continue
        selected = signals.nlargest(config.max_positions, "prob_strong_bull")
        selected_symbols = selected["symbol"].tolist()
        _LOG.debug(
            "  %s — eligible=%d class4=%d selected=%s",
            signal_date,
            len(eligible_symbols),
            len(signals),
            selected_symbols,
        )
        weights = weighter.compute(
            selected_symbols,
            config.max_positions,
            prices,
            selected.set_index("symbol")["prob_strong_bull"],
        )
        for symbol, weight in weights.items():
            rows.append({"date": execution_date, "symbol": symbol, "weight": float(weight)})

    _LOG.info(
        "build_portfolio_weights summary: total_dates=%d | no_feature=%d | filter_zero=%d | class4_zero=%d | with_positions=%d",
        n_dates_total,
        n_dates_no_feature,
        n_dates_filter_zero,
        n_dates_class4_zero,
        len(execution_dates) - n_dates_no_feature - n_dates_filter_zero - n_dates_class4_zero,
    )
    if not rows:
        _LOG.warning(
            "build_portfolio_weights: NO weights produced — "
            "filter_zero=%d, class4_zero=%d. "
            "Check filter thresholds and model class distribution.",
            n_dates_filter_zero,
            n_dates_class4_zero,
        )
    result = pd.DataFrame(rows, columns=["date", "symbol", "weight"])
    result.attrs["rebalance_dates"] = sorted(dict.fromkeys(execution_dates))
    return result.sort_values(["date", "symbol"]).reset_index(drop=True)


class PrebuiltSignalsStrategy(BaseStrategy):
    def __init__(self, signals: pl.DataFrame) -> None:
        self._signals = self.validate_signal_frame(signals)

    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        return self._signals


def dense_signal_frame(weights: pd.DataFrame, prices: pd.DataFrame) -> pl.DataFrame:
    rebalance_dates = [pd.Timestamp(value) for value in weights.attrs.get("rebalance_dates", [])]
    symbols = sorted(prices.index.get_level_values("symbol").unique().tolist())
    if not rebalance_dates or not symbols:
        return BaseStrategy.empty_signal_frame()

    grid = pd.MultiIndex.from_product([rebalance_dates, symbols], names=["date", "symbol"]).to_frame(index=False)
    merged = grid.merge(weights, on=["date", "symbol"], how="left")
    merged["weight"] = pd.to_numeric(merged["weight"], errors="coerce").fillna(0.0)
    merged["signal"] = np.where(merged["weight"] > 0.0, 1, 0)
    return pl.from_pandas(merged[["date", "symbol", "signal", "weight"]]).with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("symbol").cast(pl.String),
        pl.col("signal").cast(pl.Int32),
        pl.col("weight").cast(pl.Float64),
    )


def run_backtest(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    config: RunConfig,
    evaluation_start_date: date,
    evaluation_end_date: date,
) -> BacktestResult:
    signals = dense_signal_frame(weights, prices)
    strategy = PrebuiltSignalsStrategy(signals)
    price_frame = pl.from_pandas(prices.reset_index()).with_columns(pl.col("date").cast(pl.Date))
    engine = VectorBTProEngine()
    qts_config = BacktestConfig(
        workflow="research",
        asset_types=["vn_stock"],
        universe=QTSUniverseConfig(vn_stock=sorted(prices.index.get_level_values("symbol").unique().tolist())),
        start_date=prices.index.get_level_values("date").min().date(),
        end_date=prices.index.get_level_values("date").max().date(),
        test_start_date=evaluation_start_date,
        initial_capital=Decimal(str(config.initial_capital)),
        backtest_engine="vectorbt",
        rebalance_frequency=config.rebalance_period,
        commission=CommissionConfig(model="percentage", rate=Decimal(str(config.commission))),
        calendar="XHOSE",
    )
    qts_result = engine.run(strategy, price_frame, qts_config)
    returns_pdf = qts_result.returns.to_pandas()
    returns_pdf["date"] = pd.to_datetime(returns_pdf["date"])
    eval_returns = returns_pdf.loc[
        (returns_pdf["date"].dt.date >= evaluation_start_date)
        & (returns_pdf["date"].dt.date <= evaluation_end_date),
        "portfolio_return",
    ].astype(float)
    if eval_returns.empty:
        equity_curve = pd.Series(dtype=float, name="equity")
        metrics = {"cagr": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}
    else:
        equity_values = config.initial_capital * (1.0 + eval_returns).cumprod()
        equity_curve = pd.Series(
            equity_values.to_numpy(dtype=float),
            index=returns_pdf.loc[eval_returns.index, "date"],
            name="equity",
        )
        metrics = build_metrics(eval_returns.tolist(), [config.initial_capital, *equity_curve.tolist()])
    cagr = float(metrics.get("cagr", 0.0))
    max_drawdown = float(metrics.get("max_drawdown", 0.0))
    sharpe = float(metrics.get("sharpe", 0.0))
    calmar = 0.0 if math.isclose(max_drawdown, 0.0) else cagr / max_drawdown

    rebalance_dates = pd.Index(pd.to_datetime(weights.attrs.get("rebalance_dates", [])), name="date")
    if rebalance_dates.empty:
        avg_positions = 0.0
        avg_cash_pct = 1.0
        n_rebalances = 0
    else:
        position_counts = weights.groupby("date")["symbol"].nunique()
        invested = weights.groupby("date")["weight"].sum()
        avg_positions = float(position_counts.reindex(rebalance_dates, fill_value=0).mean())
        avg_cash_pct = float(1.0 - invested.reindex(rebalance_dates, fill_value=0.0).mean())
        n_rebalances = len(rebalance_dates)

    return BacktestResult(
        cagr=cagr,
        max_drawdown=max_drawdown,
        sharpe=sharpe,
        calmar=calmar,
        equity_curve=equity_curve,
        weights=weights.copy(),
        n_rebalances=n_rebalances,
        avg_positions=avg_positions,
        avg_cash_pct=avg_cash_pct,
        qts_result=qts_result,
    )


def cagr_with_drawdown_penalty(max_dd_limit: float = 0.30) -> ObjectiveFn:
    def _objective(result: BacktestResult) -> float:
        if result.max_drawdown > max_dd_limit:
            return float("-inf")
        return result.cagr

    return _objective


def run_optuna(
    run_fn: Callable[[dict[str, object]], BacktestResult],
    search_space: dict[str, list[object]],
    objective: ObjectiveFn,
    n_trials: int,
    seed: int,
) -> OptunaResult:
    if optuna is None:
        raise ImportError("optuna is required. Install with `.venv/bin/pip install optuna`.")

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def _objective(trial: object) -> float:
        params = {
            name: trial.suggest_categorical(name, values)
            for name, values in search_space.items()
        }
        result = run_fn(params)
        trial.set_user_attr("cagr", result.cagr)
        trial.set_user_attr("max_drawdown", result.max_drawdown)
        trial.set_user_attr("avg_positions", result.avg_positions)
        trial.set_user_attr("avg_cash_pct", result.avg_cash_pct)
        return objective(result)

    study.optimize(_objective, n_trials=n_trials)
    best_params = dict(study.best_params)
    best_result = run_fn(best_params)
    return OptunaResult(study=study, best_result=best_result, best_params=best_params)


def benchmark_returns(benchmark_prices: pd.DataFrame, start_date: date, end_date: date) -> pd.Series:
    if benchmark_prices.empty:
        return pd.Series(dtype=float, name="benchmark")
    frame = benchmark_prices.copy().sort_values("date")
    frame = frame.loc[
        (frame["date"].dt.date >= start_date)
        & (frame["date"].dt.date <= end_date)
    ].copy()
    returns = frame["close"].pct_change().fillna(0.0)
    return pd.Series(returns.to_numpy(dtype=float), index=pd.to_datetime(frame["date"], utc=True), name="benchmark")


def save_prepared_data(
    config: RunConfig,
    prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    featured: pd.DataFrame,
    feature_cols: list[str],
) -> Path:
    prepared_dir = resolve_prepared_data_dir(config)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    prices_reset = prices.reset_index().sort_values(["date", "symbol"]).copy()
    featured_reset = featured.reset_index().sort_values(["date", "symbol"]).copy()
    benchmark_reset = benchmark_prices.copy().sort_values("date").reset_index(drop=True)

    train_mask = featured_reset["date"].dt.date <= config.optimize_end_date
    inference_mask = (
        (featured_reset["date"].dt.date >= config.inference_start_date)
        & (featured_reset["date"].dt.date <= config.inference_end_date)
    )

    prices_reset.to_parquet(prepared_dir / "prices.parquet", index=False)
    benchmark_reset.to_parquet(prepared_dir / "benchmark_prices.parquet", index=False)
    featured_reset.to_parquet(prepared_dir / "featured_all.parquet", index=False)
    featured_reset.loc[train_mask].to_parquet(prepared_dir / "featured_train.parquet", index=False)
    featured_reset.loc[inference_mask].to_parquet(prepared_dir / "featured_inference.parquet", index=False)

    metadata = {
        "schema_version": PREPARED_DATA_SCHEMA_VERSION,
        "universe": config.universe.name,
        "download_start_date": config.download_start_date.isoformat(),
        "optimize_end_date": config.optimize_end_date.isoformat(),
        "inference_start_date": config.inference_start_date.isoformat(),
        "inference_end_date": config.inference_end_date.isoformat(),
        "price_interval": config.price_interval,
        "feature_cols": feature_cols,
        "n_price_rows": len(prices_reset),
        "n_feature_rows": len(featured_reset),
        "n_train_rows": int(train_mask.sum()),
        "n_inference_rows": int(inference_mask.sum()),
    }
    (prepared_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _LOG.info(
        "Saved prepared data: %s | prices=%d | featured=%d | train=%d | inference=%d",
        prepared_dir,
        len(prices_reset),
        len(featured_reset),
        int(train_mask.sum()),
        int(inference_mask.sum()),
    )
    return prepared_dir


def load_prepared_data(
    config: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    prepared_dir = resolve_prepared_data_dir(config)
    required_paths = {
        "metadata": prepared_dir / "metadata.json",
        "prices": prepared_dir / "prices.parquet",
        "benchmark": prepared_dir / "benchmark_prices.parquet",
        "featured": prepared_dir / "featured_all.parquet",
    }
    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Prepared data is incomplete at {prepared_dir}. Missing: {', '.join(sorted(missing))}"
        )

    metadata = json.loads(required_paths["metadata"].read_text(encoding="utf-8"))
    if int(metadata.get("schema_version", -1)) != PREPARED_DATA_SCHEMA_VERSION:
        raise ValueError(
            f"Prepared data schema mismatch at {prepared_dir}: "
            f"expected {PREPARED_DATA_SCHEMA_VERSION}, got {metadata.get('schema_version')}"
        )

    prices = pd.read_parquet(required_paths["prices"])
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.set_index(["date", "symbol"]).sort_index()

    benchmark_prices = pd.read_parquet(required_paths["benchmark"])
    if not benchmark_prices.empty and "date" in benchmark_prices.columns:
        benchmark_prices["date"] = pd.to_datetime(benchmark_prices["date"])
        benchmark_prices = benchmark_prices.sort_values("date").reset_index(drop=True)

    featured = pd.read_parquet(required_paths["featured"])
    featured["date"] = pd.to_datetime(featured["date"])
    featured = featured.set_index(["date", "symbol"]).sort_index()

    feature_cols = [str(column) for column in metadata.get("feature_cols", [])]
    _LOG.info(
        "Loaded prepared data: %s | prices=%d | featured=%d | feature_cols=%d",
        prepared_dir,
        len(prices),
        len(featured),
        len(feature_cols),
    )
    return prices, benchmark_prices, featured, feature_cols


def prepared_labeled_cache_path(config: RunConfig, target_horizon: int) -> Path:
    return resolve_prepared_data_dir(config) / f"labeled_h{target_horizon}.parquet"


def save_labeled_frame(config: RunConfig, labeled: pd.DataFrame, target_horizon: int) -> Path:
    path = prepared_labeled_cache_path(config, target_horizon)
    path.parent.mkdir(parents=True, exist_ok=True)
    labeled.reset_index().sort_values(["date", "symbol"]).to_parquet(path, index=False)
    _LOG.info("Saved labeled cache for horizon=%d: %s (%d rows)", target_horizon, path, len(labeled))
    return path


def load_labeled_frame(config: RunConfig, target_horizon: int) -> pd.DataFrame:
    path = prepared_labeled_cache_path(config, target_horizon)
    if not path.exists():
        raise FileNotFoundError(f"Missing labeled cache for horizon={target_horizon}: {path}")
    labeled = pd.read_parquet(path)
    labeled["date"] = pd.to_datetime(labeled["date"])
    return labeled.set_index(["date", "symbol"]).sort_index()


def load_or_build_labeled_frame(
    config: RunConfig,
    featured: pd.DataFrame,
    target_horizon: int,
    feature_cols: list[str],
    labeler: LabelScheme,
    *,
    allow_reuse: bool,
) -> pd.DataFrame:
    if allow_reuse:
        path = prepared_labeled_cache_path(config, target_horizon)
        if path.exists():
            labeled = load_labeled_frame(config, target_horizon)
            _LOG.info("Reused labeled cache for horizon=%d: %d rows", target_horizon, len(labeled))
            return labeled
    labeled = make_training_frame(featured, target_horizon, feature_cols, labeler)
    save_labeled_frame(config, labeled, target_horizon)
    return labeled


def save_outputs(
    optuna_result: OptunaResult,
    final_result: BacktestResult,
    benchmark_prices: pd.DataFrame,
    config: RunConfig,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(optuna_result.study.trials_dataframe()).to_csv(config.output_dir / "trials.csv", index=False)
    final_result.weights.to_csv(config.output_dir / "weights.csv", index=False)

    metrics = pd.DataFrame(
        [
            {
                "cagr": final_result.cagr,
                "max_drawdown": final_result.max_drawdown,
                "sharpe": final_result.sharpe,
                "calmar": final_result.calmar,
                "avg_positions": final_result.avg_positions,
                "avg_cash_pct": final_result.avg_cash_pct,
                "n_rebalances": final_result.n_rebalances,
            }
        ]
    )
    metrics.to_csv(config.output_dir / "metrics.csv", index=False)
    (config.output_dir / "best_params.json").write_text(
        json.dumps(optuna_result.best_params, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    _plot_equity_curve(final_result.equity_curve, config.output_dir / "equity_curve.png")
    _plot_drawdown(final_result.equity_curve, config.output_dir / "drawdown.png")
    benchmark_rets = benchmark_returns(
        benchmark_prices,
        start_date=config.inference_start_date,
        end_date=config.inference_end_date,
    )
    save_pyfolio_tearsheet(
        final_result.qts_result,
        benchmark_rets=benchmark_rets if not benchmark_rets.empty else None,
        live_start_date=config.inference_start_date,
        path=config.output_dir / "pyfolio_tearsheet.pdf",
    )
    tearsheet_path = save_tearsheet(
        final_result.qts_result,
        config.output_dir,
        run_id=config.output_dir.name,
        benchmark_rets=benchmark_rets if not benchmark_rets.empty else None,
    )
    if tearsheet_path is not None:
        (config.output_dir / "tearsheet_path.txt").write_text(str(tearsheet_path), encoding="utf-8")


def save_pyfolio_tearsheet(
    result: QTSBacktestResult,
    benchmark_rets: pd.Series | None,
    live_start_date: date,
    path: Path,
) -> None:
    import pyfolio as pf

    returns = returns_series(result)
    positions = positions_frame(result)
    transactions = transactions_frame(result)

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.close("all")
    try:
        pf.create_full_tear_sheet(
            returns=returns,
            positions=positions,
            transactions=transactions,
            benchmark_rets=benchmark_rets,
            live_start_date=pd.Timestamp(live_start_date).tz_localize("UTC"),
            set_context=False,
        )
    except Exception:
        plt.close("all")
        pf.create_returns_tear_sheet(
            returns=returns,
            benchmark_rets=benchmark_rets,
            live_start_date=pd.Timestamp(live_start_date).tz_localize("UTC"),
            set_context=False,
        )
    with PdfPages(path) as pdf:
        for fig_num in plt.get_fignums():
            pdf.savefig(plt.figure(fig_num), bbox_inches="tight")
    plt.close("all")


def _plot_equity_curve(series: pd.Series, path: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(series.index, series.values, linewidth=1.4)
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_drawdown(series: pd.Series, path: Path) -> None:
    if series.empty:
        plt.figure(figsize=(10, 4))
        plt.title("Drawdown")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return
    running_max = series.cummax()
    drawdown = (series / running_max) - 1.0
    plt.figure(figsize=(10, 4))
    plt.fill_between(drawdown.index, drawdown.values, 0.0, alpha=0.45)
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main(
    config: RunConfig,
    predictor_factory: Callable[[dict[str, object]], Predictor],
    labeler: LabelScheme,
    builders: list[FeatureBuilder],
    filter_set: FundamentalFilter,
    weighter: Weighter,
    annual_transformers: list[FundamentalTransformer],
    quarterly_transformers: list[FundamentalTransformer],
    objective: ObjectiveFn,
    search_space: dict[str, list[object]],
) -> tuple[OptunaResult, BacktestResult]:
    debug_dir = config.output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    prepared_data_reused = config.reuse_prepared_data

    if prepared_data_reused:
        _LOG.info("Prepared data reuse enabled — loading cached frames")
        prices, benchmark_price_frame, featured, feature_cols = load_prepared_data(config)
        _LOG.info(
            "prepared prices: %d rows | %d symbols | dates %s → %s",
            len(prices),
            prices.index.get_level_values("symbol").nunique(),
            prices.index.get_level_values("date").min().date(),
            prices.index.get_level_values("date").max().date(),
        )
        _LOG.info(
            "prepared featured: %d rows | %d feature cols",
            len(featured),
            len(feature_cols),
        )
    else:
        # ── Step 1: market data ──────────────────────────────────────────────
        _LOG.info("Step 1 — downloading market data")
        prices, benchmark_price_frame = asyncio.run(download_market_data(config))
        _LOG.info(
            "prices: %d rows | %d symbols | dates %s → %s",
            len(prices),
            prices.index.get_level_values("symbol").nunique(),
            prices.index.get_level_values("date").min().date(),
            prices.index.get_level_values("date").max().date(),
        )

        # ── Step 2: fundamentals ─────────────────────────────────────────────
        _LOG.info("Step 2 — loading fundamentals")
        annual_long = load_fundamentals(config.universe, "annual")
        quarterly_long = load_fundamentals(config.universe, "quarterly")
        _LOG.info(
            "annual_long: %d rows | %d symbols | unique item_en (first 10): %s",
            len(annual_long),
            annual_long["symbol"].nunique(),
            sorted(annual_long["item_en"].unique().tolist())[:10],
        )
        _LOG.info(
            "quarterly_long: %d rows | %d symbols | unique item_en (first 10): %s",
            len(quarterly_long),
            quarterly_long["symbol"].nunique(),
            sorted(quarterly_long["item_en"].unique().tolist())[:10],
        )
        _save_debug(
            annual_long[["report_type", "item_en"]].drop_duplicates().sort_values(["report_type", "item_en"]),
            debug_dir / "annual_item_en_list.csv",
            "annual_item_en_list",
        )
        _save_debug(
            quarterly_long[["report_type", "item_en"]].drop_duplicates().sort_values(["report_type", "item_en"]),
            debug_dir / "quarterly_item_en_list.csv",
            "quarterly_item_en_list",
        )

        # ── Step 3: pivot + transform fundamentals ───────────────────────────
        _LOG.info("Step 3 — pivoting and transforming fundamentals")
        annual_wide = pivot_fundamentals(annual_long, "annual")
        quarterly_wide = pivot_fundamentals(quarterly_long, "quarterly")
        _LOG.info(
            "annual_wide: %d rows | %d metric cols | cols=%s",
            len(annual_wide),
            len([c for c in annual_wide.columns if c not in FUNDAMENTAL_AUX_COLUMNS]),
            [c for c in annual_wide.columns if c not in FUNDAMENTAL_AUX_COLUMNS][:15],
        )
        _LOG.info(
            "quarterly_wide: %d rows | %d metric cols | cols=%s",
            len(quarterly_wide),
            len([c for c in quarterly_wide.columns if c not in FUNDAMENTAL_AUX_COLUMNS]),
            [c for c in quarterly_wide.columns if c not in FUNDAMENTAL_AUX_COLUMNS][:15],
        )
        if annual_wide.empty:
            _LOG.error(
                "annual_wide is EMPTY — CANONICAL_ITEM_LOOKUP produced no matches for annual data. "
                "Check that item_en values in annual_item_en_list.csv appear in the notebook canonical mapping."
            )
        if quarterly_wide.empty:
            _LOG.error(
                "quarterly_wide is EMPTY — CANONICAL_ITEM_LOOKUP produced no matches for quarterly data. "
                "Check that item_en values in quarterly_item_en_list.csv appear in the notebook canonical mapping."
            )
        _save_debug(annual_wide.head(200), debug_dir / "annual_wide_sample.csv", "annual_wide_sample")
        _save_debug(quarterly_wide.head(200), debug_dir / "quarterly_wide_sample.csv", "quarterly_wide_sample")

        annual_features, quarterly_features = build_fundamental_features(
            annual_wide,
            quarterly_wide,
            annual_transformers,
            quarterly_transformers,
        )
        _LOG.info("annual_features: %d rows | cols=%s", len(annual_features), list(annual_features.columns))
        _LOG.info("quarterly_features: %d rows | cols=%s", len(quarterly_features), list(quarterly_features.columns))

        if not annual_features.empty and "available_from" in annual_features.columns:
            earliest_annual = pd.Timestamp(annual_features["available_from"].min())
            first_possible_pred = earliest_annual + pd.tseries.offsets.BDay(
                config.train_window + config.target_horizon
            )
            coverage_td = int(
                (pd.Timestamp(config.optimize_end_date) - earliest_annual).days * 252 / 365
            )
            _LOG.info(
                "Coverage preflight: earliest_annual=%s  first_possible_pred≈%s  "
                "optimize_end=%s  coverage≈%d trading days",
                earliest_annual.date(),
                first_possible_pred.date(),
                config.optimize_end_date,
                coverage_td,
            )
            if first_possible_pred.date() > config.optimize_end_date:
                raise RuntimeError(
                    f"Fundamental coverage preflight FAILED.\n"
                    f"  Earliest annual available_from : {earliest_annual.date()}\n"
                    f"  train_window + target_horizon  : {config.train_window} + {config.target_horizon} trading days\n"
                    f"  First possible prediction      : {first_possible_pred.date()}\n"
                    f"  optimize_end_date              : {config.optimize_end_date}  ← must be >= first_possible_pred\n"
                    f"  Fix A: re-download with more history:\n"
                    f"    --force-refresh-fundamentals --fundamental-pages 25\n"
                    f"  Fix B: push optimize_end_date forward to >= {first_possible_pred.date()}\n"
                    f"  Fix C: reduce train_window (currently {config.train_window})"
                )
        else:
            _LOG.warning("Coverage preflight skipped — annual_features is empty or missing available_from column.")

        # ── Step 4: asof merge ───────────────────────────────────────────────
        _LOG.info("Step 4 — merging fundamentals asof into prices")
        merged = merge_fundamentals_asof(prices, annual_features, quarterly_features)
        _LOG.info("merged: %d rows | %d cols (prices had %d rows)", len(merged), len(merged.columns), len(prices))
        if len(merged) < len(prices) * 0.5:
            _LOG.warning(
                "merged has <50%% of price rows (%d vs %d) — "
                "many rows dropped because no fundamental data available yet (release lag). "
                "This is expected for earliest dates; check if it's too severe.",
                len(merged),
                len(prices),
            )

        # ── Step 5: feature building ─────────────────────────────────────────
        _LOG.info("Step 5 — building features")
        featured = build_features(merged, builders)
        all_feature_cols = [column for column in featured.columns if column not in merged.columns]

        _null_threshold = 0.99
        feature_null_stats: list[dict[str, object]] = []
        feature_cols = []
        for col in all_feature_cols:
            null_pct = float(featured[col].isna().mean())
            kept = null_pct < _null_threshold
            feature_null_stats.append({"feature": col, "null_pct": round(null_pct, 4), "included": kept})
            if kept:
                feature_cols.append(col)
            else:
                _LOG.warning(
                    "DROPPING feature '%s' — %.1f%% null (threshold %.0f%%). "
                    "Likely a missing fundamental column — check the VCI/KBS canonical mapping.",
                    col, null_pct * 100, _null_threshold * 100,
                )

        null_df = pd.DataFrame(feature_null_stats).sort_values("null_pct", ascending=False)
        _save_debug(null_df, debug_dir / "feature_null_rates.csv", "feature_null_rates")
        _LOG.info(
            "feature_cols: %d / %d retained (dropped %d all-NaN features)",
            len(feature_cols), len(all_feature_cols), len(all_feature_cols) - len(feature_cols),
        )
        if not feature_cols:
            raise RuntimeError(
                "All feature columns are null — check fundamental data loading and the VCI/KBS canonical mapping. "
                f"See {debug_dir / 'annual_item_en_list.csv'} for actual item_en names in parquet files."
            )

        save_prepared_data(config, prices, benchmark_price_frame, featured, feature_cols)

    # ── Step 6: label frame ──────────────────────────────────────────────────
    _LOG.info("Step 6 — making training frame (horizon=%d)", config.target_horizon)
    labeled = load_or_build_labeled_frame(
        config,
        featured,
        config.target_horizon,
        feature_cols,
        labeler,
        allow_reuse=prepared_data_reused,
    )
    _LOG.info(
        "labeled: %d rows | %d symbols",
        len(labeled),
        labeled.index.get_level_values("symbol").nunique() if not labeled.empty else 0,
    )
    _save_debug(labeled.reset_index().head(5000), debug_dir / "labeled_sample.csv", "labeled_sample")
    if labeled.empty:
        raise RuntimeError(
            "Labeled DataFrame is empty after make_training_frame — cannot proceed. "
            f"Review {debug_dir / 'feature_null_rates.csv'} and {debug_dir / 'annual_item_en_list.csv'}."
        )

    def run_optimization(params: dict[str, object]) -> BacktestResult:
        tuned = replace(
            config,
            train_window=int(params.get("train_window", config.train_window)),
            target_horizon=int(params.get("target_horizon", config.target_horizon)),
            max_positions=int(params.get("max_positions", config.max_positions)),
            signal_class=int(params.get("signal_class", config.signal_class)),
        )
        predictor = predictor_factory({**params, "seed": tuned.seed})
        tuned_labeled = load_or_build_labeled_frame(
            tuned,
            featured,
            tuned.target_horizon,
            feature_cols,
            labeler,
            allow_reuse=prepared_data_reused,
        )
        predictions = walk_forward_predict(
            tuned_labeled,
            predictor,
            tuned,
            feature_cols,
            prediction_start_date=tuned.download_start_date,
            prediction_end_date=tuned.optimize_end_date,
        )
        weights = build_portfolio_weights(predictions, featured, prices, filter_set, weighter, tuned)
        return run_backtest(
            weights,
            prices,
            tuned,
            evaluation_start_date=tuned.download_start_date,
            evaluation_end_date=tuned.optimize_end_date,
        )

    # ── Step 7: Optuna ───────────────────────────────────────────────────────
    _LOG.info("Step 7 — Optuna (n_trials=%d)", config.n_trials)
    optuna_result = run_optuna(
        run_fn=run_optimization,
        search_space=search_space,
        objective=objective,
        n_trials=config.n_trials,
        seed=config.seed,
    )
    _LOG.info("Best params: %s | best value: %.4f", optuna_result.best_params, optuna_result.study.best_value)

    # ── Step 8: inference with best params ───────────────────────────────────
    _LOG.info("Step 8 — inference [%s → %s]", config.inference_start_date, config.inference_end_date)
    best_cfg = replace(
        config,
        train_window=int(optuna_result.best_params.get("train_window", config.train_window)),
        target_horizon=int(optuna_result.best_params.get("target_horizon", config.target_horizon)),
        max_positions=int(optuna_result.best_params.get("max_positions", config.max_positions)),
        signal_class=int(optuna_result.best_params.get("signal_class", config.signal_class)),
    )
    best_predictor = predictor_factory({**optuna_result.best_params, "seed": best_cfg.seed})
    best_labeled = load_or_build_labeled_frame(
        best_cfg,
        featured,
        best_cfg.target_horizon,
        feature_cols,
        labeler,
        allow_reuse=prepared_data_reused,
    )
    inference_predictions = walk_forward_predict(
        best_labeled,
        best_predictor,
        best_cfg,
        feature_cols,
        prediction_start_date=best_cfg.inference_start_date,
        prediction_end_date=best_cfg.inference_end_date,
    )
    _save_debug(inference_predictions, debug_dir / "signals.csv", "inference_signals")
    _LOG.info(
        "inference_predictions: %d rows | %d dates | class_dist=%s",
        len(inference_predictions),
        inference_predictions["date"].nunique() if not inference_predictions.empty else 0,
        inference_predictions["pred_class"].value_counts().sort_index().to_dict() if not inference_predictions.empty else "empty",
    )

    final_weights = build_portfolio_weights(
        inference_predictions,
        featured,
        prices,
        filter_set,
        weighter,
        best_cfg,
    )
    _save_debug(final_weights, debug_dir / "portfolio_weights.csv", "portfolio_weights")
    _LOG.info(
        "final_weights: %d rows | %d dates | avg_weight=%.3f",
        len(final_weights),
        final_weights["date"].nunique() if not final_weights.empty else 0,
        final_weights["weight"].mean() if not final_weights.empty else 0.0,
    )

    # ── Step 9: backtest ─────────────────────────────────────────────────────
    _LOG.info("Step 9 — running backtest")
    final_result = run_backtest(
        final_weights,
        prices,
        best_cfg,
        evaluation_start_date=best_cfg.inference_start_date,
        evaluation_end_date=best_cfg.inference_end_date,
    )
    _LOG.info(
        "Backtest: CAGR=%.2f%% | MaxDD=%.2f%% | Sharpe=%.2f | avg_positions=%.1f | avg_cash=%.1f%%",
        final_result.cagr * 100,
        final_result.max_drawdown * 100,
        final_result.sharpe,
        final_result.avg_positions,
        final_result.avg_cash_pct * 100,
    )
    save_outputs(optuna_result, final_result, benchmark_price_frame, best_cfg)
    return optuna_result, final_result


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    default_universe = default_universe_config(repo_root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols-path", type=Path, default=default_universe.symbols_path)
    parser.add_argument("--runtime-root", type=Path, default=default_universe.runtime_root)
    parser.add_argument(
        "--prepared-data-dir",
        type=Path,
        default=None,
        help="Optional base directory for saved engineered train/inference inputs and labeled caches.",
    )
    parser.add_argument(
        "--reuse-prepared-data",
        action="store_true",
        help="Skip download/fundamental/feature engineering and reload prepared artifacts from --prepared-data-dir.",
    )
    # VCI annual and quarterly fundamentals cover the universe back to 2018.
    parser.add_argument("--download-start-date", type=str, default="2018-01-01")
    parser.add_argument("--optimize-end-date", type=str, default="2023-09-30")
    parser.add_argument("--inference-start-date", type=str, default="2019-06-01")
    parser.add_argument("--inference-end-date", type=str, default="2026-06-01")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=10)
    # 25 pages × 4 periods/page = 100 periods max per report type
    parser.add_argument("--fundamental-pages", type=int, default=40)
    parser.add_argument("--force-refresh-fundamentals", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / ".qts_notebook_runtime" / "end2end_stock_factor_runs",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    parser = build_parser(repo_root)
    args = parser.parse_args()

    # Bootstrap console logging before we know output_dir
    setup_logging()

    universe = UniverseConfig(
        name="vn100",
        symbols_path=args.symbols_path,
        runtime_root=args.runtime_root,
    )
    config = RunConfig(
        universe=universe,
        download_start_date=date.fromisoformat(args.download_start_date),
        optimize_end_date=date.fromisoformat(args.optimize_end_date),
        inference_start_date=date.fromisoformat(args.inference_start_date),
        inference_end_date=date.fromisoformat(args.inference_end_date),
        target_horizon=14,
        train_window=100,
        rebalance_period=14,
        max_positions=5,
        signal_class=4,
        commission=0.0015,
        initial_capital=100_000_000,
        max_drawdown_limit=0.45,
        n_trials=args.n_trials,
        seed=args.seed,
        price_interval="1d",
        batch_size=args.batch_size,
        fundamental_pages=args.fundamental_pages,
        force_refresh_fundamentals=args.force_refresh_fundamentals,
        output_dir=args.output_root / timestamp(),
        prepared_data_dir=args.prepared_data_dir,
        reuse_prepared_data=args.reuse_prepared_data,
    )
    # Add file handler now that output_dir is resolved
    setup_logging(config.output_dir)
    _LOG.info("Run output dir: %s", config.output_dir)

    optuna_result, final_result = main(
        config=config,
        predictor_factory=lambda params: XGBPredictor(
            max_depth=int(params.get("max_depth", 3)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            n_estimators=100,
            seed=int(params.get("seed", config.seed)),
        ),
        labeler=CrossSectionalQuantileLabeler(n_classes=5),
        builders=[
            MomentumFeatures(),
            VolatilityFeatures(),
            TrendFeatures(),
            LiquidityFeatures(),
            FundamentalFeatures(),
        ],
        filter_set=CompositeFilter(
            [
                ProfitabilityFilter(min_roe=5.0),
                QualityFilter(max_debt_ratio=70.0),
            ]
        ),
        weighter=FixedSlotEqualWeighter(),
        annual_transformers=[
            YoYAnnualTransformer(metrics=["revenue", "netIncome", "profitBeforeTax", "totalAssets"]),
        ],
        quarterly_transformers=[
            TTMTransformer(metrics=["revenue", "netIncome", "profitBeforeTax"]),
            QuarterlyQoQTransformer(metrics=["revenue", "netIncome", "profitBeforeTax"]),
            SameQuarterYoYTransformer(metrics=["revenue", "netIncome", "profitBeforeTax"]),
        ],
        objective=cagr_with_drawdown_penalty(max_dd_limit=0.45),
        search_space=SEARCH_SPACE,
    )

    print(json.dumps(optuna_result.best_params, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "cagr": final_result.cagr,
                "max_drawdown": final_result.max_drawdown,
                "sharpe": final_result.sharpe,
                "calmar": final_result.calmar,
                "avg_positions": final_result.avg_positions,
                "avg_cash_pct": final_result.avg_cash_pct,
            },
            indent=2,
            sort_keys=True,
        )
    )
