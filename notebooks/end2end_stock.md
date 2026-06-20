# Imports
```python
from __future__ import annotations

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
from typing import Callable, Literal

import matplotlib
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.backends.backend_pdf import PdfPages

try:
    import optuna
except ImportError:
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
```

# Notebook Paths
```python
repo_root = Path.cwd().resolve()
while not (repo_root / "configs" / "assets" / "vn100.yml").exists() and repo_root != repo_root.parent:
    repo_root = repo_root.parent

if not (repo_root / "configs" / "assets" / "vn100.yml").exists():
    raise FileNotFoundError("Could not find configs/assets/vn100.yml from the current working directory.")

repo_root
```

# Constants
```python
FUNDAMENTAL_AUX_COLUMNS = {
    "symbol",
    "fiscal_year",
    "quarter",
    "period_end",
    "report_date",
    "available_from",
}
MIN_FUNDAMENTAL_YEAR = 2018
PREPARED_DATA_SCHEMA_VERSION = 1
SEARCH_SPACE = {
    "train_window": [60, 100],
    "target_horizon": [15],
    "max_positions": [5],
    "max_depth": [8],
    "learning_rate": [0.03, 0.01, 0.1],
    "signal_class": [4],
}
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

canonical_item_lookup = {
    (report_type, item_name): column
    for report_type, feature_map in KBS_FUNDAMENTAL_ITEMS.items()
    for column, item_names in feature_map.items()
    for item_name in item_names
}

for report_type, feature_map in VCI_STATEMENT_ITEMS.items():
    for column, item_names in feature_map.items():
        for item_name in item_names:
            canonical_item_lookup[(report_type, item_name)] = column

for source_name, target_name in FMP_ALIASES.items():
    canonical_item_lookup[("CSTC", source_name)] = target_name
    canonical_item_lookup[("CSTC", source_name.lower())] = target_name
    canonical_item_lookup[("CSTC", target_name)] = target_name

canonical_item_lookup.update(
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

list(canonical_item_lookup.items())[:10]
```

# Core Types
```python
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
    best_result: BacktestResult | None
    best_params: dict[str, object]
```

# Utility Helpers
```python
_LOG = logging.getLogger("end2end_factor")


def setup_logging(output_dir: Path | None = None) -> None:
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


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def default_universe_config(repo_root: Path) -> UniverseConfig:
    return UniverseConfig(
        name="vn100",
        symbols_path=repo_root / "configs" / "assets" / "vn100.yml",
        runtime_root=repo_root / ".qts_notebook_runtime" / "live_vn100",
    )


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


def _save_debug(df: pd.DataFrame | None, path: Path, label: str) -> None:
    if df is None or df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def _combine_first(df: pd.DataFrame, primary: str, fallback: str | None = None) -> pd.Series:
    series = df[primary] if primary in df.columns else pd.Series(np.nan, index=df.index)
    if fallback and fallback in df.columns:
        return series.combine_first(df[fallback])
    return series


def _percent_point_threshold_in_series_units(series: pd.Series, threshold_pct_points: float) -> float:
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
```

# Market Data Helpers
```python
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
        import duckdb

        connection = duckdb.connect(str(db_path), read_only=True)
    except Exception:
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
    except Exception:
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
        except Exception as exc:
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
    symbols, benchmark_symbol = load_vn100_symbols(config.universe.symbols_path, max_symbols=None)
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

    if not cached_frames:
        raise ValueError("No OHLCV rows were downloaded.")

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
        await _fetch_missing_fundamentals(
            annual_missing,
            termtype=1,
            pages=config.fundamental_pages,
            force_refresh=config.force_refresh_fundamentals,
        )

    if quarterly_missing:
        await _fetch_missing_fundamentals(
            quarterly_missing,
            termtype=2,
            pages=config.fundamental_pages,
            force_refresh=config.force_refresh_fundamentals,
        )

    ohlcv_pdf = ohlcv_all.filter(pl.col("symbol") != benchmark_symbol).sort(["date", "symbol"]).to_pandas()
    benchmark_pdf = ohlcv_all.filter(pl.col("symbol") == benchmark_symbol).sort("date").to_pandas()

    if ohlcv_pdf.empty:
        raise ValueError("No equity OHLCV rows remain after removing the benchmark.")

    ohlcv_pdf["date"] = pd.to_datetime(ohlcv_pdf["date"])
    if not benchmark_pdf.empty:
        benchmark_pdf["date"] = pd.to_datetime(benchmark_pdf["date"])

    prices = ohlcv_pdf.set_index(["date", "symbol"]).sort_index()
    return prices, benchmark_pdf
```

# Fundamentals Helpers
```python
def _period_end_from_row(freq: Literal["annual", "quarterly"], fiscal_year: int, quarter: float) -> date:
    if freq == "annual":
        return date(int(fiscal_year), 12, 31)
    month = {1: 3, 2: 6, 3: 9, 4: 12}[int(quarter)]
    return date(int(fiscal_year), month, monthrange(int(fiscal_year), month)[1])


def load_fundamentals(universe: UniverseConfig, freq: Literal["annual", "quarterly"]) -> pd.DataFrame:
    symbols, benchmark_symbol = load_vn100_symbols(universe.symbols_path, max_symbols=None)
    label = "annual" if freq == "annual" else "quarterly"
    paths = [
        universe.fundamental_cache_dir / f"{strip_symbol_prefix(symbol)}_{label}.parquet"
        for symbol in symbols
        if symbol != benchmark_symbol
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing {freq} fundamentals cache files: {preview}")

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
```

# Prepared Data Helpers
```python
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
    (prepared_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return prepared_dir


def load_prepared_data(config: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    prepared_dir = resolve_prepared_data_dir(config)
    metadata_path = prepared_dir / "metadata.json"
    prices_path = prepared_dir / "prices.parquet"
    benchmark_path = prepared_dir / "benchmark_prices.parquet"
    featured_path = prepared_dir / "featured_all.parquet"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    prices = pd.read_parquet(prices_path)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.set_index(["date", "symbol"]).sort_index()

    benchmark_prices = pd.read_parquet(benchmark_path)
    if not benchmark_prices.empty and "date" in benchmark_prices.columns:
        benchmark_prices["date"] = pd.to_datetime(benchmark_prices["date"])
        benchmark_prices = benchmark_prices.sort_values("date").reset_index(drop=True)

    featured = pd.read_parquet(featured_path)
    featured["date"] = pd.to_datetime(featured["date"])
    featured = featured.set_index(["date", "symbol"]).sort_index()

    feature_cols = [str(column) for column in metadata.get("feature_cols", [])]
    return prices, benchmark_prices, featured, feature_cols


def prepared_labeled_cache_path(config: RunConfig, target_horizon: int) -> Path:
    return resolve_prepared_data_dir(config) / f"labeled_h{target_horizon}.parquet"


def save_labeled_frame(config: RunConfig, labeled: pd.DataFrame, target_horizon: int) -> Path:
    path = prepared_labeled_cache_path(config, target_horizon)
    path.parent.mkdir(parents=True, exist_ok=True)
    labeled.reset_index().sort_values(["date", "symbol"]).to_parquet(path, index=False)
    return path


def load_labeled_frame(config: RunConfig, target_horizon: int) -> pd.DataFrame:
    path = prepared_labeled_cache_path(config, target_horizon)
    labeled = pd.read_parquet(path)
    labeled["date"] = pd.to_datetime(labeled["date"])
    return labeled.set_index(["date", "symbol"]).sort_index()
```

# Backtest Helpers
```python
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


def cagr_with_drawdown_penalty(max_dd_limit: float = 0.30):
    def _objective(result: BacktestResult) -> float:
        if result.max_drawdown > max_dd_limit:
            return float("-inf")
        return result.cagr

    return _objective


def run_optuna(
    run_fn: Callable[[dict[str, object]], BacktestResult],
    search_space: dict[str, list[object]],
    objective: Callable[[BacktestResult], float],
    n_trials: int,
    seed: int,
) -> OptunaResult:
    if optuna is None:
        raise ImportError("optuna is required. Install with `.venv/bin/pip install optuna`.")

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def _objective(trial: object) -> float:
        params = {name: trial.suggest_categorical(name, values) for name, values in search_space.items()}
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


def save_outputs(
    optuna_result: OptunaResult,
    final_result: BacktestResult,
    benchmark_prices: pd.DataFrame,
    config: RunConfig,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if optuna_result.study is not None and hasattr(optuna_result.study, "trials_dataframe"):
        trials_df = pd.DataFrame(optuna_result.study.trials_dataframe())
    else:
        trials_df = pd.DataFrame(
            [
                {
                    "number": 0,
                    "value": np.nan,
                    "state": "SKIPPED_OPTUNA",
                    **{f"params_{key}": value for key, value in optuna_result.best_params.items()},
                }
            ]
        )
    trials_df.to_csv(config.output_dir / "trials.csv", index=False)
    final_result.weights.to_csv(config.output_dir / "weights.csv", index=False)
    pd.DataFrame(
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
    ).to_csv(config.output_dir / "metrics.csv", index=False)
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
```

# Run Configuration
```python
universe = default_universe_config(repo_root)
prepared_data_dir = default_prepared_data_dir(universe)

config = RunConfig(
    universe=UniverseConfig(
        name=universe.name,
        symbols_path=universe.symbols_path,
        runtime_root=universe.runtime_root,
    ),
    download_start_date=date.fromisoformat("2018-01-01"),
    optimize_end_date=date.fromisoformat("2023-09-30"),
    inference_start_date=date.fromisoformat("2019-06-01"),
    inference_end_date=date.fromisoformat("2026-06-01"),
    target_horizon=14,
    train_window=100,
    rebalance_period=14,
    max_positions=5,
    signal_class=4,
    commission=0.0015,
    initial_capital=100_000_000,
    max_drawdown_limit=0.45,
    n_trials=1,
    seed=42,
    price_interval="1d",
    batch_size=10,
    fundamental_pages=40,
    force_refresh_fundamentals=False,
    output_dir=repo_root / ".qts_notebook_runtime" / "end2end_stock_factor_runs" / timestamp(),
    prepared_data_dir=prepared_data_dir,
    reuse_prepared_data=False,
)

prepared_data_ready = (resolve_prepared_data_dir(config) / "metadata.json").exists()
if prepared_data_ready:
    config = replace(config, reuse_prepared_data=True)

debug_dir = config.output_dir / "debug"
setup_logging(config.output_dir)

config
```

# Search Mode
```python
run_optuna_search = False

fixed_best_params = {
    "learning_rate": 0.1,
    "max_depth": 2,
    "max_positions": 3,
    "signal_class": 4,
    "target_horizon": 21,
    "train_window": 100,
}

pd.DataFrame(
    {
        "value": [
            run_optuna_search,
            fixed_best_params,
        ]
    },
    index=["run_optuna_search", "fixed_best_params"],
)
```

# Universe Symbols
```python
symbols, benchmark_symbol = load_vn100_symbols(config.universe.symbols_path, max_symbols=None)

pd.DataFrame(
    {
        "value": [
            len(symbols),
            benchmark_symbol,
            symbols[:10],
            config.universe.symbols_path,
            config.universe.runtime_root,
        ]
    },
    index=[
        "n_symbols",
        "benchmark_symbol",
        "first_10_symbols",
        "symbols_path",
        "runtime_root",
    ],
)
```

# Load Prices
```python
if config.reuse_prepared_data:
    prices, benchmark_price_frame, featured, feature_cols = load_prepared_data(config)
else:
    prices, benchmark_price_frame = await download_market_data(config)

pd.DataFrame(
    {
        "value": [
            len(prices),
            prices.index.get_level_values("symbol").nunique(),
            prices.index.get_level_values("date").min(),
            prices.index.get_level_values("date").max(),
            len(benchmark_price_frame),
        ]
    },
    index=["price_rows", "price_symbols", "price_start", "price_end", "benchmark_rows"],
)
```

# Load Fundamentals
```python
if not config.reuse_prepared_data:
    debug_dir.mkdir(parents=True, exist_ok=True)
    annual_long = load_fundamentals(config.universe, "annual")
    quarterly_long = load_fundamentals(config.universe, "quarterly")

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

annual_long.head(10) if not config.reuse_prepared_data else pd.DataFrame()
```

# Pivot Annual Fundamentals
```python
if not config.reuse_prepared_data:
    annual_mapped = annual_long.copy()
    annual_mapped["metric"] = [
        canonical_item_lookup.get((report_type, item_en))
        for report_type, item_en in zip(annual_mapped["report_type"], annual_mapped["item_en"], strict=True)
    ]
    annual_mapped = annual_mapped.dropna(subset=["metric"]).copy()
    annual_mapped["available_from"] = compute_available_from(annual_mapped["period_end"])

    annual_wide = (
        annual_mapped.sort_values(["symbol", "fiscal_year", "period_end", "report_date", "metric"])
        .groupby(["symbol", "fiscal_year", "period_end", "available_from", "metric"], as_index=False)["value"]
        .last()
        .pivot_table(
            index=["symbol", "fiscal_year", "period_end", "available_from"],
            columns="metric",
            values="value",
            aggfunc="last",
        )
        .reset_index()
        .sort_values(["symbol", "available_from"])
        .reset_index(drop=True)
    )
    annual_wide.columns = [str(column) for column in annual_wide.columns]

annual_wide.head(10) if not config.reuse_prepared_data else pd.DataFrame()
```

# Annual Fundamental Features
```python
if not config.reuse_prepared_data:
    annual_features = annual_wide.copy()

    if (
        "debtToAssetsRatio" not in annual_features.columns
        and {"totalLiabilities", "totalAssets"}.issubset(annual_features.columns)
    ):
        annual_features["debtToAssetsRatio"] = (
            annual_features["totalLiabilities"] / annual_features["totalAssets"].replace(0.0, np.nan)
        ) * 100.0

    annual_features = annual_features.sort_values(["symbol", "available_from"]).reset_index(drop=True)
    annual_group = annual_features.groupby("symbol", group_keys=False)

    if "revenue" in annual_features.columns:
        annual_revenue_prev = annual_group["revenue"].shift(1)
        annual_features["revenue_yoy"] = (
            (annual_features["revenue"] - annual_revenue_prev) / annual_revenue_prev.abs()
        ).where(annual_revenue_prev.abs() > 1e-12)

    if "netIncome" in annual_features.columns:
        annual_net_income_prev = annual_group["netIncome"].shift(1)
        annual_features["netIncome_yoy"] = (
            (annual_features["netIncome"] - annual_net_income_prev) / annual_net_income_prev.abs()
        ).where(annual_net_income_prev.abs() > 1e-12)

    if "profitBeforeTax" in annual_features.columns:
        annual_profit_before_tax_prev = annual_group["profitBeforeTax"].shift(1)
        annual_features["profitBeforeTax_yoy"] = (
            (annual_features["profitBeforeTax"] - annual_profit_before_tax_prev) / annual_profit_before_tax_prev.abs()
        ).where(annual_profit_before_tax_prev.abs() > 1e-12)

    if "totalAssets" in annual_features.columns:
        annual_total_assets_prev = annual_group["totalAssets"].shift(1)
        annual_features["totalAssets_yoy"] = (
            (annual_features["totalAssets"] - annual_total_assets_prev) / annual_total_assets_prev.abs()
        ).where(annual_total_assets_prev.abs() > 1e-12)

    annual_features = annual_features.replace([np.inf, -np.inf], np.nan)

    annual_features = annual_features.drop(columns=["period_end"], errors="ignore")

annual_features[
    [
        "symbol",
        "available_from",
        "revenue",
        "revenue_yoy",
        "netIncome",
        "netIncome_yoy",
        "profitBeforeTax",
        "profitBeforeTax_yoy",
        "totalAssets",
        "totalAssets_yoy",
        "debtToAssetsRatio",
    ]
].head(10) if not config.reuse_prepared_data else pd.DataFrame()
```

# Annual Fundamental Feature NaN Summary
```python
if not config.reuse_prepared_data:
    annual_feature_nan_summary = pd.DataFrame(
        {
            "nan_pct": annual_features[
                [
                    "revenue_yoy",
                    "netIncome_yoy",
                    "profitBeforeTax_yoy",
                    "totalAssets_yoy",
                    "debtToAssetsRatio",
                ]
            ].isna().mean().round(4)
        }
    )
else:
    annual_feature_nan_summary = pd.DataFrame()

annual_feature_nan_summary
```

# Pivot Quarterly Fundamentals
```python
if not config.reuse_prepared_data:
    quarterly_mapped = quarterly_long.copy()
    quarterly_mapped["metric"] = [
        canonical_item_lookup.get((report_type, item_en))
        for report_type, item_en in zip(quarterly_mapped["report_type"], quarterly_mapped["item_en"], strict=True)
    ]
    quarterly_mapped = quarterly_mapped.dropna(subset=["metric"]).copy()
    quarterly_mapped["available_from"] = compute_available_from(quarterly_mapped["period_end"])

    quarterly_wide = (
        quarterly_mapped.sort_values(["symbol", "fiscal_year", "quarter", "period_end", "report_date", "metric"])
        .groupby(["symbol", "fiscal_year", "quarter", "period_end", "available_from", "metric"], as_index=False)["value"]
        .last()
        .pivot_table(
            index=["symbol", "fiscal_year", "quarter", "period_end", "available_from"],
            columns="metric",
            values="value",
            aggfunc="last",
        )
        .reset_index()
        .sort_values(["symbol", "available_from"])
        .reset_index(drop=True)
    )
    quarterly_wide.columns = [str(column) for column in quarterly_wide.columns]

quarterly_wide.head(10) if not config.reuse_prepared_data else pd.DataFrame()
```

# Quarterly Fundamental Features
```python
if not config.reuse_prepared_data:
    quarterly_features = quarterly_wide.copy()

    if (
        "debtToAssetsRatio" not in quarterly_features.columns
        and {"totalLiabilities", "totalAssets"}.issubset(quarterly_features.columns)
    ):
        quarterly_features["debtToAssetsRatio"] = (
            quarterly_features["totalLiabilities"] / quarterly_features["totalAssets"].replace(0.0, np.nan)
        ) * 100.0

    quarterly_features = quarterly_features.sort_values(["symbol", "available_from"]).reset_index(drop=True)
    quarterly_group = quarterly_features.groupby("symbol", group_keys=False)

    if "revenue" in quarterly_features.columns:
        quarterly_features["revenue_ttm"] = quarterly_group["revenue"].transform(
            lambda series: series.rolling(4, min_periods=4).sum()
        )
        quarterly_revenue_prev_q = quarterly_group["revenue"].shift(1)
        quarterly_revenue_prev_y = quarterly_group["revenue"].shift(4)
        quarterly_features["revenue_qoq"] = (
            (quarterly_features["revenue"] - quarterly_revenue_prev_q) / quarterly_revenue_prev_q.abs()
        ).where(quarterly_revenue_prev_q.abs() > 1e-12)
        quarterly_features["revenue_q_yoy"] = (
            (quarterly_features["revenue"] - quarterly_revenue_prev_y) / quarterly_revenue_prev_y.abs()
        ).where(quarterly_revenue_prev_y.abs() > 1e-12)

    if "netIncome" in quarterly_features.columns:
        quarterly_features["netIncome_ttm"] = quarterly_group["netIncome"].transform(
            lambda series: series.rolling(4, min_periods=4).sum()
        )
        quarterly_net_income_prev_q = quarterly_group["netIncome"].shift(1)
        quarterly_net_income_prev_y = quarterly_group["netIncome"].shift(4)
        quarterly_features["netIncome_qoq"] = (
            (quarterly_features["netIncome"] - quarterly_net_income_prev_q) / quarterly_net_income_prev_q.abs()
        ).where(quarterly_net_income_prev_q.abs() > 1e-12)
        quarterly_features["netIncome_q_yoy"] = (
            (quarterly_features["netIncome"] - quarterly_net_income_prev_y) / quarterly_net_income_prev_y.abs()
        ).where(quarterly_net_income_prev_y.abs() > 1e-12)

    if "profitBeforeTax" in quarterly_features.columns:
        quarterly_features["profitBeforeTax_ttm"] = quarterly_group["profitBeforeTax"].transform(
            lambda series: series.rolling(4, min_periods=4).sum()
        )
        quarterly_profit_before_tax_prev_q = quarterly_group["profitBeforeTax"].shift(1)
        quarterly_profit_before_tax_prev_y = quarterly_group["profitBeforeTax"].shift(4)
        quarterly_features["profitBeforeTax_qoq"] = (
            (quarterly_features["profitBeforeTax"] - quarterly_profit_before_tax_prev_q)
            / quarterly_profit_before_tax_prev_q.abs()
        ).where(quarterly_profit_before_tax_prev_q.abs() > 1e-12)
        quarterly_features["profitBeforeTax_q_yoy"] = (
            (quarterly_features["profitBeforeTax"] - quarterly_profit_before_tax_prev_y)
            / quarterly_profit_before_tax_prev_y.abs()
        ).where(quarterly_profit_before_tax_prev_y.abs() > 1e-12)

    quarterly_features = quarterly_features.replace([np.inf, -np.inf], np.nan)

    quarterly_features = quarterly_features.drop(columns=["period_end"], errors="ignore")

quarterly_features[
    [
        "symbol",
        "available_from",
        "revenue",
        "revenue_ttm",
        "revenue_qoq",
        "revenue_q_yoy",
        "netIncome",
        "netIncome_ttm",
        "netIncome_qoq",
        "netIncome_q_yoy",
        "profitBeforeTax",
        "profitBeforeTax_ttm",
        "profitBeforeTax_qoq",
        "profitBeforeTax_q_yoy",
    ]
].head(10) if not config.reuse_prepared_data else pd.DataFrame()
```

# Quarterly Fundamental Feature NaN Summary
```python
if not config.reuse_prepared_data:
    quarterly_feature_nan_summary = pd.DataFrame(
        {
            "nan_pct": quarterly_features[
                [
                    "revenue_ttm",
                    "revenue_qoq",
                    "revenue_q_yoy",
                    "netIncome_ttm",
                    "netIncome_qoq",
                    "netIncome_q_yoy",
                    "profitBeforeTax_ttm",
                    "profitBeforeTax_qoq",
                    "profitBeforeTax_q_yoy",
                ]
            ].isna().mean().round(4)
        }
    )
else:
    quarterly_feature_nan_summary = pd.DataFrame()

quarterly_feature_nan_summary
```

# Merge Fundamentals Into Prices
```python
if not config.reuse_prepared_data:
    annual_metric_columns = [column for column in annual_features.columns if column not in FUNDAMENTAL_AUX_COLUMNS]
    quarterly_metric_columns = [column for column in quarterly_features.columns if column not in FUNDAMENTAL_AUX_COLUMNS]

    annual_block = annual_features.rename(columns={column: f"{column}_a" for column in annual_metric_columns}).copy()
    quarterly_block = quarterly_features.rename(columns={column: f"{column}_q" for column in quarterly_metric_columns}).copy()

    price_df = prices.reset_index().sort_values(["symbol", "date"]).copy()
    price_df["date"] = pd.to_datetime(price_df["date"]).astype("datetime64[ns]")
    annual_block["available_from"] = pd.to_datetime(annual_block["available_from"]).astype("datetime64[ns]")
    quarterly_block["available_from"] = pd.to_datetime(quarterly_block["available_from"]).astype("datetime64[ns]")

    merged_frames = []

    for symbol, group in price_df.groupby("symbol", sort=False):
        merged_symbol = group.sort_values("date").copy()

        annual_symbol = annual_block.loc[annual_block["symbol"] == symbol].sort_values("available_from")
        if annual_symbol.empty:
            for column in [f"{column}_a" for column in annual_metric_columns]:
                merged_symbol[column] = np.nan
        else:
            merged_symbol = pd.merge_asof(
                merged_symbol,
                annual_symbol.drop(columns=["symbol", "fiscal_year"], errors="ignore").sort_values("available_from"),
                left_on="date",
                right_on="available_from",
                direction="backward",
            ).drop(columns=["available_from"], errors="ignore")

        quarterly_symbol = quarterly_block.loc[quarterly_block["symbol"] == symbol].sort_values("available_from")
        if quarterly_symbol.empty:
            for column in [f"{column}_q" for column in quarterly_metric_columns]:
                merged_symbol[column] = np.nan
        else:
            merged_symbol = pd.merge_asof(
                merged_symbol,
                quarterly_symbol.drop(columns=["symbol", "fiscal_year", "quarter"], errors="ignore").sort_values("available_from"),
                left_on="date",
                right_on="available_from",
                direction="backward",
            ).drop(columns=["available_from"], errors="ignore")

        merged_frames.append(merged_symbol)

    merged = pd.concat(merged_frames, ignore_index=True).sort_values(["date", "symbol"])
    annual_available = merged[[f"{column}_a" for column in annual_metric_columns]].notna().any(axis=1)
    quarterly_available = merged[[f"{column}_q" for column in quarterly_metric_columns]].notna().any(axis=1)
    merged = merged.loc[annual_available | quarterly_available]
    merged = merged.set_index(["date", "symbol"]).sort_index()

merged.head(10) if not config.reuse_prepared_data else featured.head(10)
```

# Momentum Features
```python
if not config.reuse_prepared_data:
    featured = merged.copy()
    safe_close = featured["close"].replace(0.0, np.nan)
    close_by_symbol = safe_close.groupby(level="symbol", sort=False)

    featured["mom_21"] = np.log(safe_close / close_by_symbol.shift(21)).replace([np.inf, -np.inf], np.nan)
    featured["mom_63"] = np.log(safe_close / close_by_symbol.shift(63)).replace([np.inf, -np.inf], np.nan)
    featured["mom_126"] = np.log(safe_close / close_by_symbol.shift(126)).replace([np.inf, -np.inf], np.nan)
    featured["mom_252"] = np.log(safe_close / close_by_symbol.shift(252)).replace([np.inf, -np.inf], np.nan)

featured[["close", "mom_21", "mom_63", "mom_126", "mom_252"]].head(10)
```

# Momentum Feature NaN Summary
```python
pd.DataFrame(
    {
        "nan_pct": featured[["mom_21", "mom_63", "mom_126", "mom_252"]].isna().mean().round(4)
    }
)
```

# Volatility Features
```python
if not config.reuse_prepared_data:
    def _volatility_block(group: pd.DataFrame) -> pd.DataFrame:
        close = group["close"].replace(0.0, np.nan)
        high = group["high"]
        low = group["low"]
        log_ret_1 = np.log(close / close.shift(1))
        prev_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return pd.DataFrame(
            {
                "vol_21": log_ret_1.rolling(21, min_periods=21).std(),
                "vol_63": log_ret_1.rolling(63, min_periods=63).std(),
                "atr_14_norm": true_range.rolling(14, min_periods=14).mean() / close.replace(0.0, np.nan),
                "vol_ratio_21_63": log_ret_1.rolling(21, min_periods=21).std()
                / log_ret_1.rolling(63, min_periods=63).std().replace(0.0, np.nan),
            },
            index=group.index,
        )

    volatility_features = (
        featured.groupby(level="symbol", group_keys=False, sort=False)
        .apply(_volatility_block)
        .sort_index()
    )

    for column in volatility_features.columns:
        featured[column] = volatility_features[column]

    featured[["vol_21", "vol_63", "atr_14_norm", "vol_ratio_21_63"]] = (
        featured[["vol_21", "vol_63", "atr_14_norm", "vol_ratio_21_63"]]
        .replace([np.inf, -np.inf], np.nan)
    )

featured[["close", "vol_21", "vol_63", "atr_14_norm", "vol_ratio_21_63"]].head(10)
```

# Volatility Feature NaN Summary
```python
pd.DataFrame(
    {
        "nan_pct": featured[["vol_21", "vol_63", "atr_14_norm", "vol_ratio_21_63"]].isna().mean().round(4)
    }
)
```

# Trend Features
```python
if not config.reuse_prepared_data:
    def _trend_block(group: pd.DataFrame) -> pd.DataFrame:
        close = group["close"].replace(0.0, np.nan)
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

    trend_features = (
        featured.groupby(level="symbol", group_keys=False, sort=False)
        .apply(_trend_block)
        .sort_index()
    )

    for column in trend_features.columns:
        featured[column] = trend_features[column]

    featured[["log_close_to_ma_50", "log_close_to_ma_200", "ma_cross"]] = (
        featured[["log_close_to_ma_50", "log_close_to_ma_200", "ma_cross"]]
        .replace([np.inf, -np.inf], np.nan)
    )

featured[["close", "log_close_to_ma_50", "log_close_to_ma_200", "ma_cross"]].head(10)
```

# Trend Feature NaN Summary
```python
pd.DataFrame(
    {
        "nan_pct": featured[["log_close_to_ma_50", "log_close_to_ma_200", "ma_cross"]].isna().mean().round(4)
    }
)
```

# Liquidity Features
```python
if not config.reuse_prepared_data:
    def _liquidity_block(group: pd.DataFrame) -> pd.DataFrame:
        close = group["close"].replace(0.0, np.nan)
        volume = group["volume"].replace(0.0, np.nan)
        log_ret_1 = np.log(close / close.shift(1)).abs()
        amihud_daily = (log_ret_1 / volume) * close
        return pd.DataFrame(
            {
                "relative_volume_20": volume / volume.rolling(20, min_periods=20).mean(),
                "amihud_illiq_21": amihud_daily.rolling(21, min_periods=21).mean(),
            },
            index=group.index,
        )

    liquidity_features = (
        featured.groupby(level="symbol", group_keys=False, sort=False)
        .apply(_liquidity_block)
        .sort_index()
    )

    for column in liquidity_features.columns:
        featured[column] = liquidity_features[column]

    featured[["relative_volume_20", "amihud_illiq_21"]] = (
        featured[["relative_volume_20", "amihud_illiq_21"]]
        .replace([np.inf, -np.inf], np.nan)
    )

featured[["close", "volume", "relative_volume_20", "amihud_illiq_21"]].head(10)
```

# Liquidity Feature NaN Summary
```python
pd.DataFrame(
    {
        "nan_pct": featured[["relative_volume_20", "amihud_illiq_21"]].isna().mean().round(4)
    }
)
```

# Fundamental Model Features
```python
if not config.reuse_prepared_data:
    featured["roe"] = _combine_first(featured, "returnOnEquity_q", "returnOnEquity_a")
    featured["roa"] = _combine_first(featured, "returnOnAssets_q", "returnOnAssets_a")
    featured["pe"] = _combine_first(featured, "priceToEarningsRatio_q", "priceToEarningsRatio_a")
    featured["pb"] = _combine_first(featured, "priceToBookRatio_q", "priceToBookRatio_a")
    featured["debt_ratio"] = _combine_first(featured, "debtToAssetsRatio_q", "debtToAssetsRatio_a")

    total_equity = _combine_first(featured, "totalEquity_a")
    total_assets = _combine_first(featured, "totalAssets_a").replace(0.0, np.nan)
    featured["equity_ratio"] = total_equity / total_assets

    featured["revenue_ttm"] = _combine_first(featured, "revenue_ttm_q")
    featured["net_income_ttm"] = _combine_first(featured, "netIncome_ttm_q")
    featured["profit_before_tax_ttm"] = _combine_first(featured, "profitBeforeTax_ttm_q")

    featured["revenue_yoy"] = _combine_first(featured, "revenue_yoy_a")
    featured["net_income_yoy"] = _combine_first(featured, "netIncome_yoy_a")
    featured["profit_before_tax_yoy"] = _combine_first(featured, "profitBeforeTax_yoy_a")

    featured["revenue_q_yoy"] = _combine_first(featured, "revenue_q_yoy_q")
    featured["net_income_q_yoy"] = _combine_first(featured, "netIncome_q_yoy_q")
    featured["profit_before_tax_q_yoy"] = _combine_first(featured, "profitBeforeTax_q_yoy_q")

    featured["revenue_qoq"] = _combine_first(featured, "revenue_qoq_q")
    featured["net_income_qoq"] = _combine_first(featured, "netIncome_qoq_q")
    featured["profit_before_tax_qoq"] = _combine_first(featured, "profitBeforeTax_qoq_q")

featured[
    [
        "roe",
        "roa",
        "pe",
        "pb",
        "debt_ratio",
        "equity_ratio",
        "revenue_ttm",
        "net_income_ttm",
        "profit_before_tax_ttm",
        "revenue_yoy",
        "net_income_yoy",
        "profit_before_tax_yoy",
        "revenue_q_yoy",
        "net_income_q_yoy",
        "profit_before_tax_q_yoy",
        "revenue_qoq",
        "net_income_qoq",
        "profit_before_tax_qoq",
    ]
].head(10)
```

# Fundamental Model Feature NaN Summary
```python
pd.DataFrame(
    {
        "nan_pct": featured[
            [
                "roe",
                "roa",
                "pe",
                "pb",
                "debt_ratio",
                "equity_ratio",
                "revenue_ttm",
                "net_income_ttm",
                "profit_before_tax_ttm",
                "revenue_yoy",
                "net_income_yoy",
                "profit_before_tax_yoy",
                "revenue_q_yoy",
                "net_income_q_yoy",
                "profit_before_tax_q_yoy",
                "revenue_qoq",
                "net_income_qoq",
                "profit_before_tax_qoq",
            ]
        ].isna().mean().round(4)
    }
)
```

# Feature Coverage
```python
if not config.reuse_prepared_data:
    base_columns = set(merged.columns)
    all_feature_cols = [column for column in featured.columns if column not in base_columns]
    null_threshold = 0.99
    feature_null_stats = []
    feature_cols = []

    for column in all_feature_cols:
        null_pct = float(featured[column].isna().mean())
        kept = null_pct < null_threshold
        feature_null_stats.append({"feature": column, "null_pct": round(null_pct, 4), "included": kept})
        if kept:
            feature_cols.append(column)

    feature_null_rates = pd.DataFrame(feature_null_stats).sort_values("null_pct", ascending=False)
    _save_debug(feature_null_rates, debug_dir / "feature_null_rates.csv", "feature_null_rates")
else:
    feature_null_rates = pd.DataFrame({"feature": feature_cols, "null_pct": np.nan, "included": True})

feature_null_rates.head(20)
```

# Save Prepared Data
```python
if not config.reuse_prepared_data:
    prepared_dir = save_prepared_data(config, prices, benchmark_price_frame, featured, feature_cols)
else:
    prepared_dir = resolve_prepared_data_dir(config)

prepared_dir
```

# Forward Returns
```python
forward_return_column = f"forward_log_return_{config.target_horizon}"

labeled_source = featured.copy().sort_index()
labeled_source[forward_return_column] = labeled_source["close"].groupby(level="symbol", sort=False).transform(
    lambda series: np.log(series.shift(-config.target_horizon) / series)
)

min_features = _min_feature_observations(feature_cols)
coverage_mask = _feature_coverage_mask(
    labeled_source,
    feature_cols,
    min_non_null_features=min_features,
)

clean_labeled_source = labeled_source.loc[coverage_mask].dropna(subset=[forward_return_column]).copy()

clean_labeled_source[[forward_return_column]].dropna().head(10)
```

# Cross Sectional Labels
```python
forward_wide = clean_labeled_source[forward_return_column].unstack("symbol")
label_wide = pd.DataFrame(index=forward_wide.index, columns=forward_wide.columns, dtype=float)

for current_date, row in forward_wide.iterrows():
    valid = row.dropna()
    if len(valid) < 5:
        continue
    ranked = valid.rank(method="first")
    label_wide.loc[current_date, valid.index] = pd.qcut(ranked, q=5, labels=False)

label_long = label_wide.stack(future_stack=True).dropna().rename("label").reset_index()
labeled = clean_labeled_source.reset_index().merge(label_long, on=["date", "symbol"], how="inner")
labeled["label"] = labeled["label"].astype(int)
labeled = labeled.set_index(["date", "symbol"]).sort_index()

save_labeled_frame(config, labeled, config.target_horizon)

labeled[["label", forward_return_column]].head(10)
```

# Labeled Frame Summary
```python
pd.DataFrame(
    {
        "value": [
            len(labeled),
            labeled.index.get_level_values("symbol").nunique(),
            labeled.reset_index()["date"].nunique(),
            labeled["label"].value_counts().sort_index().to_dict(),
        ]
    },
    index=["labeled_rows", "labeled_symbols", "labeled_dates", "label_distribution"],
)
```

# Model And Walk Forward Helpers
```python
class XGBPredictor:
    def __init__(
        self,
        *,
        max_depth: int = 3,
        n_estimators: int = 30,
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
            "n_jobs": 1,
        }
        self.model: object | None = None
        self.constant_class: int | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
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


def build_labeled_frame_for_horizon(featured: pd.DataFrame, target_horizon: int, feature_cols: list[str]) -> pd.DataFrame:
    labeled_source = featured.copy().sort_index()
    forward_return_column = f"forward_log_return_{target_horizon}"
    labeled_source[forward_return_column] = labeled_source["close"].groupby(level="symbol", sort=False).transform(
        lambda series: np.log(series.shift(-target_horizon) / series)
    )

    min_features = _min_feature_observations(feature_cols)
    coverage_mask = _feature_coverage_mask(
        labeled_source,
        feature_cols,
        min_non_null_features=min_features,
    )

    clean = labeled_source.loc[coverage_mask].dropna(subset=[forward_return_column]).copy()
    forward_wide = clean[forward_return_column].unstack("symbol")
    label_wide = pd.DataFrame(index=forward_wide.index, columns=forward_wide.columns, dtype=float)

    for current_date, row in forward_wide.iterrows():
        valid = row.dropna()
        if len(valid) < 5:
            continue
        ranked = valid.rank(method="first")
        label_wide.loc[current_date, valid.index] = pd.qcut(ranked, q=5, labels=False)

    label_long = label_wide.stack(future_stack=True).dropna().rename("label").reset_index()
    labeled = clean.reset_index().merge(label_long, on=["date", "symbol"], how="inner")
    labeled["label"] = labeled["label"].astype(int)
    return labeled.set_index(["date", "symbol"]).sort_index()


def walk_forward_predict(
    labeled: pd.DataFrame,
    predictor: XGBPredictor,
    config: RunConfig,
    feature_cols: list[str],
    prediction_start_date: date,
    prediction_end_date: date,
) -> pd.DataFrame:
    frame = labeled.reset_index().sort_values(["date", "symbol"]).copy()
    all_dates = sorted(pd.to_datetime(frame["date"]).drop_duplicates().tolist())
    rebalance_dates = [
        rebalance_date
        for index, rebalance_date in enumerate(all_dates)
        if index % config.rebalance_period == 0
        and prediction_start_date <= rebalance_date.date() <= prediction_end_date
    ]
    rows = []
    min_features = _min_feature_observations(feature_cols)

    for rebalance_date in rebalance_dates:
        idx = all_dates.index(rebalance_date)
        latest_known_label_index = idx - config.target_horizon
        if latest_known_label_index < 0:
            continue

        trainable_dates = all_dates[: latest_known_label_index + 1]
        if len(trainable_dates) < config.train_window:
            continue

        train_dates = trainable_dates[-config.train_window :]
        train = frame.loc[frame["date"].isin(train_dates)].dropna(subset=["label"])
        train = train.loc[_feature_coverage_mask(train, feature_cols, min_non_null_features=min_features)]

        predict_slice = frame.loc[frame["date"] == rebalance_date]
        predict_slice = predict_slice.loc[
            _feature_coverage_mask(predict_slice, feature_cols, min_non_null_features=min_features)
        ]

        if train.empty or predict_slice.empty:
            continue

        predictor.fit(train[feature_cols], train["label"])
        probabilities = predictor.predict_proba(predict_slice[feature_cols])
        pred_class = probabilities.idxmax(axis=1)

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

    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "pred_class", "prob_strong_bull"])

    return pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


def build_portfolio_weights(
    predictions: pd.DataFrame,
    feature_frame: pd.DataFrame,
    prices: pd.DataFrame,
    config: RunConfig,
) -> pd.DataFrame:
    if predictions.empty:
        empty = pd.DataFrame(columns=["date", "symbol", "weight"])
        empty.attrs["rebalance_dates"] = []
        return empty

    feature_pdf = feature_frame.reset_index()
    price_dates = sorted(prices.index.get_level_values("date").unique().tolist())
    next_date_map = {current: price_dates[index + 1] for index, current in enumerate(price_dates[:-1])}
    execution_dates: list[pd.Timestamp] = []
    rows: list[dict[str, object]] = []

    for signal_date, group in predictions.groupby("date", sort=True):
        signal_timestamp = pd.Timestamp(signal_date)
        execution_date = next_date_map.get(signal_timestamp)
        if execution_date is None:
            continue

        df_at_t = feature_pdf.loc[feature_pdf["date"] == signal_timestamp].copy()
        if df_at_t.empty:
            continue

        roe = df_at_t.get("roe", pd.Series(np.nan, index=df_at_t.index))
        min_roe_units = _percent_point_threshold_in_series_units(roe, 5.0)
        roe_ok = (roe >= min_roe_units).fillna(False)

        net_income_qoq = df_at_t.get("net_income_qoq", pd.Series(np.nan, index=df_at_t.index))
        income_ok = net_income_qoq.isna() | (net_income_qoq > -1.0)

        debt_ratio = df_at_t.get("debt_ratio", pd.Series(np.nan, index=df_at_t.index))
        debt_ok = (debt_ratio <= 70.0).fillna(False)

        equity_ratio = df_at_t.get("equity_ratio", pd.Series(np.nan, index=df_at_t.index))
        equity_ok = equity_ratio.isna() | (equity_ratio > 0.0)

        eligible_mask = roe_ok & income_ok & debt_ok & equity_ok
        eligible_symbols = set(df_at_t.loc[eligible_mask, "symbol"].tolist())
        if not eligible_symbols:
            continue

        signals = group.loc[group["symbol"].isin(eligible_symbols)]
        signals = signals.loc[signals["pred_class"] == config.signal_class]
        if signals.empty:
            continue

        selected = signals.nlargest(config.max_positions, "prob_strong_bull")
        execution_dates.append(pd.Timestamp(execution_date))

        for symbol in selected["symbol"].tolist():
            rows.append(
                {
                    "date": execution_date,
                    "symbol": symbol,
                    "weight": float(1.0 / config.max_positions),
                }
            )

    result = pd.DataFrame(rows, columns=["date", "symbol", "weight"])
    result.attrs["rebalance_dates"] = sorted(dict.fromkeys(execution_dates))
    return result.sort_values(["date", "symbol"]).reset_index(drop=True)
```

# Optimization Helper
```python
objective = cagr_with_drawdown_penalty(max_dd_limit=config.max_drawdown_limit)


def run_optimization(params: dict[str, object]) -> BacktestResult:
    tuned = replace(
        config,
        train_window=int(params.get("train_window", config.train_window)),
        target_horizon=int(params.get("target_horizon", config.target_horizon)),
        max_positions=int(params.get("max_positions", config.max_positions)),
        signal_class=int(params.get("signal_class", config.signal_class)),
    )

    labeled_cache_path = prepared_labeled_cache_path(tuned, tuned.target_horizon)
    if config.reuse_prepared_data and labeled_cache_path.exists():
        tuned_labeled = load_labeled_frame(tuned, tuned.target_horizon)
    else:
        tuned_labeled = build_labeled_frame_for_horizon(featured, tuned.target_horizon, feature_cols)
        save_labeled_frame(tuned, tuned_labeled, tuned.target_horizon)

    predictor = XGBPredictor(
        max_depth=int(params.get("max_depth", 3)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        n_estimators=30,
        seed=int(params.get("seed", tuned.seed)),
    )

    predictions = walk_forward_predict(
        tuned_labeled,
        predictor,
        tuned,
        feature_cols,
        prediction_start_date=tuned.download_start_date,
        prediction_end_date=tuned.optimize_end_date,
    )

    weights = build_portfolio_weights(predictions, featured, prices, tuned)

    return run_backtest(
        weights,
        prices,
        tuned,
        evaluation_start_date=tuned.download_start_date,
        evaluation_end_date=tuned.optimize_end_date,
    )
```

# Run Hyperparameter Search
```python
if run_optuna_search:
    optuna_result = run_optuna(
        run_fn=run_optimization,
        search_space=SEARCH_SPACE,
        objective=objective,
        n_trials=config.n_trials,
        seed=config.seed,
    )
    search_mode = "optuna"
else:
    optuna_result = OptunaResult(
        study=None,
        best_result=None,
        best_params=fixed_best_params.copy(),
    )
    search_mode = "fixed_params"

pd.DataFrame(
    {
        "value": [
            search_mode,
            optuna_result.study.best_value if optuna_result.study is not None else np.nan,
            optuna_result.best_params,
        ]
    },
    index=["search_mode", "best_objective_value", "best_params"],
)
```

# Inference Configuration
```python
best_cfg = replace(
    config,
    train_window=int(optuna_result.best_params.get("train_window", config.train_window)),
    target_horizon=int(optuna_result.best_params.get("target_horizon", config.target_horizon)),
    max_positions=int(optuna_result.best_params.get("max_positions", config.max_positions)),
    signal_class=int(optuna_result.best_params.get("signal_class", config.signal_class)),
)

best_cfg
```

# Inference Labeled Frame
```python
best_labeled_cache_path = prepared_labeled_cache_path(best_cfg, best_cfg.target_horizon)

if config.reuse_prepared_data and best_labeled_cache_path.exists():
    best_labeled = load_labeled_frame(best_cfg, best_cfg.target_horizon)
else:
    best_labeled = build_labeled_frame_for_horizon(featured, best_cfg.target_horizon, feature_cols)
    save_labeled_frame(best_cfg, best_labeled, best_cfg.target_horizon)

best_labeled[["label"]].head(10)
```

# Generate Inference Predictions
```python
best_predictor = XGBPredictor(
    max_depth=int(optuna_result.best_params.get("max_depth", 3)),
    learning_rate=float(optuna_result.best_params.get("learning_rate", 0.05)),
    n_estimators=30,
    seed=best_cfg.seed,
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

inference_predictions.head(20)
```

# Inference Prediction Summary
```python
pd.DataFrame(
    {
        "value": [
            len(inference_predictions),
            inference_predictions["date"].nunique() if not inference_predictions.empty else 0,
            inference_predictions["pred_class"].value_counts().sort_index().to_dict()
            if not inference_predictions.empty
            else {},
        ]
    },
    index=["prediction_rows", "prediction_dates", "class_distribution"],
)
```

# Build Portfolio Weights
```python
final_weights = build_portfolio_weights(
    inference_predictions,
    featured,
    prices,
    best_cfg,
)

_save_debug(final_weights, debug_dir / "portfolio_weights.csv", "portfolio_weights")

final_weights.head(20)
```

# Portfolio Weight Summary
```python
pd.DataFrame(
    {
        "value": [
            len(final_weights),
            final_weights["date"].nunique() if not final_weights.empty else 0,
            final_weights["weight"].mean() if not final_weights.empty else 0.0,
        ]
    },
    index=["weight_rows", "weight_dates", "average_weight"],
)
```

# Run Backtest
```python
final_result = run_backtest(
    final_weights,
    prices,
    best_cfg,
    evaluation_start_date=best_cfg.inference_start_date,
    evaluation_end_date=best_cfg.inference_end_date,
)

pd.DataFrame(
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
```

# Best Parameters
```python
pd.Series(optuna_result.best_params, name="best_param").to_frame()
```

# Equity Curve
```python
final_result.equity_curve.plot(figsize=(12, 4), title="Inference Equity Curve")
```

# Save Outputs
```python
save_outputs(optuna_result, final_result, benchmark_price_frame, best_cfg)

sorted(path.name for path in best_cfg.output_dir.iterdir())
```
