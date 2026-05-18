"""Run the VN100 quantamental ML strategy from the command line.

The notebook version is useful for exploration. This script keeps the same
strategy path but makes the data fetch, feature build, walk-forward backtest,
artifacts, and optional MLflow logging runnable end to end.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import math
import os
import sys
import threading
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from itertools import product
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "qts").exists() and (candidate / "pyproject.toml").exists():
            return candidate
        nested = candidate / "QTradeSystematic"
        if (nested / "qts").exists() and (nested / "pyproject.toml").exists():
            return nested
    raise FileNotFoundError("Could not find the QTradeSystematic project root")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("QTS_ROOT", str(PROJECT_ROOT / ".qts_notebook_runtime"))
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import polars as pl
import yaml

import qts  # noqa: F401 - import registry side effects
from qts.data.manager import DataManager
from qts.data.sources.vnstock import VnstockDataSource
from qts.data.storage.duckdb import DuckDBStorage
from qts.data.storage.parquet import ParquetStorage
from qts.research.backtest._runner import _rebalance_dates, run_backtest_frame
from qts.research.backtest.base import (
    BacktestConfig,
    BacktestResult,
    CommissionConfig,
    UniverseConfig,
)
from qts.research.features.forward_returns import ForwardReturns
from qts.research.features.fundamentals import (
    FUNDAMENTAL_FACTOR_GROUPS,
    VNFundamentalFeatures,
)
from qts.research.features.fundamentals import (
    add_factor_scores as add_fundamental_factor_scores,
)
from qts.research.features.indicators.momentum import ROCFeature, RSIFeature
from qts.research.features.indicators.trend import MACDFeature
from qts.research.features.indicators.volatility import ATRFeature, HistVolFeature
from qts.research.features.preprocessor import (
    flag_anomalies,
    preprocess_ohlcv,
    remove_flagged_symbols,
)
from qts.research.strategies.base import BaseStrategy
from qts.research.portfolio_construction import (
    long_short_equal_weight_portfolio,
)
from qts.research.strategies.ml_factor.model import train_and_predict_xgb_classifier
from qts.utils.paths import cache_dir

MLFLOW_AVAILABLE = importlib.util.find_spec("mlflow") is not None
if MLFLOW_AVAILABLE:
    import mlflow
else:
    mlflow = None

pd.options.display.float_format = "{:,.4f}".format

DEFAULT_RUNTIME_ROOT = PROJECT_ROOT / ".qts_notebook_runtime" / "vn100_quantamental_mlflow"
DEFAULT_ASSET_CONFIG = PROJECT_ROOT / "configs" / "assets" / "vn100.yml"
DEFAULT_STRATEGY_CONFIG = PROJECT_ROOT / "configs" / "strategies" / "ml_factor" / "base.yaml"
DEFAULT_MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
DEFAULT_MLFLOW_EXPERIMENT_NAME = "VN100 Quantamental ML"


def env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    parsed = int(value)
    return parsed if parsed > 0 else None


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_date(name: str, default: date) -> date:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else date.fromisoformat(value)


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def parse_csv(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be a positive integer")
    if value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:
            error["value"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if "value" in error:
        raise error["value"]
    return result["value"]


def normalize_vn_symbol(symbol: str, prefix: str = "VN:") -> str:
    raw = str(symbol).strip().upper()
    normalized_prefix = prefix.upper()
    return raw if raw.startswith(normalized_prefix) else f"{normalized_prefix}{raw}"


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def load_vn100_symbols(path: Path, max_symbols: int | None) -> tuple[list[str], str]:
    config = load_yaml_mapping(path)
    symbols = config.get("symbols") or []
    if not isinstance(symbols, list) or not symbols:
        raise ValueError(f"{path} must define a non-empty symbols list")
    prefix = str(config.get("symbol_prefix", "VN:"))
    normalized = [normalize_vn_symbol(item, prefix=prefix) for item in symbols]
    normalized = list(dict.fromkeys(normalized))
    benchmark = normalize_vn_symbol(config.get("benchmark_symbol", "VN100"), prefix=prefix)
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
        manager,
        symbols,
        start_date,
        end_date,
        interval,
        batch_size,
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


@dataclass(frozen=True)
class FeatureConfig:
    min_trading_days: int = 252
    volume_top_n: int | None = 80
    min_avg_volume: float = 50_000
    remove_large_gaps: bool = False
    max_gap_days: int = 21
    remove_low_volume: bool = False
    qsmom_fast: int = 21
    qsmom_slow: int = 252
    qsmom_returns: int = 126
    forward_period: int = 21
    fundamental_termtype: int = 1


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    feature: FeatureConfig
    predictor_cols: list[str] | None = None
    train_window: int = 504
    rebalance_period: int = 10
    num_long_positions: int = 10
    num_short_positions: int = 0
    long_threshold: float | None = None
    short_threshold: float | None = None
    model_params: dict[str, Any] | None = None
    initial_capital: Decimal = Decimal("1000000000")


TECHNICAL_BASE_COLUMNS = [
    "roc_21",
    "roc_63",
    "roc_126",
    "roc_252",
    "rsi_14",
    "rsi_63",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "atr_14",
    "hist_vol_21",
    "hist_vol_63",
    "hist_vol_126",
    "close_to_sma_50",
    "close_to_sma_200",
    "volume_ratio_20",
    "dollar_volume_zscore_63",
    "intraday_range",
    "close_location_value",
]

MODEL_PARAMS = {
    "random_state": 123,
    "objective": "multi:softprob",
    "num_class": 5,
    "eval_metric": "mlogloss",
    "n_estimators": 120,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "n_jobs": -1,
}


def qsmom_column(config: FeatureConfig) -> str:
    return f"close_qsmom_{config.qsmom_fast}_{config.qsmom_slow}_{config.qsmom_returns}"


def technical_columns(config: FeatureConfig) -> list[str]:
    return [qsmom_column(config), *TECHNICAL_BASE_COLUMNS]


def default_predictor_candidates(config: FeatureConfig) -> list[str]:
    return [
        qsmom_column(config),
        "technicalCompositeFactor",
        "fundamentalCompositeFactor",
        "hybridCompositeFactor",
        *FUNDAMENTAL_FACTOR_GROUPS.keys(),
        "roc_63",
        "roc_252",
        "rsi_63",
        "macd_hist",
        "hist_vol_63",
        "hist_vol_126",
        "close_to_sma_50",
        "close_to_sma_200",
        "volume_ratio_20",
        "dollar_volume_zscore_63",
    ]


def available_predictors(df: pl.DataFrame, candidates: list[str]) -> list[str]:
    return [
        col
        for col in candidates
        if col in df.columns and df.select(pl.col(col).is_not_null().sum()).item() > 0
    ]


def effective_long_threshold(value: float | None) -> float:
    return -math.inf if value is None else float(value)


def screen_liquid_universe(df: pl.DataFrame, config: FeatureConfig) -> pl.DataFrame:
    if config.volume_top_n is None:
        return df
    latest_date = df["date"].max()
    lookback_start = latest_date - timedelta(days=365)
    liquidity = (
        df.filter(pl.col("date") >= lookback_start)
        .group_by("symbol")
        .agg(
            pl.col("volume").mean().alias("avg_volume"),
            pl.len().alias("rows"),
            pl.col("close").last().alias("last_close"),
        )
        .filter((pl.col("avg_volume") >= config.min_avg_volume) & (pl.col("rows") >= 120))
        .sort("avg_volume", descending=True)
        .head(config.volume_top_n)
    )
    return df.filter(pl.col("symbol").is_in(liquidity["symbol"].to_list()))


def add_qsmom_features(df: pl.DataFrame, config: FeatureConfig) -> pl.DataFrame:
    fast = config.qsmom_fast
    slow = config.qsmom_slow
    returns = config.qsmom_returns
    out_col = qsmom_column(config)
    return (
        df.sort(["symbol", "date"])
        .with_columns(
            ((pl.col("close") / pl.col("close").shift(1).over("symbol")) - 1).alias(
                "_daily_return"
            ),
            (
                (
                    pl.col("close").shift(fast).over("symbol")
                    / pl.col("close").shift(slow).over("symbol")
                )
                - 1
            ).alias("_older_momentum"),
            ((pl.col("close") / pl.col("close").shift(fast).over("symbol")) - 1).alias(
                "_recent_momentum"
            ),
        )
        .with_columns(
            pl.col("_daily_return")
            .rolling_std(window_size=returns, min_samples=returns)
            .over("symbol")
            .alias("_return_vol")
        )
        .with_columns(
            (
                (pl.col("_older_momentum") - pl.col("_recent_momentum"))
                / (pl.col("_return_vol") + 1e-8)
            ).alias(out_col)
        )
        .drop(["_daily_return", "_older_momentum", "_recent_momentum", "_return_vol"])
    )


def add_price_action_features(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.sort(["symbol", "date"])
        .with_columns(
            pl.col("close").rolling_mean(20, min_samples=20).over("symbol").alias("sma_20"),
            pl.col("close").rolling_mean(50, min_samples=50).over("symbol").alias("sma_50"),
            pl.col("close").rolling_mean(200, min_samples=200).over("symbol").alias("sma_200"),
            pl.col("volume").rolling_mean(20, min_samples=20).over("symbol").alias("volume_sma_20"),
            (pl.col("close") * pl.col("volume")).alias("dollar_volume"),
            ((pl.col("high") - pl.col("low")) / (pl.col("close") + 1e-8)).alias("intraday_range"),
            ((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low") + 1e-8)).alias(
                "close_location_value"
            ),
        )
        .with_columns(
            ((pl.col("close") / (pl.col("sma_50") + 1e-8)) - 1).alias("close_to_sma_50"),
            ((pl.col("close") / (pl.col("sma_200") + 1e-8)) - 1).alias("close_to_sma_200"),
            (pl.col("volume") / (pl.col("volume_sma_20") + 1e-8)).alias("volume_ratio_20"),
            (
                (
                    pl.col("dollar_volume")
                    - pl.col("dollar_volume").rolling_mean(63, min_samples=63).over("symbol")
                )
                / (pl.col("dollar_volume").rolling_std(63, min_samples=63).over("symbol") + 1e-8)
            ).alias("dollar_volume_zscore_63"),
        )
    )


def add_technical_features(df: pl.DataFrame) -> pl.DataFrame:
    result = df
    for feature in [
        ROCFeature(periods=[21, 63, 126, 252]),
        RSIFeature(periods=[14, 63]),
        MACDFeature(fast=50, slow=200, signal=30),
        ATRFeature(periods=[14]),
        HistVolFeature(periods=[21, 63, 126]),
    ]:
        result = feature.fit_transform(result)
    return add_price_action_features(result)


def _zscore_expr(column: str) -> pl.Expr:
    return (
        (
            (pl.col(column) - pl.col(column).mean().over("date"))
            / (pl.col(column).std().over("date") + 1e-8)
        )
        .fill_nan(None)
        .fill_null(0.0)
    )


def add_factor_scores(
    df: pl.DataFrame, config: FeatureConfig
) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    result = df
    used: dict[str, list[str]] = {}

    tech_cols = [col for col in technical_columns(config) if col in result.columns]
    if tech_cols:
        result = result.with_columns(
            (sum(_zscore_expr(col) for col in tech_cols) / len(tech_cols)).alias(
                "technicalCompositeFactor"
            )
        )
        used["technicalCompositeFactor"] = tech_cols

    result, fundamental_sources = add_fundamental_factor_scores(result)
    used.update(fundamental_sources)

    hybrid_inputs = [
        col
        for col in ["technicalCompositeFactor", "fundamentalCompositeFactor"]
        if col in result.columns
    ]
    if hybrid_inputs:
        result = result.with_columns(
            (sum(pl.col(col) for col in hybrid_inputs) / len(hybrid_inputs)).alias(
                "hybridCompositeFactor"
            )
        )
        used["hybridCompositeFactor"] = hybrid_inputs

    return result, used


def _stage_row(
    name: str, frame: pl.DataFrame, previous: pl.DataFrame | None = None
) -> dict[str, Any]:
    symbols = frame.select("symbol").n_unique() if "symbol" in frame.columns and frame.height else 0
    previous_symbols = (
        previous.select("symbol").n_unique()
        if previous is not None and "symbol" in previous.columns and previous.height
        else None
    )
    return {
        "stage": name,
        "rows": frame.height,
        "symbols": symbols,
        "columns": len(frame.columns),
        "rows_delta": None if previous is None else frame.height - previous.height,
        "symbols_delta": None if previous_symbols is None else symbols - previous_symbols,
    }


def build_model_frame(
    raw_ohlcv: pl.DataFrame,
    config: FeatureConfig,
    *,
    return_diagnostics: bool = False,
) -> (
    tuple[pl.DataFrame, dict[str, list[str]]]
    | tuple[pl.DataFrame, dict[str, list[str]], pl.DataFrame]
):
    diagnostics: list[dict[str, Any]] = [_stage_row("raw", raw_ohlcv)]

    screened = screen_liquid_universe(raw_ohlcv, config)
    diagnostics.append(_stage_row("liquidity_screen", screened, raw_ohlcv))

    cleaned = preprocess_ohlcv(screened, min_trading_days=config.min_trading_days)
    diagnostics.append(_stage_row("preprocess_ohlcv", cleaned, screened))

    flagged = flag_anomalies(cleaned, max_gap_days=config.max_gap_days, min_notional_usd=None)
    diagnostics.append(_stage_row("flag_anomalies", flagged, cleaned))

    cleaned = remove_flagged_symbols(
        flagged,
        remove_large_gaps=config.remove_large_gaps,
        remove_low_volume=config.remove_low_volume,
    )
    diagnostics.append(_stage_row("remove_flagged_symbols", cleaned, flagged))

    featured = add_qsmom_features(cleaned, config)
    diagnostics.append(_stage_row("qsmom", featured, cleaned))

    featured = add_technical_features(featured)
    diagnostics.append(_stage_row("technical_features", featured))

    featured = VNFundamentalFeatures(termtype=config.fundamental_termtype).fit_transform(featured)
    diagnostics.append(_stage_row("fundamental_features", featured))

    featured, factor_sources = add_factor_scores(featured, config)
    diagnostics.append(_stage_row("factor_scores", featured))

    featured = ForwardReturns(periods=[config.forward_period]).fit_transform(featured)
    diagnostics.append(_stage_row("forward_returns", featured))

    result = featured.sort(["symbol", "date"])
    diagnostic_frame = pl.DataFrame(diagnostics)
    if return_diagnostics:
        return result, factor_sources, diagnostic_frame
    return result, factor_sources


def feature_coverage_report(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    rows = []
    total = max(df.height, 1)
    for column in columns:
        if column not in df.columns:
            rows.append(
                {
                    "column": column,
                    "present": False,
                    "dtype": None,
                    "non_null": 0,
                    "null_pct": 1.0,
                    "finite_pct": 0.0,
                    "n_unique": 0,
                }
            )
            continue
        series = df[column]
        non_null = total - series.null_count()
        finite = (
            df.select(pl.col(column).is_finite().sum()).item()
            if series.dtype.is_numeric()
            else non_null
        )
        rows.append(
            {
                "column": column,
                "present": True,
                "dtype": str(series.dtype),
                "non_null": int(non_null),
                "null_pct": float(series.null_count() / total),
                "finite_pct": float(finite / total),
                "n_unique": int(series.n_unique()),
            }
        )
    return pl.DataFrame(rows)


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


class EmptyStrategy(BaseStrategy):
    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        return self.empty_signal_frame()


def choose_predictors(
    df: pl.DataFrame,
    config: FeatureConfig,
    explicit_cols: list[str] | None = None,
) -> list[str]:
    candidates = explicit_cols or default_predictor_candidates(config)
    return available_predictors(df, candidates)


def signals_from_weights(trade_date: date, weights: dict[str, float]) -> pl.DataFrame:
    rows = [
        {
            "date": trade_date,
            "symbol": symbol,
            "signal": 1 if weight > 0 else -1,
            "weight": abs(float(weight)),
        }
        for symbol, weight in weights.items()
        if float(weight) != 0.0
    ]
    if not rows:
        return BaseStrategy.empty_signal_frame()
    return EmptyStrategy().validate_signal_frame(pl.DataFrame(rows))


def walk_forward_ml_signals(
    df: pl.DataFrame,
    experiment: ExperimentConfig,
    predictor_cols: list[str],
    target_col: str,
) -> pl.DataFrame:
    all_dates = sorted(df["date"].unique().to_list())
    rebalance = _rebalance_dates(all_dates, experiment.rebalance_period)
    date_index = {item: idx for idx, item in enumerate(all_dates)}
    signal_frames: list[pl.DataFrame] = []
    model_params = experiment.model_params or MODEL_PARAMS

    for rebalance_date in rebalance:
        idx = date_index[rebalance_date]
        if idx < max(experiment.feature.qsmom_slow, experiment.feature.forward_period, 80):
            continue
        train_start = all_dates[max(0, idx - experiment.train_window)]
        train = df.filter(
            (pl.col("date") >= train_start)
            & (pl.col("date") < rebalance_date)
            & pl.col(target_col).is_not_null()
        ).drop_nulls(predictor_cols + [target_col])
        predict = df.filter(pl.col("date") == rebalance_date).drop_nulls(predictor_cols)
        min_predict_rows = max(
            2, min(experiment.num_long_positions, df.select("symbol").n_unique())
        )
        if train.height < 100 or predict.height < min_predict_rows:
            continue

        scores = train_and_predict_xgb_classifier(
            train_data=train,
            predict_data=predict,
            predictor_cols=predictor_cols,
            target_col=target_col,
            model_params=model_params,
        )
        predictions = pd.Series(scores, index=predict["symbol"].to_list())
        weights = long_short_equal_weight_portfolio(
            predictions=predictions,
            num_long_positions=experiment.num_long_positions,
            num_short_positions=experiment.num_short_positions,
            long_threshold=effective_long_threshold(experiment.long_threshold),
            short_threshold=experiment.short_threshold,
            history_df=train.to_pandas(),
        )
        signals = signals_from_weights(rebalance_date, weights)
        if not signals.is_empty():
            signal_frames.append(signals)

    if not signal_frames:
        return BaseStrategy.empty_signal_frame()
    return pl.concat(signal_frames, how="vertical").sort(["date", "symbol"])


def run_ml_factor_experiment(
    raw_ohlcv: pl.DataFrame, experiment: ExperimentConfig
) -> dict[str, Any]:
    model_frame, factor_source_map, diagnostics = build_model_frame(
        raw_ohlcv,
        experiment.feature,
        return_diagnostics=True,
    )
    target_col = f"forward_return_{experiment.feature.forward_period}"
    predictor_cols = choose_predictors(model_frame, experiment.feature, experiment.predictor_cols)
    if not predictor_cols:
        raise RuntimeError(f"{experiment.name}: no predictor columns are available")
    signals = walk_forward_ml_signals(model_frame, experiment, predictor_cols, target_col)
    if signals.is_empty():
        raise RuntimeError(f"{experiment.name}: no signals generated")

    bt_config = BacktestConfig(
        workflow="research",
        asset_types=["vn_stock"],
        universe=UniverseConfig(vn_stock=sorted(model_frame["symbol"].unique().to_list())),
        start_date=model_frame["date"].min(),
        end_date=model_frame["date"].max(),
        initial_capital=experiment.initial_capital,
        backtest_engine="notebook",
        rebalance_frequency=experiment.rebalance_period,
        commission=CommissionConfig(model="percentage", rate=Decimal("0.0015")),
        calendar="XHOSE",
    )
    result = run_backtest_frame(
        engine_name="notebook_walk_forward",
        strategy=EmptyStrategy(),
        data=model_frame,
        config=bt_config,
        prebuilt_signals=signals,
    )
    return {
        "result": result,
        "model_frame": model_frame,
        "feature_diagnostics": diagnostics,
        "signals": signals,
        "predictor_cols": predictor_cols,
        "target_col": target_col,
        "factor_sources": factor_source_map,
    }


def strategy_params(strategy_config: dict[str, Any]) -> dict[str, Any]:
    strategy = strategy_config.get("strategy", {})
    if not isinstance(strategy, dict):
        return {}
    params = strategy.get("params", {})
    return dict(params) if isinstance(params, dict) else {}


def target_period_from_config(strategy_config: dict[str, Any]) -> int | None:
    target_col = strategy_params(strategy_config).get("target_col")
    if not isinstance(target_col, str) or not target_col.startswith("forward_return_"):
        return None
    try:
        return int(target_col.rsplit("_", 1)[1])
    except ValueError:
        return None


def configured_rebalance_period(strategy_config: dict[str, Any], override: int | None) -> int:
    if override is not None:
        return positive_int(override, "rebalance_period")
    params = strategy_params(strategy_config)
    value = (
        params.get("rebalance_period")
        or params.get("rebalance_frequency")
        or strategy_config.get("rebalance_frequency")
        or 10
    )
    return positive_int(value, "rebalance_period")


def configured_model_params(
    strategy_config: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    params = deepcopy(MODEL_PARAMS)
    trainer = strategy_params(strategy_config).get("trainer", {})
    if isinstance(trainer, dict):
        trainer_params = trainer.get("params", {})
        if isinstance(trainer_params, dict) and isinstance(
            trainer_params.get("model_params"), dict
        ):
            params.update(trainer_params["model_params"])
    if args.model_n_estimators is not None:
        params["n_estimators"] = args.model_n_estimators
    if args.model_max_depth is not None:
        params["max_depth"] = args.model_max_depth
    return params


def configured_portfolio_params(strategy_config: dict[str, Any]) -> dict[str, Any]:
    portfolio = strategy_params(strategy_config).get("portfolio", {})
    if not isinstance(portfolio, dict):
        return {}
    params = portfolio.get("params", {})
    return dict(params) if isinstance(params, dict) else {}


def build_feature_config(
    strategy_config: dict[str, Any], args: argparse.Namespace
) -> FeatureConfig:
    forward_period = args.forward_period or target_period_from_config(strategy_config) or 21
    return FeatureConfig(
        min_trading_days=args.min_trading_days,
        volume_top_n=args.volume_top_n,
        min_avg_volume=args.min_avg_volume,
        remove_large_gaps=args.remove_large_gaps,
        max_gap_days=args.max_gap_days,
        remove_low_volume=args.remove_low_volume,
        qsmom_fast=args.qsmom_fast,
        qsmom_slow=args.qsmom_slow,
        qsmom_returns=args.qsmom_returns,
        forward_period=forward_period,
        fundamental_termtype=args.fundamental_termtype,
    )


def build_baseline_experiment(
    strategy_config: dict[str, Any],
    args: argparse.Namespace,
    feature: FeatureConfig,
) -> ExperimentConfig:
    portfolio_params = configured_portfolio_params(strategy_config)
    predictor_cols = parse_csv(args.predictor_cols)
    if predictor_cols is None:
        configured_predictors = strategy_params(strategy_config).get("predictor_cols")
        if isinstance(configured_predictors, list):
            predictor_cols = [str(item) for item in configured_predictors]
    return ExperimentConfig(
        name=args.run_name,
        feature=feature,
        predictor_cols=predictor_cols,
        train_window=args.train_window or int(strategy_config.get("train_window") or 504),
        rebalance_period=configured_rebalance_period(strategy_config, args.rebalance_period),
        num_long_positions=args.num_long_positions
        or int(portfolio_params.get("num_long_positions") or 10),
        num_short_positions=args.num_short_positions
        if args.num_short_positions is not None
        else int(portfolio_params.get("num_short_positions") or 0),
        long_threshold=args.long_threshold
        if args.long_threshold is not None
        else portfolio_params.get("long_threshold"),
        short_threshold=args.short_threshold
        if args.short_threshold is not None
        else portfolio_params.get("short_threshold"),
        model_params=configured_model_params(strategy_config, args),
        initial_capital=Decimal(
            str(args.initial_capital or strategy_config.get("initial_capital") or "1000000000")
        ),
    )


def make_sweep_arms(base: ExperimentConfig) -> list[ExperimentConfig]:
    arms: list[ExperimentConfig] = []
    for top_n, fast, slow, num_long, depth, n_estimators in product(
        [50, 80],
        [21],
        [126, 252],
        [8, 12],
        [2, 3],
        [80],
    ):
        feature = FeatureConfig(
            min_trading_days=base.feature.min_trading_days,
            volume_top_n=top_n,
            min_avg_volume=base.feature.min_avg_volume,
            remove_large_gaps=base.feature.remove_large_gaps,
            max_gap_days=base.feature.max_gap_days,
            remove_low_volume=base.feature.remove_low_volume,
            qsmom_fast=fast,
            qsmom_slow=slow,
            qsmom_returns=base.feature.qsmom_returns,
            forward_period=base.feature.forward_period,
            fundamental_termtype=base.feature.fundamental_termtype,
        )
        model_params = {
            **(base.model_params or MODEL_PARAMS),
            "max_depth": depth,
            "n_estimators": n_estimators,
        }
        arms.append(
            ExperimentConfig(
                name=f"vn100_top{top_n}_qsmom{fast}_{slow}_long{num_long}_depth{depth}",
                feature=feature,
                predictor_cols=base.predictor_cols,
                train_window=base.train_window,
                rebalance_period=base.rebalance_period,
                num_long_positions=num_long,
                num_short_positions=base.num_short_positions,
                long_threshold=base.long_threshold,
                short_threshold=base.short_threshold,
                model_params=model_params,
                initial_capital=base.initial_capital,
            )
        )
    return arms


def mlflow_artifact_location(tracking_uri: str, runtime_root: Path) -> str | None:
    scheme = urlparse(tracking_uri).scheme
    if scheme in {"", "file", "sqlite"}:
        return (runtime_root / "mlartifacts").as_uri()
    return None


def setup_mlflow(
    enabled: bool, tracking_uri: str, experiment_name: str, runtime_root: Path
) -> bool:
    if not enabled:
        print("MLflow logging disabled.")
        return False
    if not MLFLOW_AVAILABLE or mlflow is None:
        print("MLflow is not installed. Skipping tracking.")
        return False
    try:
        mlflow.set_tracking_uri(tracking_uri)
        if mlflow.get_experiment_by_name(experiment_name) is None:
            artifact_location = mlflow_artifact_location(tracking_uri, runtime_root)
            if artifact_location is None:
                mlflow.create_experiment(experiment_name)
            else:
                mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
        mlflow.set_experiment(experiment_name)
    except Exception as exc:
        print(f"MLflow setup failed for {tracking_uri}; continuing without tracking: {exc}")
        print("Start the shared server with: docker compose -f docker/compose.yaml up -d mlflow")
        return False
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment: {experiment_name}")
    return True


def flatten_params(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_params(value, full_key))
        else:
            flattened[full_key] = value
    return flattened


def mlflow_param_value(value: Any) -> str | int | float | bool | None:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return repr(value)


def write_result_artifacts(run_dir: Path, payload: dict[str, Any]) -> list[Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    result: BacktestResult = payload["result"]
    csv_artifacts = {
        "returns.csv": result.returns.to_pandas(),
        "equity_curve.csv": result.equity_curve.to_pandas(),
        "signals.csv": payload["signals"].to_pandas(),
        "predictor_cols.csv": pd.DataFrame({"predictor_col": payload["predictor_cols"]}),
        "feature_diagnostics.csv": payload["feature_diagnostics"].to_pandas(),
        "factor_sources.csv": pd.DataFrame(
            [
                {"factor": factor, "source_columns": ",".join(columns)}
                for factor, columns in payload["factor_sources"].items()
            ]
        ),
    }
    paths = []
    for filename, frame in csv_artifacts.items():
        path = run_dir / filename
        frame.to_csv(path, index=False)
        paths.append(path)
    model_frame_path = run_dir / "model_frame.parquet"
    payload["model_frame"].write_parquet(model_frame_path)
    paths.append(model_frame_path)
    return paths


def log_to_mlflow(
    enabled: bool,
    experiment: ExperimentConfig,
    payload: dict[str, Any],
    runtime_root: Path,
    extra_params: dict[str, Any],
) -> None:
    if not enabled or mlflow is None:
        return
    result: BacktestResult = payload["result"]
    params = {
        "name": experiment.name,
        "train_window": experiment.train_window,
        "rebalance_period": experiment.rebalance_period,
        "num_long_positions": experiment.num_long_positions,
        "num_short_positions": experiment.num_short_positions,
        "long_threshold": experiment.long_threshold,
        "short_threshold": experiment.short_threshold,
        "predictor_cols": payload["predictor_cols"],
        "target_col": payload["target_col"],
        "feature": asdict(experiment.feature),
        "model_params": experiment.model_params or MODEL_PARAMS,
        **extra_params,
    }
    artifact_dir = runtime_root / "mlflow_artifacts" / experiment.name
    artifacts = write_result_artifacts(artifact_dir, payload)
    flat_params = {key: mlflow_param_value(value) for key, value in flatten_params(params).items()}
    try:
        with mlflow.start_run(
            run_name=experiment.name, tags={"asset_type": "vn_stock", "universe": "VN100"}
        ):
            mlflow.log_params(flat_params)
            mlflow.log_metrics({key: float(value) for key, value in result.metrics.items()})
            mlflow.log_metric("signal_rows", float(payload["signals"].height))
            mlflow.log_metric("featured_rows", float(payload["model_frame"].height))
            for artifact in artifacts:
                mlflow.log_artifact(str(artifact))
    except Exception as exc:
        print(f"MLflow logging failed for {experiment.name}; continuing: {exc}")


def result_row(
    experiment: ExperimentConfig, payload: dict[str, Any], status: str = "ok"
) -> dict[str, Any]:
    result: BacktestResult = payload["result"]
    return {
        "name": experiment.name,
        **result.metrics,
        "signal_rows": payload["signals"].height,
        "featured_rows": payload["model_frame"].height,
        "predictor_count": len(payload["predictor_cols"]),
        "rebalance_period": experiment.rebalance_period,
        "train_window": experiment.train_window,
        "status": status,
    }


def serializable(value: Any) -> Any:
    if is_dataclass(value):
        return serializable(asdict(value))
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [serializable(item) for item in value]
    if isinstance(value, date | datetime | Decimal | Path):
        return str(value)
    return value


def write_run_config(run_dir: Path, args: argparse.Namespace, experiment: ExperimentConfig) -> Path:
    path = run_dir / "run_config.yaml"
    payload = {
        "args": vars(args),
        "experiment": experiment,
        "qts_root": os.environ.get("QTS_ROOT"),
    }
    path.write_text(yaml.safe_dump(serializable(payload), sort_keys=True))
    return path


def summarize_ohlcv(ohlcv: pl.DataFrame) -> pl.DataFrame:
    return (
        ohlcv.group_by("symbol")
        .agg(
            pl.col("date").min().alias("start"),
            pl.col("date").max().alias("end"),
            pl.len().alias("rows"),
            pl.col("close").last().alias("last_close"),
            pl.col("volume").mean().alias("avg_volume"),
        )
        .sort("rows", descending=True)
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    env_strategy_config = os.environ.get("QTS_NOTEBOOK_STRATEGY_CONFIG")
    parser.add_argument("--asset-config", type=Path, default=DEFAULT_ASSET_CONFIG)
    parser.add_argument(
        "--strategy-config",
        type=Path,
        default=Path(env_strategy_config) if env_strategy_config else DEFAULT_STRATEGY_CONFIG,
        help=f"Optional ML factor config, for example {DEFAULT_STRATEGY_CONFIG}",
    )
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name", default="baseline_qsmom_fundamental_xgb")
    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=env_date("QTS_NOTEBOOK_START_DATE", date(2021, 1, 1)),
    )
    parser.add_argument(
        "--end-date", type=parse_date, default=env_date("QTS_NOTEBOOK_END_DATE", date.today())
    )
    parser.add_argument("--interval", default=os.environ.get("QTS_NOTEBOOK_INTERVAL", "1d"))
    parser.add_argument("--max-symbols", type=int, default=env_int("QTS_NOTEBOOK_MAX_SYMBOLS"))
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument(
        "--fundamental-termtype", type=int, default=env_int("QTS_NOTEBOOK_FUNDAMENTAL_TERMTYPE", 1)
    )
    parser.add_argument(
        "--fundamental-pages", type=int, default=env_int("QTS_NOTEBOOK_FUNDAMENTAL_PAGES", 3)
    )
    parser.add_argument(
        "--fetch-fundamentals",
        action=argparse.BooleanOptionalAction,
        default=env_bool("QTS_NOTEBOOK_FETCH_FUNDAMENTALS", True),
    )
    parser.add_argument(
        "--force-refresh-fundamentals",
        action=argparse.BooleanOptionalAction,
        default=env_bool("QTS_NOTEBOOK_FORCE_REFRESH_FUNDAMENTALS", False),
    )
    parser.add_argument(
        "--run-sweeps",
        action=argparse.BooleanOptionalAction,
        default=env_bool("QTS_NOTEBOOK_RUN_SWEEPS", False),
    )
    parser.add_argument(
        "--mlflow",
        action=argparse.BooleanOptionalAction,
        default=not env_bool("QTS_NOTEBOOK_DISABLE_MLFLOW", False),
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI),
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_MLFLOW_EXPERIMENT_NAME),
    )
    parser.add_argument("--train-window", type=int)
    parser.add_argument("--rebalance-period", type=int)
    parser.add_argument("--predictor-cols")
    parser.add_argument("--num-long-positions", type=int)
    parser.add_argument("--num-short-positions", type=int)
    parser.add_argument("--long-threshold", type=float)
    parser.add_argument("--short-threshold", type=float)
    parser.add_argument("--initial-capital")
    parser.add_argument("--min-trading-days", type=int, default=252)
    parser.add_argument("--volume-top-n", type=int, default=80)
    parser.add_argument("--min-avg-volume", type=float, default=50_000)
    parser.add_argument("--remove-large-gaps", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-gap-days", type=int, default=21)
    parser.add_argument("--remove-low-volume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qsmom-fast", type=int, default=21)
    parser.add_argument("--qsmom-slow", type=int, default=252)
    parser.add_argument("--qsmom-returns", type=int, default=126)
    parser.add_argument("--forward-period", type=int)
    parser.add_argument("--model-n-estimators", type=int)
    parser.add_argument("--model-max-depth", type=int)
    args = parser.parse_args(argv)
    if args.volume_top_n is not None and args.volume_top_n <= 0:
        args.volume_top_n = None
    return args


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runtime_root = args.runtime_root.resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    os.environ["QTS_ROOT"] = str(runtime_root)

    run_dir = args.output_dir or runtime_root / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    strategy_config = load_yaml_mapping(args.strategy_config) if args.strategy_config else {}
    feature_config = build_feature_config(strategy_config, args)
    baseline = build_baseline_experiment(strategy_config, args, feature_config)
    write_run_config(run_dir, args, baseline)

    symbols, benchmark_symbol = load_vn100_symbols(args.asset_config, args.max_symbols)
    request_symbols = symbols + [benchmark_symbol]

    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"QTS_ROOT: {os.environ['QTS_ROOT']}")
    print(f"Run directory: {run_dir}")
    print(f"VN100 equity symbols: {len(symbols)}")
    print(f"Benchmark symbol: {benchmark_symbol}")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Rebalance period: {baseline.rebalance_period}")

    ohlcv, benchmark_ohlcv, fetch_failures = run_async(
        fetch_prices_and_fundamentals(
            request_symbols,
            benchmark_symbol,
            runtime_root,
            start_date=args.start_date,
            end_date=args.end_date,
            interval=args.interval,
            batch_size=args.batch_size,
            fetch_fundamentals=args.fetch_fundamentals,
            fundamental_termtype=args.fundamental_termtype,
            fundamental_pages=args.fundamental_pages,
            force_refresh_fundamentals=args.force_refresh_fundamentals,
        )
    )

    if ohlcv.is_empty():
        raise RuntimeError(
            "No OHLCV rows were fetched. Check network access or reduce the universe."
        )

    summarize_ohlcv(ohlcv).to_pandas().to_csv(run_dir / "ohlcv_summary.csv", index=False)
    benchmark_ohlcv.write_parquet(run_dir / "benchmark_ohlcv.parquet")
    pd.DataFrame(fetch_failures, columns=["symbol", "error"]).to_csv(
        run_dir / "fetch_failures.csv",
        index=False,
    )
    fundamental_cache_report(symbols, termtype=args.fundamental_termtype).to_pandas().to_csv(
        run_dir / "fundamental_cache.csv",
        index=False,
    )

    print(f"Fetched OHLCV rows: {ohlcv.height:,}")
    print(f"Fetched symbols: {ohlcv.select('symbol').n_unique():,}")
    print(f"Benchmark rows: {benchmark_ohlcv.height:,}")
    print(f"Fetch failures: {len(fetch_failures)}")

    mlflow_enabled = setup_mlflow(
        args.mlflow,
        args.mlflow_tracking_uri,
        args.mlflow_experiment_name,
        runtime_root,
    )

    rows: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}

    print(f"Running baseline: {baseline.name}")
    baseline_payload = run_ml_factor_experiment(ohlcv, baseline)
    write_result_artifacts(run_dir / baseline.name, baseline_payload)
    log_to_mlflow(
        mlflow_enabled, baseline, baseline_payload, runtime_root, {"run_type": "baseline"}
    )
    rows.append(result_row(baseline, baseline_payload))
    payloads[baseline.name] = baseline_payload

    if args.run_sweeps:
        sweep_arms = make_sweep_arms(baseline)
        print(f"Prepared sweep arms: {len(sweep_arms)}")
        for experiment in sweep_arms:
            print(f"Running sweep arm: {experiment.name}")
            try:
                payload = run_ml_factor_experiment(ohlcv, experiment)
                write_result_artifacts(run_dir / experiment.name, payload)
                log_to_mlflow(
                    mlflow_enabled, experiment, payload, runtime_root, {"run_type": "sweep"}
                )
                rows.append(result_row(experiment, payload))
                payloads[experiment.name] = payload
            except Exception as exc:
                rows.append({"name": experiment.name, "status": "failed", "error": str(exc)})
                print(f"  failed: {exc}")

    results_df = pd.DataFrame(rows)
    results_path = run_dir / "results_summary.csv"
    results_df.to_csv(results_path, index=False)

    ok_results = (
        results_df[results_df["status"] == "ok"] if "status" in results_df else pd.DataFrame()
    )
    if not ok_results.empty and "sharpe" in ok_results:
        best = ok_results.sort_values("sharpe", ascending=False).iloc[0]
        print(f"Best run by Sharpe: {best['name']} ({best['sharpe']:.4f})")
    baseline_result: BacktestResult = baseline_payload["result"]
    print(f"Baseline metrics: {baseline_result.metrics}")
    print(f"Results summary: {results_path}")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
