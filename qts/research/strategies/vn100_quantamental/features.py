"""Feature engineering pipeline for the VN100 quantamental strategy."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import polars as pl

from qts.research.features.forward_returns import ForwardReturns
from qts.research.features.fundamentals import (
    FUNDAMENTAL_FACTOR_GROUPS,
    VNFundamentalFeatures,
)
from qts.research.features.fundamentals import add_factor_scores as add_fundamental_factor_scores
from qts.research.features.indicators.momentum import ROCFeature, RSIFeature
from qts.research.features.indicators.trend import MACDFeature
from qts.research.features.indicators.volatility import ATRFeature, HistVolFeature
from qts.research.features.preprocessor import (
    flag_anomalies,
    preprocess_ohlcv,
    remove_flagged_symbols,
)

from .config import TECHNICAL_BASE_COLUMNS, FeatureConfig, qsmom_column


def technical_columns(config: FeatureConfig) -> list[str]:
    return [qsmom_column(config), *TECHNICAL_BASE_COLUMNS]


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
            ((pl.col("close") / pl.col("close").shift(1).over("symbol")) - 1).alias("_daily_return"),
            (
                (pl.col("close").shift(fast).over("symbol") / pl.col("close").shift(slow).over("symbol")) - 1
            ).alias("_older_momentum"),
            ((pl.col("close") / pl.col("close").shift(fast).over("symbol")) - 1).alias("_recent_momentum"),
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


__all__ = [
    "FUNDAMENTAL_FACTOR_GROUPS",
    "add_factor_scores",
    "add_qsmom_features",
    "add_technical_features",
    "build_model_frame",
    "feature_coverage_report",
    "screen_liquid_universe",
    "technical_columns",
]
