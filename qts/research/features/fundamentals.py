"""Fundamental feature joins."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from qts.core.registry import Registry
from qts.research.features.base import BaseFeature
from qts.utils.paths import cache_dir


def _fund_cache_path(ticker: str, termtype: int) -> Path:
    label = "annual" if termtype == 1 else "quarterly"
    return cache_dir() / "vn_fundamentals" / f"{ticker}_{label}.parquet"


def _safe_col(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "pct")
        .replace("-", "_")
    )


@Registry.register_feature("fundamental")
class FundamentalFeatures(BaseFeature):
    """Stock-only fundamental features (FMP wide format)."""

    def __init__(self, fundamentals: pl.DataFrame | None = None) -> None:
        self.fundamentals = fundamentals if fundamentals is not None else pl.DataFrame()

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        original = df
        if self.fundamentals.is_empty():
            return df
        required = {"symbol", "pe_ratio", "ev_ebitda"}
        if not required.issubset(self.fundamentals.columns):
            return df
        df_syms = set(df["symbol"].unique().to_list())
        fund_syms = set(self.fundamentals["symbol"].unique().to_list())
        if not df_syms & fund_syms:
            return df
        transformed = df.join(self.fundamentals.select(sorted(required)), on="symbol", how="left")
        return self._validate_append_only(original, transformed)


@Registry.register_feature("vn_fundamental")
class VNFundamentalFeatures(BaseFeature):
    """VN stock fundamental ratio features from vnstock/KBS parquet cache.

    Reads ~/.qts/cache/vn_fundamentals/{ticker}_{annual|quarterly}.parquet,
    filters to CSTC report type (financial ratios), pivots item_en → columns,
    then as-of joins backward on report_date against the price panel to prevent
    look-ahead bias.
    """

    def __init__(self, termtype: int = 1) -> None:
        self.termtype = termtype  # 1=annual, 2=quarterly

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        original = df
        fund_frames: list[pl.DataFrame] = []
        for symbol in df["symbol"].unique().to_list():
            if not (symbol.startswith("VN:") or symbol.startswith("VNW:")):
                continue
            ticker = symbol.split(":")[1]
            path = _fund_cache_path(ticker, self.termtype)
            if path.exists():
                # Overwrite the raw-ticker symbol column with the full QTS symbol.
                fund_frames.append(
                    pl.read_parquet(path).with_columns(pl.lit(symbol).alias("symbol"))
                )

        if not fund_frames:
            return df

        fund = pl.concat(fund_frames, how="vertical").filter(pl.col("report_type") == "CSTC")
        if fund.is_empty():
            return df

        wide = (
            fund.pivot(
                values="value",
                index=["symbol", "report_date"],
                on="item_en",
                aggregate_function="first",
            )
            .rename({c: _safe_col(c) for c in fund["item_en"].unique().to_list()})
            .sort(["symbol", "report_date"])
        )

        joined = (
            df.select(["date", "symbol"])
            .sort(["symbol", "date"])
            .join_asof(
                wide,
                left_on="date",
                right_on="report_date",
                by="symbol",
                strategy="backward",
            )
        )

        ratio_cols = [c for c in joined.columns if c not in {"date", "symbol"}]
        if not ratio_cols:
            return df

        transformed = df.join(
            joined.select(["date", "symbol", *ratio_cols]),
            on=["date", "symbol"],
            how="left",
        )
        return self._validate_append_only(original, transformed)
