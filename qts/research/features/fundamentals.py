"""Fundamental feature normalization and joins."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from qts.core.registry import Registry
from qts.research.features.base import BaseFeature
from qts.utils.paths import cache_dir

CANONICAL_FUNDAMENTAL_COLUMNS = [
    "reportDate",
    "trailingEPS",
    "bookValuePerShare",
    "cashFlowPerShare",
    "priceToEarningsRatio",
    "priceToBookRatio",
    "priceToSalesRatio",
    "dividendYield",
    "beta",
    "enterpriseValueOverEBIT",
    "enterpriseValueOverEBITDA",
    "grossProfitMargin",
    "ebitMargin",
    "ebitdaMargin",
    "netProfitMargin",
    "returnOnEquity",
    "returnOnAssets",
    "returnOnCapitalEmployed",
    "cashReturnOnAssets",
    "cashReturnOnEquity",
    "cashRatio",
    "quickRatio",
    "currentRatio",
    "interestCoverage",
    "debtToAssetsRatio",
    "debtToEquityRatio",
    "liabilitiesToAssetsRatio",
    "liabilitiesToEquityRatio",
    "shortTermLiabilitiesToEquityRatio",
    "debtCoverage",
    "netCashFlowsToShortTermLiabilities",
    "accrualRatioBalanceSheet",
    "accrualRatioCashFlow",
    "accrualRatioCF",
    "netInterestMargin",
    "costIncomeRatio",
    "loanLossProvisionRatio",
    "revenue",
    "netRevenue",
    "grossProfit",
    "profitBeforeTax",
    "netIncome",
    "netIncomeParent",
    "eps",
    "totalAssets",
    "totalLiabilities",
    "totalEquity",
    "cashAndCashEquivalents",
    "shortTermAssets",
    "longTermAssets",
    "shortTermLiabilities",
    "longTermLiabilities",
    "inventory",
    "operatingCashFlow",
    "investingCashFlow",
    "financingCashFlow",
    "netCashFlow",
    "cashEndPeriod",
    "totalLiabilitiesToAssets",
    "cashAndCashEquivalentsToAssets",
    "growthRevenue",
    "growthNetRevenue",
    "growthGrossProfit",
    "growthNetIncome",
    "growthNetIncomeParent",
    "growthEPS",
    "growthTrailingEPS",
    "growthOperatingCashFlow",
    "growthTotalAssets",
    "growthBookValuePerShare",
]

FUNDAMENTAL_FACTOR_GROUPS: dict[str, dict[str, Any]] = {
    "qualityFactor": {
        "columns": [
            "returnOnEquity",
            "returnOnAssets",
            "returnOnCapitalEmployed",
            "grossProfitMargin",
            "ebitMargin",
            "netProfitMargin",
            "cashReturnOnEquity",
            "cashReturnOnAssets",
            "interestCoverage",
        ],
        "sign": 1.0,
    },
    "valuationFactor": {
        "columns": [
            "priceToEarningsRatio",
            "priceToBookRatio",
            "priceToSalesRatio",
            "enterpriseValueOverEBIT",
            "enterpriseValueOverEBITDA",
        ],
        "sign": -1.0,
    },
    "yieldFactor": {
        "columns": ["dividendYield"],
        "sign": 1.0,
    },
    "growthFactor": {
        "columns": [
            "growthRevenue",
            "growthNetRevenue",
            "growthGrossProfit",
            "growthNetIncome",
            "growthNetIncomeParent",
            "growthEPS",
            "growthTrailingEPS",
            "growthOperatingCashFlow",
            "growthTotalAssets",
            "growthBookValuePerShare",
        ],
        "sign": 1.0,
    },
    "leverageFactor": {
        "columns": [
            "debtToAssetsRatio",
            "debtToEquityRatio",
            "liabilitiesToAssetsRatio",
            "liabilitiesToEquityRatio",
            "shortTermLiabilitiesToEquityRatio",
            "totalLiabilitiesToAssets",
        ],
        "sign": -1.0,
    },
    "liquidityFactor": {
        "columns": [
            "cashRatio",
            "quickRatio",
            "currentRatio",
            "cashAndCashEquivalentsToAssets",
        ],
        "sign": 1.0,
    },
    "cashflowFactor": {
        "columns": [
            "cashFlowPerShare",
            "operatingCashFlow",
            "netCashFlow",
            "netCashFlowsToShortTermLiabilities",
        ],
        "sign": 1.0,
    },
    "accrualQualityFactor": {
        "columns": [
            "accrualRatioBalanceSheet",
            "accrualRatioCashFlow",
            "accrualRatioCF",
        ],
        "sign": -1.0,
    },
}

FMP_ALIASES = {
    "date": "reportDate",
    "calendarYear": "fiscalYear",
    "pe": "priceToEarningsRatio",
    "p_e": "priceToEarningsRatio",
    "peRatio": "priceToEarningsRatio",
    "pe_ratio": "priceToEarningsRatio",
    "pb": "priceToBookRatio",
    "p_b": "priceToBookRatio",
    "pbRatio": "priceToBookRatio",
    "pb_ratio": "priceToBookRatio",
    "ps": "priceToSalesRatio",
    "p_s": "priceToSalesRatio",
    "psRatio": "priceToSalesRatio",
    "ev_ebit": "enterpriseValueOverEBIT",
    "ev_ebitda": "enterpriseValueOverEBITDA",
    "roe": "returnOnEquity",
    "roa": "returnOnAssets",
    "net_margin": "netProfitMargin",
    "gross_margin": "grossProfitMargin",
    "debt_to_equity": "debtToEquityRatio",
    "debt_equity_ratio": "debtToEquityRatio",
    "eps_growth": "growthEPS",
    "revenue_growth": "growthRevenue",
}

KBS_FUNDAMENTAL_ITEMS = {
    "CSTC": {
        "trailingEPS": ["Trailing EPS"],
        "bookValuePerShare": ["Book value per share (BVPS)"],
        "cashFlowPerShare": ["Cash flow per share (CPS)"],
        "priceToEarningsRatio": ["P/E"],
        "priceToBookRatio": ["P/B"],
        "priceToSalesRatio": ["P/S"],
        "dividendYield": ["Dividend yield"],
        "beta": ["Beta"],
        "enterpriseValueOverEBIT": ["EV/EBIT"],
        "enterpriseValueOverEBITDA": ["EV/EBITDA"],
        "grossProfitMargin": ["Gross profit margin"],
        "ebitMargin": ["EBIT margin"],
        "ebitdaMargin": ["EBITDA/Net revenue"],
        "netProfitMargin": ["Net profit margin"],
        "returnOnEquity": ["ROE"],
        "returnOnAssets": ["ROA"],
        "returnOnCapitalEmployed": ["Return on capital employed (ROCE)"],
        "cashReturnOnAssets": ["Cash return to assets"],
        "cashReturnOnEquity": ["Cash return on equity"],
        "cashRatio": ["Cash ratio"],
        "quickRatio": ["Quick ratio"],
        "currentRatio": ["Short-term ratio"],
        "interestCoverage": ["Interest coverage"],
        "debtToAssetsRatio": ["Debt to assets"],
        "debtToEquityRatio": ["Debt to equity"],
        "liabilitiesToAssetsRatio": ["Liabilities to assets"],
        "liabilitiesToEquityRatio": ["Liabilities to equity"],
        "shortTermLiabilitiesToEquityRatio": ["Short-term liabilities to equity"],
        "debtCoverage": ["Debt coverage"],
        "netCashFlowsToShortTermLiabilities": ["Net cash flows/Short -term liabilities"],
        "accrualRatioBalanceSheet": ["Accrual ratio (Balance sheet method)"],
        "accrualRatioCashFlow": ["Accrual ratio (Cash flow method)"],
        "accrualRatioCF": ["Accrual ratio CF"],
        "netInterestMargin": ["Net interest margin (NIM)"],
        "costIncomeRatio": ["Cost Income Ratio (CIR)"],
        "loanLossProvisionRatio": ["Loan loss provision ratio"],
    },
    "KQKD": {
        "revenue": ["1. Revenue", "Revenue from securities business (01->11)"],
        "netRevenue": ["3. Net revenue", "Net sales", "I. Net interest income"],
        "grossProfit": ["5. Gross profit", "Gross profit"],
        "profitBeforeTax": [
            "15. Profit before tax",
            "XI. Profit before tax",
            "IX. Profit before tax",
        ],
        "netIncome": [
            "18. Net profit after tax",
            "XIII. Net profit after tax",
            "XI.  Net profit after tax",
        ],
        "netIncomeParent": [
            "Profit after tax for shareholders of parent company",
            "XV. Net profit atttributable to the equity holders of the Bank",
            "11.1. Profit after tax for shareholders of the parents company",
        ],
        "eps": [
            "19. Earnings per share (VND)",
            "Earning per share (VND)",
            "13.1. Earning per share (VND)",
        ],
    },
    "CDKT": {
        "totalAssets": ["TOTAL ASSETS"],
        "totalLiabilities": ["C. LIABILITIES", "TOTAL LIABILITIES"],
        "totalEquity": ["I. Owner's equity", "D. OWNER'S EQUITY", "VIII. Capital and Reserves"],
        "cashAndCashEquivalents": ["I. Cash and cash equivalents", "1. Cash"],
        "shortTermAssets": ["A. SHORT-TERM ASSETS"],
        "longTermAssets": ["B. LONG-TERM ASSETS"],
        "shortTermLiabilities": ["I. Short-term liabilities"],
        "longTermLiabilities": ["II. Long-term liabilities"],
        "inventory": ["IV. Inventories", "1. Inventories"],
    },
    "LCTT": {
        "operatingCashFlow": ["Net cash flows from operating activities"],
        "investingCashFlow": ["Net cash flows from investing activities"],
        "financingCashFlow": ["Net cash flows from financing activities"],
        "netCashFlow": ["Net cash flows during the period", "IV. Net cash flows during the period"],
        "cashEndPeriod": [
            "Cash and cash equivalents at end of the period",
            "Cash and cash equivalents at end of period",
        ],
    },
}

GROWTH_SOURCE_COLUMNS = {
    "revenue": "growthRevenue",
    "netRevenue": "growthNetRevenue",
    "grossProfit": "growthGrossProfit",
    "netIncome": "growthNetIncome",
    "netIncomeParent": "growthNetIncomeParent",
    "eps": "growthEPS",
    "trailingEPS": "growthTrailingEPS",
    "operatingCashFlow": "growthOperatingCashFlow",
    "totalAssets": "growthTotalAssets",
    "bookValuePerShare": "growthBookValuePerShare",
}


def _fund_cache_path(ticker: str, termtype: int) -> Path:
    label = "annual" if termtype == 1 else "quarterly"
    return cache_dir() / "vn_fundamentals" / f"{ticker}_{label}.parquet"


def _canonical_item_map() -> pl.DataFrame:
    rows = []
    for report_type, feature_map in KBS_FUNDAMENTAL_ITEMS.items():
        for column, item_names in feature_map.items():
            for item_name in item_names:
                rows.append({"report_type": report_type, "item_en": item_name, "column": column})
    return pl.DataFrame(rows)


def _add_if_missing(frame: pl.DataFrame, target: str, source: str) -> pl.DataFrame:
    if target in frame.columns or source not in frame.columns:
        return frame
    return frame.with_columns(pl.col(source).alias(target))


def normalize_fmp_fundamentals(fundamentals: pl.DataFrame) -> pl.DataFrame:
    """Return fundamentals with FMP-like canonical camelCase columns appended."""

    if fundamentals.is_empty():
        return fundamentals
    renamed = fundamentals
    for source, target in FMP_ALIASES.items():
        renamed = _add_if_missing(renamed, target, source)
    if "reportDate" in renamed.columns:
        renamed = renamed.with_columns(pl.col("reportDate").cast(pl.Date, strict=False))
    return renamed


def load_vn_fundamental_cache(symbols: list[str], termtype: int = 1) -> pl.DataFrame:
    """Load cached KBS/VN fundamentals for QTS VN symbols."""

    frames: list[pl.DataFrame] = []
    for symbol in symbols:
        if not (symbol.startswith("VN:") or symbol.startswith("VNW:")):
            continue
        ticker = symbol.split(":", 1)[1]
        path = _fund_cache_path(ticker, termtype)
        if path.exists():
            frames.append(pl.read_parquet(path).with_columns(pl.lit(symbol).alias("symbol")))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical")


def vn_fundamentals_to_fmp_like(raw: pl.DataFrame) -> pl.DataFrame:
    """Convert tidy KBS statement rows into FMP-like canonical wide rows."""

    if raw.is_empty():
        return raw
    selected = raw.join(_canonical_item_map(), on=["report_type", "item_en"], how="inner")
    if selected.is_empty():
        return pl.DataFrame()

    wide = (
        selected.group_by(["symbol", "report_date", "column"])
        .agg(pl.col("value").drop_nulls().first().alias("value"))
        .pivot(
            values="value",
            index=["symbol", "report_date"],
            on="column",
            aggregate_function="first",
        )
        .rename({"report_date": "reportDate"})
        .with_columns(pl.col("reportDate").cast(pl.Date, strict=False))
        .sort(["symbol", "reportDate"])
    )

    if {"totalLiabilities", "totalAssets"}.issubset(wide.columns):
        wide = wide.with_columns(
            (pl.col("totalLiabilities") / (pl.col("totalAssets") + 1e-8)).alias(
                "totalLiabilitiesToAssets"
            )
        )
    if {"cashAndCashEquivalents", "totalAssets"}.issubset(wide.columns):
        wide = wide.with_columns(
            (pl.col("cashAndCashEquivalents") / (pl.col("totalAssets") + 1e-8)).alias(
                "cashAndCashEquivalentsToAssets"
            )
        )

    for source, target in GROWTH_SOURCE_COLUMNS.items():
        if source not in wide.columns:
            continue
        previous = pl.col(source).shift(1).over("symbol")
        wide = wide.with_columns(
            pl.when(previous.abs() > 1e-8)
            .then((pl.col(source) / previous) - 1)
            .otherwise(None)
            .alias(target)
        )
    return wide


def join_fundamentals_asof(
    prices: pl.DataFrame,
    fundamentals: pl.DataFrame,
    *,
    date_column: str = "date",
    report_date_column: str = "reportDate",
    reporting_lag_months: int = 0,
    availability_date_column: str = "fundamentalAvailableDate",
) -> pl.DataFrame:
    """As-of join FMP-like fundamentals onto a price panel."""

    if fundamentals.is_empty() or report_date_column not in fundamentals.columns:
        return prices
    if reporting_lag_months < 0:
        raise ValueError("reporting_lag_months must be non-negative")
    join_date_column = report_date_column
    fundamentals_for_join = fundamentals
    if reporting_lag_months:
        join_date_column = availability_date_column
        fundamentals_for_join = fundamentals.with_columns(
            pl.col(report_date_column)
            .dt.offset_by(f"{reporting_lag_months}mo")
            .alias(availability_date_column)
        )
    left = (
        prices.select([date_column, "symbol"])
        .sort([date_column, "symbol"])
        .set_sorted(date_column)
    )
    right_columns = [
        column
        for column in fundamentals_for_join.columns
        if column != date_column or column == report_date_column
    ]
    right = (
        fundamentals_for_join.select(right_columns)
        .sort([join_date_column, "symbol"])
        .set_sorted(join_date_column)
    )
    joined = (
        left
        .join_asof(
            right,
            left_on=date_column,
            right_on=join_date_column,
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
    )
    new_columns = [column for column in joined.columns if column not in {date_column, "symbol"}]
    if not new_columns:
        return prices
    return prices.join(
        joined.select([date_column, "symbol", *new_columns]),
        on=[date_column, "symbol"],
        how="left",
    )


def add_factor_scores(
    df: pl.DataFrame,
    *,
    factor_groups: dict[str, dict[str, Any]] | None = None,
) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    """Append cross-sectional canonical fundamental factor scores."""

    result = df
    used: dict[str, list[str]] = {}
    group_config = factor_groups or FUNDAMENTAL_FACTOR_GROUPS
    factor_columns: list[str] = []
    for factor_name, spec in group_config.items():
        columns = [column for column in spec["columns"] if column in result.columns]
        if not columns:
            continue
        sign = float(spec.get("sign", 1.0))
        result = result.with_columns(
            (
                sum(
                    sign
                    * (
                        (pl.col(column) - pl.col(column).mean().over("date"))
                        / (pl.col(column).std().over("date") + 1e-8)
                    )
                    for column in columns
                )
                / len(columns)
            )
            .fill_nan(None)
            .fill_null(0.0)
            .alias(factor_name)
        )
        used[factor_name] = columns
        factor_columns.append(factor_name)
    if factor_columns:
        result = result.with_columns(
            (sum(pl.col(column) for column in factor_columns) / len(factor_columns)).alias(
                "fundamentalCompositeFactor"
            )
        )
        used["fundamentalCompositeFactor"] = factor_columns
    return result, used


@Registry.register_feature("fundamental")
class FundamentalFeatures(BaseFeature):
    """Market-neutral fundamental features in FMP-like wide format."""

    def __init__(self, fundamentals: pl.DataFrame | None = None) -> None:
        self.fundamentals = fundamentals if fundamentals is not None else pl.DataFrame()

    def requires_fundamentals(self) -> bool:
        return True

    def with_fundamentals(self, fundamentals: pl.DataFrame) -> BaseFeature:
        return type(self)(fundamentals=fundamentals)

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        original = df
        fundamentals = normalize_fmp_fundamentals(self.fundamentals)
        if fundamentals.is_empty() or "symbol" not in fundamentals.columns:
            return df
        if "reportDate" in fundamentals.columns and "date" in df.columns:
            transformed = join_fundamentals_asof(df, fundamentals)
        else:
            transformed = df.join(fundamentals, on="symbol", how="left")
        return self._validate_append_only(original, transformed)


@Registry.register_feature("vn_fundamental")
class VNFundamentalFeatures(BaseFeature):
    """VN fundamentals normalized to the same FMP-like wide format."""

    def __init__(
        self,
        termtype: int = 1,
        include_factor_scores: bool = False,
        reporting_lag_months: int = 5,
    ) -> None:
        self.termtype = termtype
        self.include_factor_scores = include_factor_scores
        self.reporting_lag_months = reporting_lag_months

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        original = df
        raw = load_vn_fundamental_cache(df["symbol"].unique().to_list(), termtype=self.termtype)
        fundamentals = vn_fundamentals_to_fmp_like(raw)
        transformed = join_fundamentals_asof(
            df,
            fundamentals,
            reporting_lag_months=self.reporting_lag_months,
        )
        if self.include_factor_scores:
            transformed, _ = add_factor_scores(transformed)
        return self._validate_append_only(original, transformed)
