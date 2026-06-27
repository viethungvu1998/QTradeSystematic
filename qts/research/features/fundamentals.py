"""Fundamental feature normalization and joins."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import polars as pl

from qts.core.registry import Registry
from qts.research.features.base import BaseFeature
from qts.utils.paths import cache_dir

TARGET_COLUMNS = [
    "priceToSalesRatio",
    "debtToEquityRatio",
    "cashRatio",
    "quickRatio",
    "cashEndPeriod",
    "priceToBookRatio",
    "preTaxProfitMargin",
    "totalEquity",
    "netIncome",
    "priceToEarningsRatio",
    "currentRatio",
    "returnOnAssets",
    "totalLiabilities",
    "returnOnEquity",
    "totalAssets",
    "grossProfitMargin",
    "netProfitMargin",
    "debtToAssetsRatio",
    "totalLiabilitiesToAssets",
]

ANNUAL_AUDIT_BASE_COLUMNS = [
    "returnOnEquity",
    "totalAssets",
    "shortTermAssets",
    "revenue",
    "netProfitMargin",
    "financingCashFlow",
    "totalLiabilities",
    "priceToEarningsRatio",
    "preTaxProfitMargin",
    "grossProfit",
    "netRevenue",
    "returnOnAssets",
    "netCashFlow",
    "debtToEquityRatio",
    "grossProfitMargin",
    "cashAndCashEquivalents",
    "operatingCashFlow",
    "profitBeforeTax",
    "totalEquity",
    "shortTermLiabilities",
    "priceToBookRatio",
    "investingCashFlow",
    "cashEndPeriod",
    "cashRatio",
    "longTermLiabilities",
    "enterpriseValueOverEBITDA",
    "currentRatio",
    "netIncome",
    "quickRatio",
    "priceToSalesRatio",
    "longTermAssets",
    "report_date",
    "debtToAssetsRatio",
    "totalLiabilitiesToAssets",
    "cashAndCashEquivalentsToAssets",
]
ANNUAL_AUDIT_YOY_COLUMNS = [
    "revenue_yoy",
    "netIncome_yoy",
    "profitBeforeTax_yoy",
    "totalAssets_yoy",
]
ANNUAL_AUDIT_COLUMNS = [*ANNUAL_AUDIT_BASE_COLUMNS, *ANNUAL_AUDIT_YOY_COLUMNS]
_ANNUAL_YOY_METADATA_COLUMNS = frozenset(
    {"symbol", "fiscal_year", "quarter", "period_end", "available_from", "report_date"}
)
_ANNUAL_YOY_NUMERIC_DTYPES = frozenset(
    {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
        pl.Decimal,
    }
)
_DERIVED_ANNUAL_COLUMNS: dict[str, tuple[str, ...]] = {
    "debtToAssetsRatio": ("totalLiabilities", "totalAssets"),
    "totalLiabilitiesToAssets": ("totalLiabilities", "totalAssets"),
    "cashAndCashEquivalentsToAssets": ("cashAndCashEquivalents", "totalAssets"),
}
_RAW_REPORT_TYPE_BY_COLUMN = {
    "returnOnEquity": "CSTC",
    "totalAssets": "CDKT",
    "shortTermAssets": "CDKT",
    "revenue": "KQKD",
    "netProfitMargin": "CSTC",
    "financingCashFlow": "LCTT",
    "totalLiabilities": "CDKT",
    "priceToEarningsRatio": "CSTC",
    "preTaxProfitMargin": "CSTC",
    "grossProfit": "KQKD",
    "netRevenue": "KQKD",
    "returnOnAssets": "CSTC",
    "netCashFlow": "LCTT",
    "debtToEquityRatio": "CSTC",
    "grossProfitMargin": "CSTC",
    "cashAndCashEquivalents": "CDKT",
    "operatingCashFlow": "LCTT",
    "profitBeforeTax": "KQKD",
    "totalEquity": "CDKT",
    "shortTermLiabilities": "CDKT",
    "priceToBookRatio": "CSTC",
    "investingCashFlow": "LCTT",
    "cashEndPeriod": "LCTT",
    "cashRatio": "CSTC",
    "longTermLiabilities": "CDKT",
    "enterpriseValueOverEBITDA": "CSTC",
    "currentRatio": "CSTC",
    "netIncome": "KQKD",
    "quickRatio": "CSTC",
    "priceToSalesRatio": "CSTC",
    "longTermAssets": "CDKT",
}
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

VCI_FUNDAMENTAL_ITEMS = {
    "KQKD": {
        "revenue": [
            "Sales",
            "Total Operating Income",
            "1. Insurance premium (01=01.1+01.2-01.3)",
            "Revenue from insurance premium",
        ],
        "netRevenue": [
            "Net sales",
            "Net Interest Income",
            "7. Total net revenue from insurance business",
            "Net revenue of insurance premium",
        ],
        "grossProfit": ["Gross Profit", "Gross profit", "Net Operating Profit Before Allowance for Credit Loss"],
        "profitBeforeTax": [
            "Net accounting profit/(loss) before tax",
            "Net Accounting Profit/(loss) before tax",
            "29. Total profit before tax",
            "Profit before tax",
            "IX. Profit before tax",
        ],
        "netIncome": [
            "Net profit/(loss) after tax",
            "34. Profit after tax",
            "Profit after tax",
            "NET PROFIT/(LOSS) AFTER TAX",
            "XI.  Net profit after tax",
        ],
    },
    "CDKT": {
        "totalAssets": ["Total Assets", "TOTAL ASSETS"],
        "totalLiabilities": ["Liabilities", "TOTAL LIABILITIES", "LIABILITIES", "A.  LIABILITIES"],
        "totalEquity": [
            "Owner's Equity",
            "Owner's equity",
            "OWNER'S EQUITY",
            "Shareholders' equity",
            "Owners' equity",
            "I. Owner's equity",
            "B. OWNER'S EQUITY",
        ],
        "cashAndCashEquivalents": ["Cash and cash equivalents"],
        "shortTermAssets": ["CURRENT ASSETS", "Current assets"],
        "longTermAssets": ["LONG-TERM ASSETS", "B. LONG-TERM ASSETS", "Long-term assets"],
        "shortTermLiabilities": ["Current liabilities", "Short-term liabilities", "SHORT-TERM LIABILITIES"],
        "longTermLiabilities": [
            "Long-term liabilities",
            "LONG-TERM LIABILITIES",
            "II. Long-term liabilities",
            "Long-term borrowings and liabilities",
            "3. Other long-term liabilities",
        ],
    },
    "LCTT": {
        "operatingCashFlow": [
            "Net cash flows from operating activities",
            "Net cash from operating activities",
            "Net cash inflows/(outflows) from operating activities",
        ],
        "investingCashFlow": [
            "Net cash flows from investing activities",
            "Net cash from investing activities",
            "Net cash inflows/(outflows) from investing activities",
        ],
        "financingCashFlow": [
            "Net cash flows from financing activities",
            "Net cash from financing activities",
            "Net cash inflows/(outflows) from financing activities",
        ],
        "netCashFlow": [
            "Net increase in cash and cash equivalents",
            "Net cash flows during the period",
            "Net Increase/(Decrease) in cash and cash equivalents",
            "IV. Net cash flows during the period",
        ],
        "cashEndPeriod": [
            "Cash and cash equivalents at the end of period",
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

_VCI_ALIAS_PRIORITY: dict[tuple[str, str], int] = {
    ("CDKT", "Total Assets"): 20,
    ("CDKT", "Liabilities"): 20,
    ("CDKT", "Owner's Equity"): 20,
    ("CDKT", "Owner's equity"): 20,
    ("CDKT", "Current assets"): 20,
    ("CDKT", "CURRENT ASSETS"): 20,
    ("CDKT", "Long-term assets"): 20,
    ("CDKT", "LONG-TERM ASSETS"): 20,
    ("CDKT", "TOTAL ASSETS"): 10,
    ("CDKT", "TOTAL LIABILITIES"): 10,
    ("CDKT", "OWNER'S EQUITY"): 10,
    ("CSTC", "evToEbitda"): 20,
}


def _fund_cache_path(ticker: str, termtype: int) -> Path:
    label = "annual" if termtype == 1 else "quarterly"
    return cache_dir() / "vn_fundamentals" / f"{ticker}_{label}.parquet"


def _canonical_item_map() -> pl.DataFrame:
    rows = []
    for report_type, feature_map in VCI_FUNDAMENTAL_ITEMS.items():
        for column, item_names in feature_map.items():
            for item_name in item_names:
                rows.append(
                    {
                        "report_type": report_type,
                        "item_en": item_name,
                        "column": column,
                        "priority": _VCI_ALIAS_PRIORITY.get((report_type, item_name), 0),
                    }
                )
    for source, target in FMP_ALIASES.items():
        rows.extend(
            [
                {"report_type": "CSTC", "item_en": source, "column": target, "priority": 0},
                {"report_type": "CSTC", "item_en": source.lower(), "column": target, "priority": 0},
                {"report_type": "CSTC", "item_en": target, "column": target, "priority": 0},
            ]
        )
    rows.extend(
        [
            {"report_type": "CSTC", "item_en": "debtToEquity", "column": "debtToEquityRatio", "priority": 0},
            {"report_type": "CSTC", "item_en": "debtPerEquity", "column": "debtToEquityRatio", "priority": 0},
            {"report_type": "CSTC", "item_en": "evToEbitda", "column": "enterpriseValueOverEBITDA", "priority": 20},
            {"report_type": "CSTC", "item_en": "grossMargin", "column": "grossProfitMargin", "priority": 0},
            {"report_type": "CSTC", "item_en": "preTaxProfitMargin", "column": "preTaxProfitMargin", "priority": 0},
            {"report_type": "CSTC", "item_en": "afterTaxProfitMargin", "column": "netProfitMargin", "priority": 0},
            {"report_type": "CSTC", "item_en": "currentRatio", "column": "currentRatio", "priority": 0},
            {"report_type": "CSTC", "item_en": "quickRatio", "column": "quickRatio", "priority": 0},
            {"report_type": "CSTC", "item_en": "cashRatio", "column": "cashRatio", "priority": 0},
        ]
    )
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


def compute_fundamental_available_date(
    period_end: pl.Expr,
    *,
    reporting_lag_months: int = 5,
) -> pl.Expr:
    """Return the month-start date when a filing is assumed tradable."""

    lagged = period_end.dt.offset_by(f"{reporting_lag_months}mo")
    return pl.date(lagged.dt.year(), lagged.dt.month(), pl.lit(1, dtype=pl.Int8))


def add_derived_fundamental_metrics(frame: pl.DataFrame) -> pl.DataFrame:
    """Append derived balance-sheet ratios used by annual/quarterly pipelines."""

    result = frame
    denominator = pl.col("totalAssets").cast(pl.Float64, strict=False)
    safe_denominator = pl.when(denominator.abs() > 1e-12).then(denominator).otherwise(None)
    if {"totalLiabilities", "totalAssets"}.issubset(result.columns):
        derived_debt_to_assets = (
            (pl.col("totalLiabilities").cast(pl.Float64, strict=False) / safe_denominator) * 100.0
        )
        derived_liabilities_to_assets = (
            pl.col("totalLiabilities").cast(pl.Float64, strict=False) / safe_denominator
        )
        if "debtToAssetsRatio" in result.columns:
            result = result.with_columns(
                pl.coalesce(pl.col("debtToAssetsRatio"), derived_debt_to_assets).alias(
                    "debtToAssetsRatio"
                )
            )
        else:
            result = result.with_columns(derived_debt_to_assets.alias("debtToAssetsRatio"))
        if "totalLiabilitiesToAssets" in result.columns:
            result = result.with_columns(
                pl.coalesce(pl.col("totalLiabilitiesToAssets"), derived_liabilities_to_assets).alias(
                    "totalLiabilitiesToAssets"
                )
            )
        else:
            result = result.with_columns(
                derived_liabilities_to_assets.alias("totalLiabilitiesToAssets")
            )
    if {"cashAndCashEquivalents", "totalAssets"}.issubset(result.columns):
        derived_cash_to_assets = (
            pl.col("cashAndCashEquivalents").cast(pl.Float64, strict=False) / safe_denominator
        )
        if "cashAndCashEquivalentsToAssets" in result.columns:
            result = result.with_columns(
                pl.coalesce(
                    pl.col("cashAndCashEquivalentsToAssets"),
                    derived_cash_to_assets,
                ).alias("cashAndCashEquivalentsToAssets")
            )
        else:
            result = result.with_columns(
                derived_cash_to_assets.alias("cashAndCashEquivalentsToAssets")
            )
    return result


def add_yoy_growth_columns(
    frame: pl.DataFrame,
    *,
    metrics: list[str],
    entity_column: str = "symbol",
    sort_column: str = "available_from",
    lag: int = 1,
    suffix: str = "_yoy",
) -> pl.DataFrame:
    """Append growth columns as ``<metric>{suffix}`` using the provided lag."""

    available_metrics = [metric for metric in metrics if metric in frame.columns]
    if not available_metrics:
        return frame
    sorted_frame = frame.sort([entity_column, sort_column])
    expressions: list[pl.Expr] = []
    for metric in available_metrics:
        current = pl.col(metric).cast(pl.Float64, strict=False)
        previous = current.shift(lag).over(entity_column)
        expressions.append(
            pl.when(previous.abs() > 1e-12)
            .then((current - previous) / previous.abs())
            .otherwise(None)
            .alias(f"{metric}{suffix}")
        )
    return sorted_frame.with_columns(expressions)


def _is_numeric_dtype(dtype: pl.DataType) -> bool:
    return dtype in _ANNUAL_YOY_NUMERIC_DTYPES


def _default_annual_yoy_metrics(frame: pl.DataFrame) -> list[str]:
    metrics: list[str] = []
    for column, dtype in frame.schema.items():
        if column in _ANNUAL_YOY_METADATA_COLUMNS or column.endswith("_yoy"):
            continue
        if _is_numeric_dtype(dtype):
            metrics.append(column)
    return metrics


def _normalize_alias_phrase(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _configured_aliases_by_column() -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = {}
    for report_type, feature_map in VCI_FUNDAMENTAL_ITEMS.items():
        for column, item_names in feature_map.items():
            aliases.setdefault(column, set()).update(item_names)
    for source, target in FMP_ALIASES.items():
        aliases.setdefault(target, set()).update({source, source.lower(), target})
    aliases.setdefault("debtToEquityRatio", set()).update({"debtToEquity", "debtPerEquity"})
    aliases.setdefault("enterpriseValueOverEBITDA", set()).add("evToEbitda")
    aliases.setdefault("grossProfitMargin", set()).add("grossMargin")
    aliases.setdefault("preTaxProfitMargin", set()).add("preTaxProfitMargin")
    aliases.setdefault("netProfitMargin", set()).add("afterTaxProfitMargin")
    aliases.setdefault("currentRatio", set()).add("currentRatio")
    aliases.setdefault("quickRatio", set()).add("quickRatio")
    aliases.setdefault("cashRatio", set()).add("cashRatio")
    return aliases


_CONFIGURED_ALIASES_BY_COLUMN = _configured_aliases_by_column()


def _audit_raw_candidates_by_column() -> dict[str, set[str]]:
    candidates: dict[str, set[str]] = {}
    for item_map in (KBS_FUNDAMENTAL_ITEMS, VCI_FUNDAMENTAL_ITEMS):
        for _, feature_map in item_map.items():
            for column, item_names in feature_map.items():
                candidates.setdefault(column, set()).update(item_names)
    for source, target in FMP_ALIASES.items():
        candidates.setdefault(target, set()).update({source, source.lower(), target})
    candidates.setdefault("debtToEquityRatio", set()).update({"debtToEquity", "debtPerEquity"})
    candidates.setdefault("enterpriseValueOverEBITDA", set()).add("evToEbitda")
    candidates.setdefault("grossProfitMargin", set()).add("grossMargin")
    candidates.setdefault("preTaxProfitMargin", set()).add("preTaxProfitMargin")
    candidates.setdefault("netProfitMargin", set()).add("afterTaxProfitMargin")
    candidates.setdefault("currentRatio", set()).add("currentRatio")
    candidates.setdefault("quickRatio", set()).add("quickRatio")
    candidates.setdefault("cashRatio", set()).add("cashRatio")
    return candidates


_AUDIT_RAW_CANDIDATES_BY_COLUMN = _audit_raw_candidates_by_column()


def _looks_like_unmapped_alias(column: str, raw_items: list[str]) -> bool:
    expected_aliases = _CONFIGURED_ALIASES_BY_COLUMN.get(column, set())
    if not expected_aliases:
        return False
    for item in raw_items:
        item_phrase = _normalize_alias_phrase(item)
        for alias in expected_aliases:
            alias_phrase = _normalize_alias_phrase(alias)
            if len(alias_phrase) >= 8 and alias_phrase in item_phrase and alias_phrase != item_phrase:
                return True
    return False


def _has_exact_audit_candidate(column: str, raw_items: list[str]) -> bool:
    return any(item in _AUDIT_RAW_CANDIDATES_BY_COLUMN.get(column, set()) for item in raw_items)


def _validate_vn_fundamental_mode(mode: int) -> None:
    if mode not in {1, 2}:
        raise ValueError(f"Unsupported mode={mode}. Use 1 for annual or 2 for quarterly.")


def _period_end_expr(mode: int) -> pl.Expr:
    fiscal_year = pl.col("fiscal_year")
    if mode == 1:
        return pl.date(fiscal_year, pl.lit(12, dtype=pl.Int8), pl.lit(31, dtype=pl.Int8))
    quarter = pl.col("quarter").cast(pl.Int8, strict=False)
    return (
        pl.when(quarter == 1)
        .then(pl.date(fiscal_year, pl.lit(3, dtype=pl.Int8), pl.lit(31, dtype=pl.Int8)))
        .when(quarter == 2)
        .then(pl.date(fiscal_year, pl.lit(6, dtype=pl.Int8), pl.lit(30, dtype=pl.Int8)))
        .when(quarter == 3)
        .then(pl.date(fiscal_year, pl.lit(9, dtype=pl.Int8), pl.lit(30, dtype=pl.Int8)))
        .when(quarter == 4)
        .then(pl.date(fiscal_year, pl.lit(12, dtype=pl.Int8), pl.lit(31, dtype=pl.Int8)))
        .otherwise(None)
    )


def prepare_vn_fundamental_features(
    raw: pl.DataFrame,
    *,
    mode: int = 1,
    reporting_lag_months: int = 5,
    yoy_metrics: list[str] | None = None,
) -> pl.DataFrame:
    """Convert raw VN fundamentals into a canonical wide feature table.

    Flow:
    1. Keep rows for the requested frequency.
    2. Map ``(report_type, item_en)`` into canonical metric columns.
    3. Deduplicate per symbol / period / metric using the latest ``report_date``.
    4. Pivot into one wide row per reporting period.
    5. Append derived ratios and selected ``*_yoy`` columns.
    """

    if raw.is_empty():
        return raw
    _validate_vn_fundamental_mode(mode)

    frame = raw
    if "frequency" in frame.columns:
        frame = frame.filter(pl.col("frequency") == ("annual" if mode == 1 else "quarterly"))
    if "quarter" in frame.columns and mode == 1:
        frame = frame.filter(pl.col("quarter").is_null())
    if "quarter" in frame.columns and mode == 2:
        frame = frame.filter(pl.col("quarter").is_not_null())
    if frame.is_empty():
        return pl.DataFrame()

    cast_columns = [
        pl.col("symbol").cast(pl.Utf8),
        pl.col("fiscal_year").cast(pl.Int32, strict=False),
        pl.col("report_date").cast(pl.Date, strict=False),
    ]
    if "quarter" in frame.columns:
        cast_columns.append(pl.col("quarter").cast(pl.Int8, strict=False))
    frame = frame.with_columns(cast_columns)
    mapped = frame.join(_canonical_item_map(), on=["report_type", "item_en"], how="inner")
    if mapped.is_empty():
        return pl.DataFrame()

    period_end = _period_end_expr(mode)
    normalized = mapped.with_columns(
        [
            period_end.alias("period_end"),
            compute_fundamental_available_date(
                period_end,
                reporting_lag_months=reporting_lag_months,
            ).alias("available_from"),
        ]
    )
    period_keys = ["symbol", "fiscal_year", "period_end", "available_from"]
    group_keys = [*period_keys, "column"]
    sort_keys = ["symbol", "fiscal_year", "report_date", "column"]
    pivot_index = [*period_keys]
    if mode == 2:
        period_keys.insert(2, "quarter")
        group_keys.insert(2, "quarter")
        sort_keys.insert(2, "quarter")
        pivot_index.insert(2, "quarter")
    period_report_dates = normalized.group_by(period_keys).agg(
        pl.col("report_date").max().alias("report_date")
    )
    preferred_priority_by_column = normalized.group_by(group_keys).agg(
        pl.col("priority").max().alias("preferred_priority")
    )
    preferred_rows = (
        normalized.join(preferred_priority_by_column, on=group_keys, how="inner")
        .filter(pl.col("priority") == pl.col("preferred_priority"))
    )
    latest_report_dates_by_column = preferred_rows.group_by(group_keys).agg(
        pl.col("report_date").max().alias("latest_report_date")
    )
    latest_rows = (
        preferred_rows.join(latest_report_dates_by_column, on=group_keys, how="inner")
        .filter(pl.col("report_date") == pl.col("latest_report_date"))
    )
    deduped = latest_rows.group_by(group_keys).agg(
        pl.col("value")
        .drop_nulls()
        .sort_by(pl.col("value").abs())
        .last()
        .alias("value")
    )
    wide = (
        deduped.pivot(
            values="value",
            index=pivot_index,
            on="column",
            aggregate_function="first",
        )
        .sort(["symbol", "available_from"])
    )
    wide = wide.join(period_report_dates, on=period_keys, how="left")
    with_derived = add_derived_fundamental_metrics(wide)
    resolved_yoy_metrics = yoy_metrics
    if mode == 1 and resolved_yoy_metrics is None:
        resolved_yoy_metrics = _default_annual_yoy_metrics(with_derived)
    return add_yoy_growth_columns(
        with_derived,
        metrics=resolved_yoy_metrics or ["revenue", "netIncome", "profitBeforeTax", "totalAssets"],
        lag=1 if mode == 1 else 4,
        suffix="_yoy" if mode == 1 else "_q_yoy",
    )


def prepare_vn_annual_fundamental_features(
    raw: pl.DataFrame,
    *,
    reporting_lag_months: int = 5,
    yoy_metrics: list[str] | None = None,
) -> pl.DataFrame:
    """Backward-compatible annual wrapper around ``prepare_vn_fundamental_features``.

    Annual processing always appends ``*_yoy`` for every numeric processed column,
    excluding metadata such as ``report_date``.
    """

    return prepare_vn_fundamental_features(
        raw,
        mode=1,
        reporting_lag_months=reporting_lag_months,
        yoy_metrics=None,
    )


def prepare_vn_quarterly_fundamental_features(
    raw: pl.DataFrame,
    *,
    reporting_lag_months: int = 5,
    qoq_metrics: list[str] | None = None,
) -> pl.DataFrame:
    """Quarterly wrapper around ``prepare_vn_fundamental_features``.

    Notes:
    - ``qoq_metrics`` names the set of quarterly metrics to transform.
    - The percentage change is computed against the same fiscal quarter in the
      previous financial year, so the output columns remain ``*_q_yoy``.
    """

    return prepare_vn_fundamental_features(
        raw,
        mode=2,
        reporting_lag_months=reporting_lag_months,
        yoy_metrics=qoq_metrics,
    )


def audit_annual_fundamental_availability(
    raw: pl.DataFrame,
    processed: pl.DataFrame | None = None,
    *,
    columns: list[str] | None = None,
) -> pl.DataFrame:
    """Return row-level availability and root-cause audit for annual fundamentals."""

    annual_raw = raw
    if "frequency" in annual_raw.columns:
        annual_raw = annual_raw.filter(pl.col("frequency") == "annual")
    if "quarter" in annual_raw.columns:
        annual_raw = annual_raw.filter(pl.col("quarter").is_null())
    annual_processed = processed if processed is not None else prepare_vn_annual_fundamental_features(raw)
    requested_columns = columns or ANNUAL_AUDIT_COLUMNS

    raw_rows = annual_raw.to_dicts()
    processed_rows = annual_processed.to_dicts()
    keys = {
        (str(row["symbol"]), int(row["fiscal_year"]))
        for row in raw_rows
        if row.get("symbol") is not None and row.get("fiscal_year") is not None
    }
    keys.update(
        {
            (str(row["symbol"]), int(row["fiscal_year"]))
            for row in processed_rows
            if row.get("symbol") is not None and row.get("fiscal_year") is not None
        }
    )
    raw_by_key: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in raw_rows:
        if row.get("symbol") is None or row.get("fiscal_year") is None:
            continue
        raw_by_key.setdefault((str(row["symbol"]), int(row["fiscal_year"])), []).append(row)
    processed_by_key = {
        (str(row["symbol"]), int(row["fiscal_year"])): row
        for row in processed_rows
        if row.get("symbol") is not None and row.get("fiscal_year") is not None
    }
    mapped_raw = annual_raw.join(_canonical_item_map(), on=["report_type", "item_en"], how="left").to_dicts()
    mapped_by_key: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in mapped_raw:
        if row.get("symbol") is None or row.get("fiscal_year") is None:
            continue
        mapped_by_key.setdefault((str(row["symbol"]), int(row["fiscal_year"])), []).append(row)

    records: list[dict[str, object]] = []
    sorted_keys = sorted(keys)
    for symbol, fiscal_year in sorted_keys:
        current_processed = processed_by_key.get((symbol, fiscal_year), {})
        previous_processed = processed_by_key.get((symbol, fiscal_year - 1), {})
        current_raw_rows = raw_by_key.get((symbol, fiscal_year), [])
        current_mapped_rows = mapped_by_key.get((symbol, fiscal_year), [])

        for column in requested_columns:
            value = current_processed.get(column)
            has_value = value is not None
            root_cause = "available"

            if column.endswith("_yoy"):
                base_column = column[: -len("_yoy")]
                current_base = current_processed.get(base_column)
                previous_base = previous_processed.get(base_column)
                if has_value:
                    root_cause = "available"
                elif not previous_processed:
                    root_cause = "YoY missing because prior year is unavailable"
                elif current_base is None or previous_base is None or abs(float(previous_base)) <= 1e-12:
                    root_cause = "YoY missing because current or prior value is null/zero"
                else:
                    root_cause = "duplicate alias conflict resolved incorrectly"
            elif column == "report_date":
                if has_value:
                    root_cause = "available"
                elif current_raw_rows:
                    root_cause = "duplicate alias conflict resolved incorrectly"
                else:
                    root_cause = "source missing in raw annual data"
            elif column in _DERIVED_ANNUAL_COLUMNS:
                if has_value:
                    root_cause = "available"
                else:
                    prerequisites = _DERIVED_ANNUAL_COLUMNS[column]
                    if any(current_processed.get(prerequisite) is None for prerequisite in prerequisites):
                        root_cause = "derived metric missing because prerequisite columns are missing"
                    else:
                        root_cause = "duplicate alias conflict resolved incorrectly"
            else:
                expected_report_type = _RAW_REPORT_TYPE_BY_COLUMN.get(column)
                mapped_candidates = [
                    row
                    for row in current_mapped_rows
                    if row.get("column") == column
                ]
                report_type_rows = (
                    [row for row in current_raw_rows if row.get("report_type") == expected_report_type]
                    if expected_report_type is not None
                    else current_raw_rows
                )
                if has_value:
                    root_cause = "available"
                elif mapped_candidates:
                    root_cause = "duplicate alias conflict resolved incorrectly"
                elif _has_exact_audit_candidate(
                    column,
                    [str(row.get("item_en")) for row in report_type_rows if row.get("item_en") is not None],
                ):
                    root_cause = "raw item exists but is excluded by VCI-only policy"
                elif report_type_rows and _looks_like_unmapped_alias(
                    column,
                    [str(row.get("item_en")) for row in report_type_rows if row.get("item_en") is not None],
                ):
                    root_cause = "raw item exists but alias is unmapped"
                else:
                    root_cause = "source missing in raw annual data"

            records.append(
                {
                    "symbol": symbol,
                    "fiscal_year": fiscal_year,
                    "column": column,
                    "value_available": has_value,
                    "root_cause": root_cause,
                }
            )
    return pl.DataFrame(records)


def summarize_annual_fundamental_availability(audit: pl.DataFrame) -> pl.DataFrame:
    """Summarize annual availability audit by column."""

    if audit.is_empty():
        return pl.DataFrame(
            schema={
                "column": pl.Utf8,
                "rows": pl.UInt32,
                "available_rows": pl.UInt32,
                "null_rows": pl.UInt32,
                "coverage_pct": pl.Float64,
                "root_causes": pl.List(pl.Utf8),
                "sample_null_symbols": pl.List(pl.Utf8),
            }
        )
    return (
        audit.group_by("column")
        .agg(
            [
                pl.len().alias("rows"),
                pl.col("value_available").sum().alias("available_rows"),
                (pl.len() - pl.col("value_available").sum()).alias("null_rows"),
                ((pl.col("value_available").sum() * 100.0) / pl.len()).alias("coverage_pct"),
                pl.col("root_cause").filter(pl.col("root_cause") != "available").unique().sort().alias("root_causes"),
                pl.col("symbol")
                .filter(~pl.col("value_available"))
                .unique()
                .sort()
                .head(5)
                .alias("sample_null_symbols"),
            ]
        )
        .sort("column")
    )


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
    """Convert tidy VN statement rows into FMP-like canonical wide rows."""

    if raw.is_empty():
        return raw
    selected = raw.join(_canonical_item_map(), on=["report_type", "item_en"], how="inner")
    if selected.is_empty():
        return pl.DataFrame()
    preferred_priority = selected.group_by(["symbol", "report_date", "column"]).agg(
        pl.col("priority").max().alias("preferred_priority")
    )
    preferred_selected = (
        selected.join(preferred_priority, on=["symbol", "report_date", "column"], how="inner")
        .filter(pl.col("priority") == pl.col("preferred_priority"))
    )

    wide = (
        preferred_selected.group_by(["symbol", "report_date", "column"])
        .agg(
            pl.col("value")
            .drop_nulls()
            .sort_by(pl.col("value").abs())
            .last()
            .alias("value")
        )
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
