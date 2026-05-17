from __future__ import annotations

from datetime import date

import polars as pl

from qts.research.features.fundamentals import (
    FundamentalFeatures,
    VNFundamentalFeatures,
    join_fundamentals_asof,
    vn_fundamentals_to_fmp_like,
)


def test_fundamental_features_append_fmp_aliases() -> None:
    prices = pl.DataFrame(
        {
            "date": [date(2024, 1, 2)],
            "symbol": ["AAPL"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1_000.0],
        }
    )
    fundamentals = pl.DataFrame(
        {
            "symbol": ["AAPL"],
            "date": [date(2024, 1, 1)],
            "pe_ratio": [21.5],
            "ev_ebitda": [15.2],
        }
    )

    featured = FundamentalFeatures(fundamentals).fit_transform(prices)

    assert featured["pe_ratio"].to_list() == [21.5]
    assert featured["reportDate"].to_list() == [date(2024, 1, 1)]
    assert featured["priceToEarningsRatio"].to_list() == [21.5]
    assert featured["enterpriseValueOverEBITDA"].to_list() == [15.2]
    assert "date_right" not in featured.columns


def test_vn_fundamentals_to_fmp_like_maps_kbs_rows_and_growth() -> None:
    raw = pl.DataFrame(
        [
            {
                "symbol": "VN:VNM",
                "report_date": date(2022, 12, 31),
                "report_type": "CSTC",
                "item_en": "P/E",
                "value": 18.0,
            },
            {
                "symbol": "VN:VNM",
                "report_date": date(2022, 12, 31),
                "report_type": "KQKD",
                "item_en": "1. Revenue",
                "value": 100.0,
            },
            {
                "symbol": "VN:VNM",
                "report_date": date(2022, 12, 31),
                "report_type": "CDKT",
                "item_en": "TOTAL ASSETS",
                "value": 250.0,
            },
            {
                "symbol": "VN:VNM",
                "report_date": date(2022, 12, 31),
                "report_type": "LCTT",
                "item_en": "Net cash flows from operating activities",
                "value": 12.0,
            },
            {
                "symbol": "VN:VNM",
                "report_date": date(2023, 12, 31),
                "report_type": "CSTC",
                "item_en": "P/E",
                "value": 20.0,
            },
            {
                "symbol": "VN:VNM",
                "report_date": date(2023, 12, 31),
                "report_type": "KQKD",
                "item_en": "1. Revenue",
                "value": 125.0,
            },
            {
                "symbol": "VN:VNM",
                "report_date": date(2023, 12, 31),
                "report_type": "CDKT",
                "item_en": "TOTAL ASSETS",
                "value": 300.0,
            },
            {
                "symbol": "VN:VNM",
                "report_date": date(2023, 12, 31),
                "report_type": "LCTT",
                "item_en": "Net cash flows from operating activities",
                "value": 15.0,
            },
        ]
    )

    wide = vn_fundamentals_to_fmp_like(raw)
    latest = wide.filter(pl.col("reportDate") == date(2023, 12, 31)).row(
        0,
        named=True,
    )

    assert latest["priceToEarningsRatio"] == 20.0
    assert latest["revenue"] == 125.0
    assert latest["totalAssets"] == 300.0
    assert latest["operatingCashFlow"] == 15.0
    assert latest["growthRevenue"] == 0.25


def test_join_fundamentals_asof_uses_report_date() -> None:
    prices = pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 7, 1)],
            "symbol": ["AAPL", "AAPL"],
            "close": [100.0, 110.0],
        }
    )
    fundamentals = pl.DataFrame(
        {
            "symbol": ["AAPL"],
            "reportDate": [date(2024, 3, 31)],
            "priceToEarningsRatio": [19.5],
        }
    )

    joined = join_fundamentals_asof(prices, fundamentals)

    assert joined["priceToEarningsRatio"].to_list() == [None, 19.5]


def test_join_fundamentals_asof_can_lag_fiscal_period_date() -> None:
    prices = pl.DataFrame(
        {
            "date": [date(2024, 7, 1), date(2024, 8, 31), date(2024, 9, 2)],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "close": [100.0, 105.0, 110.0],
        }
    )
    fundamentals = pl.DataFrame(
        {
            "symbol": ["AAPL"],
            "reportDate": [date(2024, 3, 31)],
            "priceToEarningsRatio": [19.5],
        }
    )

    joined = join_fundamentals_asof(prices, fundamentals, reporting_lag_months=5)

    assert joined["priceToEarningsRatio"].to_list() == [None, 19.5, 19.5]
    assert joined["reportDate"].to_list() == [None, date(2024, 3, 31), date(2024, 3, 31)]


def test_vn_fundamental_features_default_to_five_month_reporting_lag() -> None:
    assert VNFundamentalFeatures().reporting_lag_months == 5
