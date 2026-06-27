from __future__ import annotations

from datetime import date

import polars as pl

from qts.research.features.fundamentals import (
    FundamentalFeatures,
    VNFundamentalFeatures,
    add_derived_fundamental_metrics,
    audit_annual_fundamental_availability,
    prepare_vn_annual_fundamental_features,
    prepare_vn_fundamental_features,
    prepare_vn_quarterly_fundamental_features,
    join_fundamentals_asof,
    summarize_annual_fundamental_availability,
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


def test_vn_fundamentals_to_fmp_like_maps_vci_rows_and_growth() -> None:
    raw = pl.DataFrame(
        [
            {
                "symbol": "VN:VNM",
                "report_date": date(2022, 12, 31),
                "report_type": "CSTC",
                "item_en": "pe",
                "value": 18.0,
            },
            {
                "symbol": "VN:VNM",
                "report_date": date(2022, 12, 31),
                "report_type": "KQKD",
                "item_en": "Sales",
                "value": 100.0,
            },
            {
                "symbol": "VN:VNM",
                "report_date": date(2022, 12, 31),
                "report_type": "CDKT",
                "item_en": "Total Assets",
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
                "item_en": "pe",
                "value": 20.0,
            },
            {
                "symbol": "VN:VNM",
                "report_date": date(2023, 12, 31),
                "report_type": "KQKD",
                "item_en": "Sales",
                "value": 125.0,
            },
            {
                "symbol": "VN:VNM",
                "report_date": date(2023, 12, 31),
                "report_type": "CDKT",
                "item_en": "Total Assets",
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


def test_vn_fundamentals_to_fmp_like_prefers_reasonable_latest_duplicate_value() -> None:
    raw = pl.DataFrame(
        [
            {
                "symbol": "VN:VNM",
                "report_date": date(2024, 3, 31),
                "report_type": "CDKT",
                "item_en": "Total Assets",
                "value": 300000.0,
            },
            {
                "symbol": "VN:VNM",
                "report_date": date(2024, 3, 31),
                "report_type": "CDKT",
                "item_en": "TOTAL ASSETS",
                "value": 300.0,
            },
            {
                "symbol": "VN:VNM",
                "report_date": date(2024, 3, 31),
                "report_type": "CSTC",
                "item_en": "EV/EBITDA",
                "value": 0.0,
            },
            {
                "symbol": "VN:VNM",
                "report_date": date(2024, 3, 31),
                "report_type": "CSTC",
                "item_en": "evToEbitda",
                "value": 5.6,
            },
        ]
    )

    wide = vn_fundamentals_to_fmp_like(raw)
    row = wide.row(0, named=True)

    assert row["totalAssets"] == 300000.0
    assert row["enterpriseValueOverEBITDA"] == 5.6


def test_prepare_vn_annual_fundamental_features_prefers_higher_priority_vci_alias() -> None:
    raw = pl.DataFrame(
        [
            {
                "symbol": "VN:CMG",
                "report_type": "CDKT",
                "period": "2025",
                "fiscal_year": 2025,
                "quarter": None,
                "report_date": date(2026, 2, 2),
                "item_en": "Current assets",
                "value": 5728700000000.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:CMG",
                "report_type": "CDKT",
                "period": "2025",
                "fiscal_year": 2025,
                "quarter": None,
                "report_date": date(2026, 2, 2),
                "item_en": "Total Assets",
                "value": 10048000000000.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:CMG",
                "report_type": "CDKT",
                "period": "2025",
                "fiscal_year": 2025,
                "quarter": None,
                "report_date": date(2026, 4, 29),
                "item_en": "TOTAL ASSETS",
                "value": 10485000000.0,
                "frequency": "annual",
            },
        ]
    )

    featured = prepare_vn_annual_fundamental_features(raw)
    row = featured.row(0, named=True)

    assert row["totalAssets"] == 10048000000000.0
    assert row["shortTermAssets"] == 5728700000000.0


def test_prepare_vn_annual_fundamental_features_maps_vci_bank_style_aliases() -> None:
    raw = pl.DataFrame(
        [
            {
                "symbol": "VN:ACB",
                "report_type": "KQKD",
                "period": "2025",
                "fiscal_year": 2025,
                "quarter": None,
                "report_date": date(2026, 2, 27),
                "item_en": "Total Operating Income",
                "value": 33800000000000.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:ACB",
                "report_type": "KQKD",
                "period": "2025",
                "fiscal_year": 2025,
                "quarter": None,
                "report_date": date(2026, 2, 27),
                "item_en": "Net Accounting Profit/(loss) before tax",
                "value": 19540000000000.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:ACB",
                "report_type": "KQKD",
                "period": "2025",
                "fiscal_year": 2025,
                "quarter": None,
                "report_date": date(2026, 2, 27),
                "item_en": "Net profit/(loss) after tax",
                "value": 15620000000000.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:ACB",
                "report_type": "LCTT",
                "period": "2025",
                "fiscal_year": 2025,
                "quarter": None,
                "report_date": date(2026, 2, 27),
                "item_en": "Cash and cash equivalents at end of the period",
                "value": 97000000000000.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:ACB",
                "report_type": "LCTT",
                "period": "2025",
                "fiscal_year": 2025,
                "quarter": None,
                "report_date": date(2026, 2, 27),
                "item_en": "Net cash from operating activities",
                "value": 22000000000000.0,
                "frequency": "annual",
            },
        ]
    )

    featured = prepare_vn_annual_fundamental_features(raw)
    row = featured.row(0, named=True)

    assert row["revenue"] == 33800000000000.0
    assert row["profitBeforeTax"] == 19540000000000.0
    assert row["netIncome"] == 15620000000000.0
    assert row["cashEndPeriod"] == 97000000000000.0
    assert row["operatingCashFlow"] == 22000000000000.0


def test_prepare_vn_annual_fundamental_features_maps_vci_section_headers_and_cashflow_variants() -> None:
    raw = pl.DataFrame(
        [
            {
                "symbol": "VN:VTP",
                "report_type": "CDKT",
                "period": "2025",
                "fiscal_year": 2025,
                "quarter": None,
                "report_date": date(2026, 3, 31),
                "item_en": "B. LONG-TERM ASSETS",
                "value": 450.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:VTP",
                "report_type": "LCTT",
                "period": "2025",
                "fiscal_year": 2025,
                "quarter": None,
                "report_date": date(2026, 3, 31),
                "item_en": "Net cash inflows/(outflows) from investing activities",
                "value": -20.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:VTP",
                "report_type": "LCTT",
                "period": "2025",
                "fiscal_year": 2025,
                "quarter": None,
                "report_date": date(2026, 3, 31),
                "item_en": "Net cash inflows/(outflows) from financing activities",
                "value": 35.0,
                "frequency": "annual",
            },
        ]
    )

    featured = prepare_vn_annual_fundamental_features(raw)
    row = featured.row(0, named=True)

    assert row["longTermAssets"] == 450.0
    assert row["investingCashFlow"] == -20.0
    assert row["financingCashFlow"] == 35.0


def test_prepare_vn_annual_fundamental_features_maps_securities_firm_aliases() -> None:
    raw = pl.DataFrame(
        [
            {
                "symbol": "VN:VCI",
                "report_type": "CDKT",
                "period": "2025",
                "fiscal_year": 2025,
                "quarter": None,
                "report_date": date(2026, 3, 31),
                "item_en": "LIABILITIES",
                "value": 900.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:VCI",
                "report_type": "CDKT",
                "period": "2025",
                "fiscal_year": 2025,
                "quarter": None,
                "report_date": date(2026, 3, 31),
                "item_en": "Shareholders' equity",
                "value": 600.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:VCI",
                "report_type": "KQKD",
                "period": "2025",
                "fiscal_year": 2025,
                "quarter": None,
                "report_date": date(2026, 3, 31),
                "item_en": "IX. Profit before tax",
                "value": 120.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:VCI",
                "report_type": "KQKD",
                "period": "2025",
                "fiscal_year": 2025,
                "quarter": None,
                "report_date": date(2026, 3, 31),
                "item_en": "NET PROFIT/(LOSS) AFTER TAX",
                "value": 96.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:VCI",
                "report_type": "LCTT",
                "period": "2025",
                "fiscal_year": 2025,
                "quarter": None,
                "report_date": date(2026, 3, 31),
                "item_en": "IV. Net cash flows during the period",
                "value": 45.0,
                "frequency": "annual",
            },
        ]
    )

    featured = prepare_vn_annual_fundamental_features(raw)
    row = featured.row(0, named=True)

    assert row["totalLiabilities"] == 900.0
    assert row["totalEquity"] == 600.0
    assert row["profitBeforeTax"] == 120.0
    assert row["netIncome"] == 96.0
    assert row["netCashFlow"] == 45.0


def test_prepare_vn_annual_fundamental_features_maps_insurer_aliases() -> None:
    raw = pl.DataFrame(
        [
            {
                "symbol": "VN:BVH",
                "report_type": "KQKD",
                "period": "2025",
                "fiscal_year": 2025,
                "quarter": None,
                "report_date": date(2026, 3, 31),
                "item_en": "Revenue from insurance premium",
                "value": 1200.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:BVH",
                "report_type": "KQKD",
                "period": "2025",
                "fiscal_year": 2025,
                "quarter": None,
                "report_date": date(2026, 3, 31),
                "item_en": "7. Total net revenue from insurance business",
                "value": 1000.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:BVH",
                "report_type": "KQKD",
                "period": "2025",
                "fiscal_year": 2025,
                "quarter": None,
                "report_date": date(2026, 3, 31),
                "item_en": "29. Total profit before tax",
                "value": 180.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:BVH",
                "report_type": "KQKD",
                "period": "2025",
                "fiscal_year": 2025,
                "quarter": None,
                "report_date": date(2026, 3, 31),
                "item_en": "34. Profit after tax",
                "value": 144.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:BVH",
                "report_type": "CDKT",
                "period": "2025",
                "fiscal_year": 2025,
                "quarter": None,
                "report_date": date(2026, 3, 31),
                "item_en": "A.  LIABILITIES",
                "value": 2500.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:BVH",
                "report_type": "CDKT",
                "period": "2025",
                "fiscal_year": 2025,
                "quarter": None,
                "report_date": date(2026, 3, 31),
                "item_en": "I. Owner's equity",
                "value": 1700.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:BVH",
                "report_type": "CDKT",
                "period": "2025",
                "fiscal_year": 2025,
                "quarter": None,
                "report_date": date(2026, 3, 31),
                "item_en": "II. Long-term liabilities",
                "value": 400.0,
                "frequency": "annual",
            },
        ]
    )

    featured = prepare_vn_annual_fundamental_features(raw)
    row = featured.row(0, named=True)

    assert row["revenue"] == 1200.0
    assert row["netRevenue"] == 1000.0
    assert row["profitBeforeTax"] == 180.0
    assert row["netIncome"] == 144.0
    assert row["totalLiabilities"] == 2500.0
    assert row["totalEquity"] == 1700.0
    assert row["longTermLiabilities"] == 400.0


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


def test_prepare_vn_annual_fundamental_features_dedups_derives_and_adds_yoy() -> None:
    raw = pl.DataFrame(
        [
            {
                "symbol": "VN:ACB",
                "report_type": "KQKD",
                "period": "2022",
                "fiscal_year": 2022,
                "quarter": None,
                "report_date": date(2023, 2, 15),
                "item_en": "Sales",
                "value": 100.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:ACB",
                "report_type": "KQKD",
                "period": "2022",
                "fiscal_year": 2022,
                "quarter": None,
                "report_date": date(2023, 2, 28),
                "item_en": "Sales",
                "value": 110.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:ACB",
                "report_type": "KQKD",
                "period": "2022",
                "fiscal_year": 2022,
                "quarter": None,
                "report_date": date(2023, 2, 28),
                "item_en": "Net profit/(loss) after tax",
                "value": 10.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:ACB",
                "report_type": "CDKT",
                "period": "2022",
                "fiscal_year": 2022,
                "quarter": None,
                "report_date": date(2023, 2, 28),
                "item_en": "Total Assets",
                "value": 250.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:ACB",
                "report_type": "CDKT",
                "period": "2022",
                "fiscal_year": 2022,
                "quarter": None,
                "report_date": date(2023, 2, 28),
                "item_en": "Liabilities",
                "value": 100.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:ACB",
                "report_type": "KQKD",
                "period": "2023",
                "fiscal_year": 2023,
                "quarter": None,
                "report_date": date(2024, 2, 29),
                "item_en": "Sales",
                "value": 132.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:ACB",
                "report_type": "KQKD",
                "period": "2023",
                "fiscal_year": 2023,
                "quarter": None,
                "report_date": date(2024, 2, 29),
                "item_en": "Net profit/(loss) after tax",
                "value": 12.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:ACB",
                "report_type": "CDKT",
                "period": "2023",
                "fiscal_year": 2023,
                "quarter": None,
                "report_date": date(2024, 2, 29),
                "item_en": "Total Assets",
                "value": 300.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:ACB",
                "report_type": "CDKT",
                "period": "2023",
                "fiscal_year": 2023,
                "quarter": None,
                "report_date": date(2024, 2, 29),
                "item_en": "Liabilities",
                "value": 120.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:ACB",
                "report_type": "KQKD",
                "period": "2023-Q1",
                "fiscal_year": 2023,
                "quarter": 1,
                "report_date": date(2023, 5, 1),
                "item_en": "Sales",
                "value": 999.0,
                "frequency": "quarterly",
            },
        ]
    )

    featured = prepare_vn_annual_fundamental_features(raw)

    assert featured.height == 2
    first = featured.row(0, named=True)
    second = featured.row(1, named=True)

    assert first["fiscal_year"] == 2022
    assert first["revenue"] == 110.0
    assert first["period_end"] == date(2022, 12, 31)
    assert first["available_from"] == date(2023, 5, 1)
    assert first["report_date"] == date(2023, 2, 28)
    assert first["debtToAssetsRatio"] == 40.0
    assert first["revenue_yoy"] is None

    assert second["fiscal_year"] == 2023
    assert second["revenue"] == 132.0
    assert second["report_date"] == date(2024, 2, 29)
    assert second["netIncome"] == 12.0
    assert second["totalAssets"] == 300.0
    assert second["debtToAssetsRatio"] == 40.0
    assert second["revenue_yoy"] == 0.2
    assert second["netIncome_yoy"] == 0.2
    assert second["totalAssets_yoy"] == 0.2


def test_prepare_vn_annual_fundamental_features_auto_adds_yoy_for_numeric_columns() -> None:
    raw = pl.DataFrame(
        [
            {
                "symbol": "VN:ANV",
                "report_type": "CSTC",
                "period": "2022",
                "fiscal_year": 2022,
                "quarter": None,
                "report_date": date(2023, 2, 20),
                "item_en": "pb",
                "value": 2.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:ANV",
                "report_type": "CDKT",
                "period": "2022",
                "fiscal_year": 2022,
                "quarter": None,
                "report_date": date(2023, 2, 20),
                "item_en": "Total Assets",
                "value": 200.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:ANV",
                "report_type": "CDKT",
                "period": "2022",
                "fiscal_year": 2022,
                "quarter": None,
                "report_date": date(2023, 2, 20),
                "item_en": "Liabilities",
                "value": 80.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:ANV",
                "report_type": "CDKT",
                "period": "2022",
                "fiscal_year": 2022,
                "quarter": None,
                "report_date": date(2023, 2, 20),
                "item_en": "Cash and cash equivalents",
                "value": 40.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:ANV",
                "report_type": "CSTC",
                "period": "2023",
                "fiscal_year": 2023,
                "quarter": None,
                "report_date": date(2024, 2, 20),
                "item_en": "pb",
                "value": 2.5,
                "frequency": "annual",
            },
            {
                "symbol": "VN:ANV",
                "report_type": "CDKT",
                "period": "2023",
                "fiscal_year": 2023,
                "quarter": None,
                "report_date": date(2024, 2, 20),
                "item_en": "Total Assets",
                "value": 250.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:ANV",
                "report_type": "CDKT",
                "period": "2023",
                "fiscal_year": 2023,
                "quarter": None,
                "report_date": date(2024, 2, 20),
                "item_en": "Liabilities",
                "value": 100.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:ANV",
                "report_type": "CDKT",
                "period": "2023",
                "fiscal_year": 2023,
                "quarter": None,
                "report_date": date(2024, 2, 20),
                "item_en": "Cash and cash equivalents",
                "value": 50.0,
                "frequency": "annual",
            },
        ]
    )

    featured = prepare_vn_annual_fundamental_features(raw)

    assert "priceToBookRatio_yoy" in featured.columns
    assert "totalLiabilities_yoy" in featured.columns
    assert "debtToAssetsRatio_yoy" in featured.columns
    assert "cashAndCashEquivalentsToAssets_yoy" in featured.columns
    assert "report_date_yoy" not in featured.columns

    first = featured.row(0, named=True)
    second = featured.row(1, named=True)

    assert first["priceToBookRatio_yoy"] is None
    assert first["totalLiabilities_yoy"] is None
    assert second["priceToBookRatio_yoy"] == 0.25
    assert second["totalLiabilities_yoy"] == 0.25
    assert second["debtToAssetsRatio_yoy"] == 0.0
    assert second["cashAndCashEquivalentsToAssets_yoy"] == 0.0


def test_prepare_vn_quarterly_fundamental_features_dedups_derives_and_adds_q_yoy() -> None:
    raw = pl.DataFrame(
        [
            {
                "symbol": "VN:ACB",
                "report_type": "KQKD",
                "period": "2022-Q1",
                "fiscal_year": 2022,
                "quarter": 1,
                "report_date": date(2022, 4, 20),
                "item_en": "Sales",
                "value": 100.0,
                "frequency": "quarterly",
            },
            {
                "symbol": "VN:ACB",
                "report_type": "KQKD",
                "period": "2022-Q2",
                "fiscal_year": 2022,
                "quarter": 2,
                "report_date": date(2022, 7, 20),
                "item_en": "Sales",
                "value": 120.0,
                "frequency": "quarterly",
            },
            {
                "symbol": "VN:ACB",
                "report_type": "KQKD",
                "period": "2022-Q3",
                "fiscal_year": 2022,
                "quarter": 3,
                "report_date": date(2022, 10, 20),
                "item_en": "Sales",
                "value": 130.0,
                "frequency": "quarterly",
            },
            {
                "symbol": "VN:ACB",
                "report_type": "KQKD",
                "period": "2022-Q4",
                "fiscal_year": 2022,
                "quarter": 4,
                "report_date": date(2023, 1, 20),
                "item_en": "Sales",
                "value": 140.0,
                "frequency": "quarterly",
            },
            {
                "symbol": "VN:ACB",
                "report_type": "KQKD",
                "period": "2023-Q1",
                "fiscal_year": 2023,
                "quarter": 1,
                "report_date": date(2023, 4, 15),
                "item_en": "Sales",
                "value": 145.0,
                "frequency": "quarterly",
            },
            {
                "symbol": "VN:ACB",
                "report_type": "KQKD",
                "period": "2023-Q1",
                "fiscal_year": 2023,
                "quarter": 1,
                "report_date": date(2023, 4, 30),
                "item_en": "Sales",
                "value": 150.0,
                "frequency": "quarterly",
            },
            {
                "symbol": "VN:ACB",
                "report_type": "CDKT",
                "period": "2023-Q1",
                "fiscal_year": 2023,
                "quarter": 1,
                "report_date": date(2023, 4, 30),
                "item_en": "Total Assets",
                "value": 300.0,
                "frequency": "quarterly",
            },
            {
                "symbol": "VN:ACB",
                "report_type": "CDKT",
                "period": "2023-Q1",
                "fiscal_year": 2023,
                "quarter": 1,
                "report_date": date(2023, 4, 30),
                "item_en": "Liabilities",
                "value": 120.0,
                "frequency": "quarterly",
            },
            {
                "symbol": "VN:ACB",
                "report_type": "KQKD",
                "period": "2023",
                "fiscal_year": 2023,
                "quarter": None,
                "report_date": date(2024, 2, 29),
                "item_en": "Sales",
                "value": 999.0,
                "frequency": "annual",
            },
        ]
    )

    featured = prepare_vn_quarterly_fundamental_features(
        raw,
        qoq_metrics=["revenue", "totalAssets"],
    )

    assert featured.height == 5
    latest = featured.row(-1, named=True)

    assert latest["fiscal_year"] == 2023
    assert latest["quarter"] == 1
    assert latest["revenue"] == 150.0
    assert latest["period_end"] == date(2023, 3, 31)
    assert latest["available_from"] == date(2023, 8, 1)
    assert latest["report_date"] == date(2023, 4, 30)
    assert latest["debtToAssetsRatio"] == 40.0
    assert latest["revenue_q_yoy"] == 0.5
    assert latest["totalAssets_q_yoy"] is None


def test_add_derived_fundamental_metrics_fills_null_existing_ratios() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["VN:ACB"],
            "fiscal_year": [2024],
            "totalLiabilities": [900.0],
            "totalAssets": [1000.0],
            "debtToAssetsRatio": [None],
            "totalLiabilitiesToAssets": [None],
        }
    )

    featured = add_derived_fundamental_metrics(frame)
    row = featured.row(0, named=True)

    assert row["debtToAssetsRatio"] == 90.0
    assert row["totalLiabilitiesToAssets"] == 0.9


def test_audit_annual_fundamental_availability_classifies_source_mapper_and_yoy_gaps() -> None:
    raw = pl.DataFrame(
        [
            {
                "symbol": "VN:ACB",
                "report_type": "KQKD",
                "period": "2023",
                "fiscal_year": 2023,
                "quarter": None,
                "report_date": date(2024, 2, 28),
                "item_en": "Sales",
                "value": 100.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:ACB",
                "report_type": "KQKD",
                "period": "2024",
                "fiscal_year": 2024,
                "quarter": None,
                "report_date": date(2025, 2, 28),
                "item_en": "Total Operating Income before provisions",
                "value": 120.0,
                "frequency": "annual",
            },
            {
                "symbol": "VN:ACB",
                "report_type": "CDKT",
                "period": "2024",
                "fiscal_year": 2024,
                "quarter": None,
                "report_date": date(2025, 2, 28),
                "item_en": "Liabilities",
                "value": 80.0,
                "frequency": "annual",
            },
        ]
    )

    processed = prepare_vn_annual_fundamental_features(raw)
    audit = audit_annual_fundamental_availability(
        raw,
        processed,
        columns=[
            "revenue",
            "totalAssets",
            "debtToAssetsRatio",
            "revenue_yoy",
        ],
    )

    revenue_2024 = audit.filter(
        (pl.col("symbol") == "VN:ACB")
        & (pl.col("fiscal_year") == 2024)
        & (pl.col("column") == "revenue")
    ).row(0, named=True)
    total_assets_2024 = audit.filter(
        (pl.col("symbol") == "VN:ACB")
        & (pl.col("fiscal_year") == 2024)
        & (pl.col("column") == "totalAssets")
    ).row(0, named=True)
    derived_2024 = audit.filter(
        (pl.col("symbol") == "VN:ACB")
        & (pl.col("fiscal_year") == 2024)
        & (pl.col("column") == "debtToAssetsRatio")
    ).row(0, named=True)
    yoy_2023 = audit.filter(
        (pl.col("symbol") == "VN:ACB")
        & (pl.col("fiscal_year") == 2023)
        & (pl.col("column") == "revenue_yoy")
    ).row(0, named=True)
    yoy_2024 = audit.filter(
        (pl.col("symbol") == "VN:ACB")
        & (pl.col("fiscal_year") == 2024)
        & (pl.col("column") == "revenue_yoy")
    ).row(0, named=True)

    assert revenue_2024["root_cause"] == "raw item exists but alias is unmapped"
    assert total_assets_2024["root_cause"] == "source missing in raw annual data"
    assert derived_2024["root_cause"] == "derived metric missing because prerequisite columns are missing"
    assert yoy_2023["root_cause"] == "YoY missing because prior year is unavailable"
    assert yoy_2024["root_cause"] == "YoY missing because current or prior value is null/zero"

    summary = summarize_annual_fundamental_availability(audit)
    revenue_summary = summary.filter(pl.col("column") == "revenue").row(0, named=True)

    assert revenue_summary["rows"] == 2
    assert revenue_summary["available_rows"] == 1
    assert revenue_summary["null_rows"] == 1
    assert revenue_summary["coverage_pct"] == 50.0
    assert revenue_summary["root_causes"] == ["raw item exists but alias is unmapped"]


def test_audit_annual_fundamental_availability_classifies_vci_policy_exclusions() -> None:
    raw = pl.DataFrame(
        [
            {
                "symbol": "VN:DSE",
                "report_type": "CSTC",
                "period": "2022",
                "fiscal_year": 2022,
                "quarter": None,
                "report_date": date(2023, 3, 31),
                "item_en": "ROE",
                "value": 12.5,
                "frequency": "annual",
            },
            {
                "symbol": "VN:SSI",
                "report_type": "KQKD",
                "period": "2022",
                "fiscal_year": 2022,
                "quarter": None,
                "report_date": date(2023, 3, 31),
                "item_en": "Revenue from securities business (01->11)",
                "value": 500.0,
                "frequency": "annual",
            },
        ]
    )

    processed = prepare_vn_annual_fundamental_features(raw)
    audit = audit_annual_fundamental_availability(
        raw,
        processed,
        columns=["returnOnEquity", "revenue"],
    )

    dse_roe = audit.filter(
        (pl.col("symbol") == "VN:DSE")
        & (pl.col("fiscal_year") == 2022)
        & (pl.col("column") == "returnOnEquity")
    ).row(0, named=True)
    ssi_revenue = audit.filter(
        (pl.col("symbol") == "VN:SSI")
        & (pl.col("fiscal_year") == 2022)
        & (pl.col("column") == "revenue")
    ).row(0, named=True)

    assert dse_roe["root_cause"] == "raw item exists but is excluded by VCI-only policy"
    assert ssi_revenue["root_cause"] == "raw item exists but is excluded by VCI-only policy"
