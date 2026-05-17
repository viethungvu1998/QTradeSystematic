"""Tests for features.preprocessor — flag_anomalies and remove_flagged_symbols."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from qts.research.features.preprocessor import (
    flag_anomalies,
    preprocess_ohlcv,
    remove_flagged_symbols,
)

FLAG_COLS = {"flag_large_gap", "flag_high_price", "flag_low_volume", "flag_high_volume"}


def _make_ohlcv(n_days: int, symbol: str, base_date: date | None = None, close_override: list | None = None) -> list[dict]:
    base = base_date or date(2023, 1, 2)
    rows = []
    close = 100.0
    for i in range(n_days):
        c = close_override[i] if close_override else close
        rows.append({
            "date": base + timedelta(days=i),
            "symbol": symbol,
            "open": c * 0.999,
            "high": c * 1.005,
            "low": c * 0.995,
            "close": c,
            "volume": 1_000_000.0,
        })
        close *= 1.001
    return rows


@pytest.fixture
def clean_df():
    rows = _make_ohlcv(50, "A") + _make_ohlcv(50, "B")
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


@pytest.fixture
def gap_df():
    # X has a 10-day gap after day 20
    rows_x = []
    for i in range(20):
        rows_x.append({
            "date": date(2023, 1, 2) + timedelta(days=i),
            "symbol": "X",
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000.0,
        })
    for i in range(30):
        rows_x.append({
            "date": date(2023, 1, 2) + timedelta(days=20 + 10 + i),
            "symbol": "X",
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000.0,
        })
    rows_clean = _make_ohlcv(50, "Y")
    return pl.DataFrame(rows_x + rows_clean).with_columns(pl.col("date").cast(pl.Date))


@pytest.fixture
def anomaly_df(rng):
    closes = [100.0] * 49 + [100.0 + 15 * 100.0]  # last row is 15σ spike
    rows = _make_ohlcv(50, "Y", close_override=closes) + _make_ohlcv(50, "Z")
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


@pytest.fixture
def low_volume_df():
    rows_w = _make_ohlcv(50, "W")
    # Symbol Z: 6 of 50 days with zero volume and close (12%)
    rows_z = []
    base = date(2023, 1, 2)
    for i in range(50):
        v = 0.0 if i < 6 else 1_000_000.0
        c = 0.01 if i < 6 else 100.0
        rows_z.append({
            "date": base + timedelta(days=i),
            "symbol": "Z",
            "open": c, "high": c, "low": c, "close": c, "volume": v,
        })
    return pl.DataFrame(rows_w + rows_z).with_columns(pl.col("date").cast(pl.Date))


# ---- flag_anomalies ----

def test_flag_anomalies_adds_all_columns(clean_df):
    result = flag_anomalies(clean_df)
    assert FLAG_COLS.issubset(set(result.columns))


def test_flag_anomalies_no_rows_removed(clean_df):
    result = flag_anomalies(clean_df)
    assert result.height == clean_df.height


def test_flag_anomalies_clean_all_false(clean_df):
    result = flag_anomalies(clean_df, max_gap_days=7, volatility_threshold=5.0)
    for col in FLAG_COLS:
        assert not result[col].any(), f"{col} unexpectedly True on clean data"


def test_flag_anomalies_gap_flag(gap_df):
    result = flag_anomalies(gap_df, max_gap_days=7)
    x_flagged = result.filter((pl.col("symbol") == "X") & pl.col("flag_large_gap"))
    # Exactly the row after the 10-day gap should be flagged
    assert x_flagged.height == 1


def test_flag_anomalies_gap_flag_not_on_clean_symbol(gap_df):
    result = flag_anomalies(gap_df, max_gap_days=7)
    y_flagged = result.filter((pl.col("symbol") == "Y") & pl.col("flag_large_gap"))
    assert y_flagged.is_empty()


def test_flag_anomalies_high_price(anomaly_df):
    result = flag_anomalies(anomaly_df, volatility_threshold=5.0)
    y_flagged = result.filter((pl.col("symbol") == "Y") & pl.col("flag_high_price"))
    assert y_flagged.height >= 1


def test_preprocess_ohlcv_unchanged_by_enhancement(clean_df):
    # preprocess_ohlcv must work independently of flag_anomalies
    result = preprocess_ohlcv(clean_df, min_trading_days=10)
    assert set(result.columns) >= {"date", "symbol", "open", "high", "low", "close", "volume"}
    assert not any(c in result.columns for c in FLAG_COLS)


# ---- remove_flagged_symbols ----

def test_remove_large_gaps_removes_gapped_symbol(gap_df):
    flagged = flag_anomalies(gap_df, max_gap_days=7)
    result = remove_flagged_symbols(flagged, remove_large_gaps=True)
    assert "X" not in result["symbol"].unique().to_list()
    assert "Y" in result["symbol"].unique().to_list()


def test_remove_anomalies_removes_high_price_symbol(anomaly_df):
    flagged = flag_anomalies(anomaly_df, volatility_threshold=5.0)
    result = remove_flagged_symbols(flagged, remove_anomalies=True)
    assert "Y" not in result["symbol"].unique().to_list()
    assert "Z" in result["symbol"].unique().to_list()


def test_remove_anomalies_false_keeps_symbol(anomaly_df):
    flagged = flag_anomalies(anomaly_df, volatility_threshold=5.0)
    result = remove_flagged_symbols(flagged, remove_anomalies=False)
    assert "Y" in result["symbol"].unique().to_list()


def test_remove_low_volume_threshold(low_volume_df):
    flagged = flag_anomalies(low_volume_df, min_volume_threshold=1.0, min_notional_usd=None)
    # Z has 6/50 = 12% low-volume days → removed; W has 0/50 = 0% → kept
    result = remove_flagged_symbols(flagged, remove_low_volume=True, low_volume_fraction_threshold=0.05)
    symbols = result["symbol"].unique().to_list()
    assert "Z" not in symbols
    assert "W" in symbols
