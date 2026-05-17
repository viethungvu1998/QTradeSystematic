from __future__ import annotations

from datetime import date, timedelta
from math import isnan

import polars as pl
import pytest

from qts.core.registry import Registry


def _indicator_fixture() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol, base_price in (("AAPL", 100.0), ("MSFT", 200.0)):
        for index in range(150):
            current = date(2023, 1, 1) + timedelta(days=index)
            close = base_price + (index * 0.75) + ((index % 7) - 3) * 0.2
            rows.append(
                {
                    "date": current,
                    "symbol": symbol,
                    "open": close - 0.4,
                    "high": close + 0.8,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1_000.0 + (index * 15) + (50 if symbol == "MSFT" else 0),
                }
            )
    return pl.DataFrame(rows)


@pytest.mark.parametrize(
    ("key", "params", "expected_columns", "warmup"),
    [
        ("rsi", {"periods": [14]}, ["rsi_14"], 14),
        ("roc", {"periods": [1, 5, 21]}, ["roc_1", "roc_5", "roc_21"], 21),
        ("macd", {"fast": 12, "slow": 26, "signal": 9}, ["macd_line", "macd_signal", "macd_hist"], 35),
        ("adx", {"period": 14}, ["adx_14"], 28),
        ("atr", {"periods": [14]}, ["atr_14"], 14),
        (
            "bollinger",
            {"window": 20, "n_std": 2.0},
            ["bb_upper_20", "bb_mid_20", "bb_lower_20"],
            20,
        ),
        ("hist_vol", {"periods": [20, 60]}, ["hist_vol_20", "hist_vol_60"], 61),
        ("obv", {}, ["obv"], 0),
        ("volume_ratio", {"window": 20}, ["vol_ratio_20"], 20),
        ("zscore", {"windows": [21, 63]}, ["zscore_21", "zscore_63"], 63),
    ],
)
def test_registered_indicator_features_append_expected_columns(
    key: str,
    params: dict[str, object],
    expected_columns: list[str],
    warmup: int,
):
    fixture = _indicator_fixture()
    feature = Registry.get_feature(key)(**params)
    result = feature.fit_transform(fixture)

    for column in fixture.columns:
        assert column in result.columns
    for column in expected_columns:
        assert column in result.columns

    for partition in result.partition_by("symbol", as_dict=False):
        for column in expected_columns:
            values = partition[column].to_list()[warmup:]
            assert all(value is not None for value in values)
            assert all(not (isinstance(value, float) and isnan(value)) for value in values)
