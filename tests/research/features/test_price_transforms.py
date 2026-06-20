from __future__ import annotations

from datetime import date
from math import isclose, log

import polars as pl

from qts.core.registry import Registry
from qts.research.features.transforms.price import (
    average_price_transform,
    log_price_transform,
)


def _price_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "symbol": ["AAPL", "AAPL"],
            "open": [99.0, 100.0],
            "high": [103.0, 106.0],
            "low": [97.0, 100.0],
            "close": [101.0, 103.0],
            "volume": [1_000.0, 1_100.0],
        }
    )


def test_average_price_transform_appends_typical_price_without_mutating_ohlcv():
    original = _price_frame()
    result = average_price_transform(original)

    assert "avg_price" in result.columns
    for column in original.columns:
        assert result[column].to_list() == original[column].to_list()
    assert result["avg_price"].to_list() == [
        (103.0 + 97.0 + 101.0) / 3.0,
        (106.0 + 100.0 + 103.0) / 3.0,
    ]


def test_log_price_transform_appends_log_of_selected_price_column():
    with_average = average_price_transform(_price_frame(), output_column="typical_price")
    result = log_price_transform(
        with_average,
        price_column="typical_price",
        output_column="typical_log_price",
    )

    assert "typical_log_price" in result.columns
    assert isclose(result["typical_log_price"][0], log(result["typical_price"][0]))
    assert isclose(result["typical_log_price"][1], log(result["typical_price"][1]))


def test_log_price_transform_nulls_non_positive_prices():
    frame = pl.DataFrame({"price": [10.0, 0.0, -1.0]})
    result = log_price_transform(frame, price_column="price")

    assert isclose(result["log_price"][0], log(10.0))
    assert result["log_price"][1] is None
    assert result["log_price"][2] is None


def test_price_transforms_register_with_registry_after_module_import():
    assert Registry.get_transform("average_price") is average_price_transform
    assert Registry.get_transform("log_price") is log_price_transform
