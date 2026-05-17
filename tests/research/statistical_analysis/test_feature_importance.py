"""Tests for statistical_analysis.feature_importance."""

from __future__ import annotations

import polars as pl
import pytest

pytest.importorskip("xgboost")

from qts.research.statistical_analysis.feature_importance import get_feature_importance


@pytest.fixture
def feature_df(rng):
    """200-row Polars frame with 5 features and a target."""
    import numpy as np

    n = 200
    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)
    f3 = rng.normal(0, 1, n)
    f4 = rng.normal(0, 1, n)
    f5 = rng.normal(0, 1, n)
    target = 0.3 * f1 - 0.2 * f2 + rng.normal(0, 0.1, n)
    return pl.DataFrame({
        "f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5, "fwd_ret": target,
    })


def test_get_feature_importance_columns(feature_df):
    result = get_feature_importance(feature_df, "fwd_ret", ["f1", "f2", "f3", "f4", "f5"])
    assert set(result.columns) == {"feature", "importance"}


def test_get_feature_importance_sorted_descending(feature_df):
    result = get_feature_importance(feature_df, "fwd_ret", ["f1", "f2", "f3", "f4", "f5"])
    importances = result["importance"].to_list()
    assert importances == sorted(importances, reverse=True)


def test_get_feature_importance_default_cols(feature_df):
    result = get_feature_importance(feature_df, "fwd_ret")
    assert result.height >= 1  # at least one feature with non-zero gain


def test_get_feature_importance_empty_df():
    empty = pl.DataFrame(schema={"f1": pl.Float64, "f2": pl.Float64, "fwd_ret": pl.Float64})
    result = get_feature_importance(empty, "fwd_ret", ["f1", "f2"])
    assert result.is_empty()
    assert set(result.columns) == {"feature", "importance"}


def test_get_feature_importance_empty_feature_cols(feature_df):
    result = get_feature_importance(feature_df, "fwd_ret", [])
    assert result.is_empty()
    assert set(result.columns) == {"feature", "importance"}
