"""Tests for strategies.factor.portfolio_construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qts.research.strategies.factor.portfolio_construction import (
    apply_correlation_penalty,
    apply_factor_neutrality,
    apply_liquidity_cap,
    apply_volatility_cap,
    apply_weight_constraints,
    long_short_equal_weight_portfolio,
    long_short_exponential_weight_portfolio,
    long_short_hrp_portfolio,
    long_short_inverse_volatility_portfolio,
    long_short_kelly_portfolio,
    long_short_mean_variance_portfolio,
    long_short_min_variance_portfolio,
    long_short_risk_parity_portfolio,
    long_short_volatility_target_portfolio,
)


# ---- long_short_equal_weight_portfolio ----

def test_equal_weight_one_long(predictions):
    result = long_short_equal_weight_portfolio(predictions, num_long_positions=1, num_short_positions=0)
    assert len(result) == 1
    assert list(result.values())[0] == pytest.approx(1.0)
    assert all(v > 0 for v in result.values())


def test_equal_weight_one_long_one_short(predictions):
    result = long_short_equal_weight_portfolio(predictions, num_long_positions=1, num_short_positions=1)
    positives = [v for v in result.values() if v > 0]
    negatives = [v for v in result.values() if v < 0]
    assert len(positives) == 1
    assert len(negatives) == 1


def test_equal_weight_negative_positions_raises(predictions):
    with pytest.raises(ValueError):
        long_short_equal_weight_portfolio(predictions, num_long_positions=-1)


def test_equal_weight_empty_predictions():
    result = long_short_equal_weight_portfolio(pd.Series(dtype=float), num_long_positions=1)
    assert result == {}


# ---- long_short_inverse_volatility_portfolio ----

def test_inverse_vol_two_longs(predictions, history_pd):
    result = long_short_inverse_volatility_portfolio(predictions, num_long_positions=2, history_df=history_pd)
    positives = {k: v for k, v in result.items() if v > 0}
    assert len(positives) == 2
    assert all(v > 0 for v in positives.values())
    assert sum(positives.values()) == pytest.approx(1.0, abs=1e-6)


def test_inverse_vol_no_history_no_exception(predictions):
    result = long_short_inverse_volatility_portfolio(predictions, num_long_positions=2, history_df=None)
    assert isinstance(result, dict)


# ---- long_short_risk_parity_portfolio ----

def test_risk_parity_positive_weights(predictions, history_pd):
    result = long_short_risk_parity_portfolio(predictions, num_long_positions=2, history_df=history_pd)
    positives = {k: v for k, v in result.items() if v > 0}
    assert len(positives) >= 1
    assert all(v > 0 for v in positives.values())


def test_risk_parity_no_history_no_exception(predictions):
    result = long_short_risk_parity_portfolio(predictions, num_long_positions=2, history_df=None)
    assert isinstance(result, dict)


# ---- long_short_mean_variance_portfolio ----

def test_mean_variance_returns_dict(predictions, history_pd):
    result = long_short_mean_variance_portfolio(predictions, num_long_positions=2, history_df=history_pd)
    assert isinstance(result, dict)
    for k in result:
        assert k in predictions.index


def test_mean_variance_no_history_no_exception(predictions):
    result = long_short_mean_variance_portfolio(predictions, num_long_positions=2, history_df=None)
    assert isinstance(result, dict)


# ---- long_short_min_variance_portfolio ----

def test_min_variance_long_weights_non_negative(predictions, history_pd):
    result = long_short_min_variance_portfolio(predictions, num_long_positions=2, history_df=history_pd)
    positives = {k: v for k, v in result.items() if v > 0}
    assert all(v >= 0 for v in positives.values())


# ---- long_short_hrp_portfolio ----

def test_hrp_valid_result(predictions, history_pd):
    scipy = pytest.importorskip("scipy")
    result = long_short_hrp_portfolio(predictions, num_long_positions=2, history_df=history_pd)
    assert isinstance(result, dict)


def test_hrp_no_history_no_exception(predictions):
    pytest.importorskip("scipy")
    result = long_short_hrp_portfolio(predictions, num_long_positions=2, history_df=None)
    assert isinstance(result, dict)


# ---- long_short_kelly_portfolio ----

def test_kelly_max_weight_respected(predictions, history_pd):
    result = long_short_kelly_portfolio(predictions, num_long_positions=2, history_df=history_pd,
                                        max_abs_weight=0.3)
    assert all(abs(v) <= 0.3 + 1e-9 for v in result.values())


# ---- apply_weight_constraints ----

def test_apply_weight_constraints_max_weight():
    weights = {"A": 0.5, "B": -0.3}
    result = apply_weight_constraints(weights, max_weight=0.1)
    assert all(abs(v) <= 0.1 + 1e-9 for v in result.values())


def test_apply_weight_constraints_sector():
    weights = {"A": 0.15, "B": 0.15, "C": 0.1}
    sector_map = {"A": "tech", "B": "tech", "C": "energy"}
    result = apply_weight_constraints(weights, sector_map=sector_map, sector_max_weight=0.2)
    tech_total = abs(result.get("A", 0)) + abs(result.get("B", 0))
    assert tech_total <= 0.2 + 1e-9


# ---- apply_factor_neutrality ----

def test_apply_factor_neutrality_removes_exposure():
    weights = {"A": 0.3, "B": -0.2, "C": 0.1}
    exposures = pd.DataFrame({"factor1": [1.0, -1.0, 0.5]}, index=["A", "B", "C"])
    result = apply_factor_neutrality(weights, exposures)
    w = pd.Series(result)
    residual = float((w * exposures["factor1"].reindex(w.index).fillna(0)).sum())
    assert abs(residual) < 1e-4


# ---- apply_volatility_cap ----

def test_apply_volatility_cap_with_history(predictions, history_pd):
    weights = {"A": 0.5, "B": 0.5, "C": 0.3}
    result = apply_volatility_cap(weights, history_df=history_pd, max_position_vol=0.05)
    assert isinstance(result, dict)


def test_apply_volatility_cap_no_history_unchanged():
    weights = {"A": 0.5, "B": 0.3}
    result = apply_volatility_cap(weights, history_df=None, max_position_vol=0.05)
    assert result == weights


# ---- apply_correlation_penalty ----

def test_apply_correlation_penalty_with_history(predictions, history_pd):
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    result = apply_correlation_penalty(weights, history_df=history_pd)
    assert set(result.keys()) == set(weights.keys())
    assert all(abs(v) > 0 for v in result.values())


def test_apply_correlation_penalty_no_history_unchanged():
    weights = {"A": 0.5, "B": 0.3}
    result = apply_correlation_penalty(weights, history_df=None)
    assert result == weights


# ---- apply_liquidity_cap ----

def test_apply_liquidity_cap_no_capital_base_unchanged():
    weights = {"A": 0.5, "B": 0.3}
    result = apply_liquidity_cap(weights, capital_base=None)
    assert result == weights


def test_apply_liquidity_cap_with_history(history_pd):
    weights = {"A": 0.8, "B": 0.8, "C": 0.8}
    result = apply_liquidity_cap(
        weights, history_df=history_pd, capital_base=1_000_000, max_adv_fraction=0.01
    )
    assert isinstance(result, dict)
    assert all(abs(v) <= 0.8 + 1e-9 for v in result.values())
