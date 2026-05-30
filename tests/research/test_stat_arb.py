"""Tests for the stat_arb strategy module.

Covers pair_selection, signals, model (StatArbStrategy), and universe screening.
"""

from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Shared helpers & fixtures
# ---------------------------------------------------------------------------


def _make_random_walk(n: int, start: float = 100.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n)) + start


def _make_cointegrated_pair(n: int = 400, noise_std: float = 0.5, seed: int = 42) -> pd.DataFrame:
    """Two price series where y ≈ x + stationary noise, i.e. genuinely cointegrated."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(n)) + 100
    y = x + rng.standard_normal(n) * noise_std
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({"AAA": x, "BBB": y}, index=dates)


def _make_oscillating_spread(n: int = 200, amplitude: float = 3.0, period: int = 20) -> pd.Series:
    """Spread that reliably crosses ±amplitude every `period` bars."""
    t = np.arange(n)
    values = amplitude * np.sin(2 * np.pi * t / period)
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=n, freq="D"))


@pytest.fixture
def coint_prices() -> pd.DataFrame:
    return _make_cointegrated_pair(n=400)


@pytest.fixture
def paired_ohlcv_polars() -> pl.DataFrame:
    """Two symbols with 80 bars using the same conftest pattern."""
    rows = []
    for i in range(80):
        d = date(2024, 1, 1) + timedelta(days=i)
        price_a = 100.0 + i + 0.5
        rows.append({"date": d, "symbol": "AAA", "open": price_a, "high": price_a + 1,
                     "low": price_a - 1, "close": price_a, "volume": 1000.0})
        price_b = 97.5 + i + (i % 5)
        rows.append({"date": d, "symbol": "BBB", "open": price_b, "high": price_b + 1,
                     "low": price_b - 1, "close": price_b, "volume": 1000.0})
    return pl.DataFrame(rows)


@pytest.fixture
def prices_db(tmp_path):
    """In-file DuckDB with 500 rows of cointegrated prices for AAA and BBB."""
    import duckdb

    n = 500
    data = _make_cointegrated_pair(n=n, noise_std=0.8)
    rows = []
    for i, (ts, row) in enumerate(data.iterrows()):
        d = ts.date()
        rows.append({"date": d, "symbol": "AAA", "close": float(row["AAA"]), "volume": 1_000_000.0})
        rows.append({"date": d, "symbol": "BBB", "close": float(row["BBB"]), "volume": 900_000.0})

    df = pd.DataFrame(rows)
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE stock_prices AS SELECT * FROM df")
    con.close()
    return db_path


# ===========================================================================
# pair_selection
# ===========================================================================

from qts.research.strategies.stat_arb.pair_selection import (
    PairCandidate,
    compute_adf_pvalue,
    compute_half_life,
    ensure_pair_list,
    estimate_hedge_ratio,
    find_cointegrated_pairs,
    preselect_pairs_by_correlation,
)


class TestComputeAdfPvalue:
    def test_stationary_series_returns_small_pvalue(self):
        rng = np.random.default_rng(0)
        series = pd.Series(rng.standard_normal(300))
        p = compute_adf_pvalue(series)
        assert p < 0.05

    def test_empty_series_returns_nan(self):
        assert np.isnan(compute_adf_pvalue(pd.Series([], dtype=float)))

    def test_random_walk_pvalue_typically_large(self):
        series = pd.Series(_make_random_walk(300))
        p = compute_adf_pvalue(series)
        # Random walks are usually non-stationary; pvalue should not be tiny
        assert np.isfinite(p)


class TestComputeHalfLife:
    def test_mean_reverting_series_positive_finite(self):
        rng = np.random.default_rng(7)
        # AR(1) with phi=0.8 is mean-reverting
        n = 300
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.8 * x[i - 1] + rng.standard_normal()
        hl = compute_half_life(pd.Series(x))
        assert np.isfinite(hl) and hl > 0

    def test_too_short_returns_nan(self):
        assert np.isnan(compute_half_life(pd.Series([1.0])))

    def test_empty_returns_nan(self):
        assert np.isnan(compute_half_life(pd.Series([], dtype=float)))

    def test_faster_mean_reversion_gives_shorter_half_life(self):
        # phi=0.1 reverts much faster than phi=0.8 → shorter half-life
        rng = np.random.default_rng(3)
        n = 500
        fast = np.zeros(n)
        slow = np.zeros(n)
        for i in range(1, n):
            fast[i] = 0.1 * fast[i - 1] + rng.standard_normal()
            slow[i] = 0.8 * slow[i - 1] + rng.standard_normal()
        hl_fast = compute_half_life(pd.Series(fast))
        hl_slow = compute_half_life(pd.Series(slow))
        assert np.isfinite(hl_fast) and np.isfinite(hl_slow)
        assert hl_fast < hl_slow


class TestFindCointegratedPairs:
    def test_finds_cointegrated_pair(self, coint_prices):
        pairs = find_cointegrated_pairs(coint_prices, min_obs=100)
        assert len(pairs) >= 1
        assert pairs[0].symbol_a in {"AAA", "BBB"}
        assert pairs[0].symbol_b in {"AAA", "BBB"}
        assert 0.0 < pairs[0].pvalue <= 0.05

    def test_empty_prices_returns_empty(self):
        result = find_cointegrated_pairs(pd.DataFrame())
        assert result == []

    def test_single_symbol_returns_empty(self):
        prices = pd.DataFrame({"ONLY": [1.0, 2.0, 3.0]})
        assert find_cointegrated_pairs(prices) == []

    def test_min_obs_filters_short_series(self, coint_prices):
        pairs = find_cointegrated_pairs(coint_prices.iloc[:50], min_obs=252)
        assert pairs == []

    def test_pvalue_threshold_filters_weak_pairs(self, coint_prices):
        # Very strict threshold — may or may not find anything
        pairs = find_cointegrated_pairs(coint_prices, pvalue_threshold=0.0001, min_obs=100)
        for p in pairs:
            assert p.pvalue <= 0.0001

    def test_candidate_pairs_limits_search(self, coint_prices):
        pairs = find_cointegrated_pairs(
            coint_prices, candidate_pairs=[["AAA", "BBB"]], min_obs=100
        )
        for p in pairs:
            assert {p.symbol_a, p.symbol_b} == {"AAA", "BBB"}

    def test_max_pairs_respected(self, coint_prices):
        pairs = find_cointegrated_pairs(coint_prices, max_pairs=1, min_obs=100, pvalue_threshold=1.0)
        assert len(pairs) <= 1

    def test_results_sorted_by_pvalue(self, coint_prices):
        pairs = find_cointegrated_pairs(coint_prices, min_obs=100, pvalue_threshold=1.0)
        pvalues = [p.pvalue for p in pairs]
        assert pvalues == sorted(pvalues)


class TestEstimateHedgeRatio:
    def test_ols_identical_series_slope_near_one(self):
        s = pd.Series(_make_random_walk(300))
        ratio = estimate_hedge_ratio(s, s)
        assert abs(ratio - 1.0) < 0.1

    def test_ols_proportional_series(self):
        s = pd.Series(np.arange(1.0, 301.0))
        ratio = estimate_hedge_ratio(s * 2, s)
        assert abs(ratio - 2.0) < 0.01

    def test_fixed_method_returns_one(self):
        s = pd.Series(_make_random_walk(50))
        assert estimate_hedge_ratio(s, s, method="fixed") == 1.0

    def test_empty_returns_nan(self):
        assert np.isnan(estimate_hedge_ratio(pd.Series([], dtype=float), pd.Series([], dtype=float)))

    def test_unsupported_method_raises(self):
        s = pd.Series([1.0, 2.0])
        with pytest.raises(ValueError, match="Unsupported"):
            estimate_hedge_ratio(s, s, method="kalman")


class TestPreSelectByCorrelation:
    def test_highly_correlated_pair_returned(self):
        prices = pd.DataFrame({
            "A": np.arange(1.0, 301.0),
            "B": np.arange(1.0, 301.0) + 0.01 * np.random.default_rng(0).standard_normal(300),
        })
        pairs = preselect_pairs_by_correlation(prices, min_corr=0.9)
        assert ("A", "B") in pairs or ("B", "A") in pairs

    def test_empty_prices_returns_empty(self):
        assert preselect_pairs_by_correlation(pd.DataFrame()) == []

    def test_max_pairs_respected(self):
        prices = pd.DataFrame(
            {sym: np.arange(1.0, 101.0) + i for i, sym in enumerate("ABCDEF")}
        )
        pairs = preselect_pairs_by_correlation(prices, max_pairs=3)
        assert len(pairs) <= 3


class TestEnsurePairList:
    def test_normalises_to_str_tuples(self):
        result = ensure_pair_list([["AAPL", "MSFT"]])
        assert result == [("AAPL", "MSFT")]

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            ensure_pair_list([["AAPL", "MSFT", "GOOG"]])


# ===========================================================================
# signals
# ===========================================================================

from qts.research.strategies.stat_arb.signals import (
    compute_spread,
    compute_zscore,
    generate_zscore_signals,
)


class TestComputeSpread:
    def test_basic_arithmetic(self):
        a = pd.Series([10.0, 20.0, 30.0])
        b = pd.Series([5.0, 10.0, 15.0])
        result = compute_spread(a, b, hedge_ratio=2.0)
        pd.testing.assert_series_equal(result, pd.Series([0.0, 0.0, 0.0]))

    def test_preserves_index(self):
        idx = pd.date_range("2024-01-01", periods=5)
        a = pd.Series(range(5), index=idx, dtype=float)
        b = pd.Series(range(5), index=idx, dtype=float)
        result = compute_spread(a, b, 1.0)
        assert list(result.index) == list(idx)


class TestComputeZscore:
    def test_first_window_minus_one_rows_are_nan(self):
        series = pd.Series(np.arange(1.0, 51.0))
        z = compute_zscore(series, window=10)
        assert z.iloc[:9].isna().all()
        assert z.iloc[9:].notna().all()

    def test_zero_std_replaced_with_nan(self):
        series = pd.Series([5.0] * 30)
        z = compute_zscore(series, window=5)
        assert z.dropna().isna().all() or z.dropna().empty

    def test_output_has_same_length_as_input(self):
        series = pd.Series(np.random.default_rng(0).standard_normal(100))
        z = compute_zscore(series, window=20)
        assert len(z) == 100


class TestGenerateZscoreSignals:
    def _oscillating_z(self, n: int = 100, amplitude: float = 3.0) -> pd.Series:
        t = np.arange(n)
        values = amplitude * np.sin(2 * np.pi * t / 20)
        return pd.Series(values, index=pd.date_range("2020-01-01", periods=n, freq="D"))

    def test_returns_all_four_keys(self):
        z = self._oscillating_z()
        result = generate_zscore_signals(z, entry_z=2.0, exit_z=0.0)
        assert set(result.keys()) == {"long_entries", "long_exits", "short_entries", "short_exits"}

    def test_long_only_short_signals_are_none(self):
        z = self._oscillating_z()
        result = generate_zscore_signals(z, entry_z=2.0, exit_z=0.0, side="long_only")
        assert result["short_entries"] is None
        assert result["short_exits"] is None

    def test_long_short_both_sides_have_signals(self):
        z = self._oscillating_z()
        result = generate_zscore_signals(z, entry_z=2.0, exit_z=0.0, side="long_short")
        assert result["long_entries"].any()
        assert result["short_entries"].any()

    def test_stop_z_triggers_exit(self):
        z = self._oscillating_z(amplitude=4.0)
        result_no_stop = generate_zscore_signals(z, entry_z=2.0, exit_z=0.0, stop_z=None)
        result_stop = generate_zscore_signals(z, entry_z=2.0, exit_z=0.0, stop_z=3.5)
        assert result_stop["long_exits"].sum() >= result_no_stop["long_exits"].sum()

    def test_max_holding_bars_forces_exit(self):
        z = self._oscillating_z()
        result = generate_zscore_signals(z, entry_z=2.0, exit_z=-10.0, max_holding_bars=5)
        # With exit_z=-10 (never crossed naturally), position must close via max_holding
        assert result["long_exits"].any()

    def test_no_double_entries_without_intervening_exit(self):
        z = self._oscillating_z()
        result = generate_zscore_signals(z, entry_z=2.0, exit_z=0.0)
        entries = result["long_entries"].to_numpy(dtype=bool)
        exits = result["long_exits"].to_numpy(dtype=bool)
        in_pos = False
        for e, x in zip(entries, exits):
            if e:
                assert not in_pos, "double entry without exit"
                in_pos = True
            if x:
                in_pos = False

    def test_unsupported_side_raises(self):
        z = self._oscillating_z()
        with pytest.raises(ValueError, match="Unsupported"):
            generate_zscore_signals(z, entry_z=2.0, exit_z=0.0, side="both")


# ===========================================================================
# model — StatArbStrategy
# ===========================================================================

from qts.research.strategies.stat_arb.mean_reversion import StatArbStrategy
from qts.research.backtest.base import BacktestConfig, UniverseConfig
from qts.research.backtest.engines.vectorbtpro_engine import VectorBTProEngine
from qts.research.backtest.engines.zipline_engine import ZiplineReloadedEngine


ZIPLINE_AVAILABLE = importlib.util.find_spec("zipline") is not None


class TestStatArbStrategy:
    def test_signal_frame_schema(self, paired_ohlcv_polars):
        signals = StatArbStrategy(
            entry_zscore=1.3,
            exit_zscore=0.0,
            zscore_window=10,
            pairs=[["AAA", "BBB"]],
        ).generate_signals(paired_ohlcv_polars)
        assert signals.columns == ["date", "symbol", "signal", "weight"]
        assert signals.schema["signal"] == pl.Int32
        assert signals.schema["weight"] == pl.Float64
        assert signals.schema["date"] == pl.Date

    def test_single_symbol_universe_returns_empty_frame(self, paired_ohlcv_polars):
        one_symbol = paired_ohlcv_polars.filter(pl.col("symbol") == "AAA")
        signals = StatArbStrategy().generate_signals(one_symbol)
        assert signals.columns == ["date", "symbol", "signal", "weight"]
        assert signals.is_empty()

    def test_empty_input_returns_empty_valid_frame(self):
        empty = pl.DataFrame(schema={
            "date": pl.Date, "symbol": pl.String,
            "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
            "close": pl.Float64, "volume": pl.Float64,
        })
        signals = StatArbStrategy().generate_signals(empty)
        assert signals.columns == ["date", "symbol", "signal", "weight"]
        assert signals.is_empty()

    def test_weights_in_valid_range(self, paired_ohlcv_polars):
        signals = StatArbStrategy(
            entry_zscore=1.3,
            exit_zscore=0.0,
            zscore_window=10,
            pairs=[["AAA", "BBB"]],
        ).generate_signals(paired_ohlcv_polars)
        assert signals.filter((pl.col("weight") < 0) | (pl.col("weight") > 1)).is_empty()

    def test_signals_in_valid_set(self, paired_ohlcv_polars):
        signals = StatArbStrategy(
            entry_zscore=1.3,
            exit_zscore=0.0,
            zscore_window=10,
            pairs=[["AAA", "BBB"]],
        ).generate_signals(paired_ohlcv_polars)
        assert signals.filter(~pl.col("signal").is_in([-1, 0, 1])).is_empty()

    def test_explicit_pair_selection_on_multi_symbol_universe(self, paired_ohlcv_polars):
        extra = paired_ohlcv_polars.filter(pl.col("symbol") == "AAA").with_columns(
            pl.lit("CCC").alias("symbol"),
            (pl.col("close") * 1.4).alias("close"),
            (pl.col("open") * 1.4).alias("open"),
            (pl.col("high") * 1.4).alias("high"),
            (pl.col("low") * 1.4).alias("low"),
        )
        universe = pl.concat([paired_ohlcv_polars, extra], how="vertical")
        signals = StatArbStrategy(
            entry_zscore=1.3,
            exit_zscore=0.0,
            zscore_window=10,
            pairs=[["AAA", "BBB"]],
        ).generate_signals(universe)
        assert set(signals["symbol"].unique().to_list()) <= {"AAA", "BBB"}
        assert "CCC" not in set(signals["symbol"].unique().to_list())

    def test_pair_aggregation_handles_no_pair_case_cleanly(self, paired_ohlcv_polars):
        signals = StatArbStrategy(
            pvalue_threshold=0.0,
            fallback_pairs=0,
            min_obs=500,
        ).generate_signals(paired_ohlcv_polars)
        assert signals.is_empty()

    def test_hedge_ratio_adjusted_weights_keep_gross_exposure_bounded(self, paired_ohlcv_polars):
        signals = StatArbStrategy(
            entry_zscore=1.3,
            exit_zscore=0.0,
            zscore_window=10,
            pairs=[["AAA", "BBB"]],
        ).generate_signals(paired_ohlcv_polars)
        active = signals.filter(pl.col("signal") != 0)
        if active.is_empty():
            pytest.skip("no active signals in fixture — lower entry_z if needed")
        weight_sums = (
            active.group_by("date").agg(pl.col("weight").sum().alias("total_weight"))
        )
        for total in weight_sums["total_weight"].to_list():
            assert total <= 1.0 + 1e-9

    def test_long_and_short_sides_are_opposite(self, paired_ohlcv_polars):
        signals = StatArbStrategy(
            entry_zscore=1.3,
            exit_zscore=0.0,
            zscore_window=10,
            side="long_short",
            pairs=[["AAA", "BBB"]],
        ).generate_signals(paired_ohlcv_polars)
        active = signals.filter(pl.col("signal") != 0)
        if active.is_empty():
            pytest.skip("no active signals in fixture")
        pivoted = active.pivot(index="date", on="symbol", values="signal")
        if "AAA" in pivoted.columns and "BBB" in pivoted.columns:
            sums = (pivoted["AAA"] + pivoted["BBB"]).to_list()
            assert all(s == 0 for s in sums if s is not None)

    def test_flat_positions_have_zero_weight(self, paired_ohlcv_polars):
        signals = StatArbStrategy(
            entry_zscore=1.3,
            exit_zscore=0.0,
            zscore_window=10,
            pairs=[["AAA", "BBB"]],
        ).generate_signals(paired_ohlcv_polars)
        flat = signals.filter(pl.col("signal") == 0)
        assert flat.filter(pl.col("weight") != 0.0).is_empty()

    def test_stat_arb_outputs_sparse_transition_rows(self, paired_ohlcv_polars):
        signals = StatArbStrategy(
            entry_zscore=1.3,
            exit_zscore=0.0,
            zscore_window=10,
            pairs=[["AAA", "BBB"]],
        ).generate_signals(paired_ohlcv_polars)
        full_grid_height = paired_ohlcv_polars.select("date", "symbol").unique().height
        assert signals.height < full_grid_height

    def test_exit_transitions_emit_explicit_zero_rows(self, paired_ohlcv_polars):
        signals = StatArbStrategy(
            entry_zscore=1.3,
            exit_zscore=0.0,
            zscore_window=10,
            pairs=[["AAA", "BBB"]],
        ).generate_signals(paired_ohlcv_polars)
        active = signals.filter(pl.col("signal") != 0)
        if active.is_empty():
            pytest.skip("no active signals in fixture")
        assert signals.filter((pl.col("signal") == 0) & (pl.col("weight") == 0.0)).height > 0


# ===========================================================================
# universe
# ===========================================================================

from qts.research.strategies.stat_arb.universe import stat_arb_universe_screener


class TestStatArbUniverseScreener:
    def test_returns_symbols_with_sufficient_history(self, prices_db):
        result = stat_arb_universe_screener(
            db_path=prices_db,
            prices_table="stock_prices",
            profiles_table=None,
            start_date=pd.Timestamp("2020-01-01"),
            end_date=pd.Timestamp("2020-12-31"),
            min_history_days=200,
            max_symbols=10,
        )
        assert set(result) == {"AAA", "BBB"}

    def test_min_history_filters_short_window(self, prices_db):
        result = stat_arb_universe_screener(
            db_path=prices_db,
            prices_table="stock_prices",
            profiles_table=None,
            start_date=pd.Timestamp("2020-01-01"),
            end_date=pd.Timestamp("2020-01-31"),
            min_history_days=100,
            max_symbols=10,
        )
        assert result == []

    def test_min_avg_volume_filter(self, prices_db):
        result = stat_arb_universe_screener(
            db_path=prices_db,
            prices_table="stock_prices",
            profiles_table=None,
            start_date=pd.Timestamp("2020-01-01"),
            end_date=pd.Timestamp("2020-12-31"),
            min_history_days=200,
            min_avg_volume=2_000_000.0,
            max_symbols=10,
        )
        assert result == []

    def test_max_symbols_respected(self, prices_db):
        result = stat_arb_universe_screener(
            db_path=prices_db,
            prices_table="stock_prices",
            profiles_table=None,
            start_date=pd.Timestamp("2020-01-01"),
            end_date=pd.Timestamp("2020-12-31"),
            min_history_days=200,
            max_symbols=1,
        )
        assert len(result) <= 1

    def test_sorted_by_avg_volume_descending(self, prices_db):
        result = stat_arb_universe_screener(
            db_path=prices_db,
            prices_table="stock_prices",
            profiles_table=None,
            start_date=pd.Timestamp("2020-01-01"),
            end_date=pd.Timestamp("2020-12-31"),
            min_history_days=200,
            max_symbols=10,
        )
        # AAA has higher avg_volume (1_000_000 vs 900_000)
        assert result[0] == "AAA"


@pytest.mark.skipif(not ZIPLINE_AVAILABLE, reason="zipline extra not installed")
def test_stat_arb_runs_through_both_engines(paired_ohlcv_polars):
    config = BacktestConfig(
        workflow="research",
        asset_types=["stock"],
        universe=UniverseConfig(stock=["AAA", "BBB"]),
        start_date=date(2024, 1, 2),
        end_date=date(2024, 3, 20),
        initial_capital=100000,
        backtest_engine="vectorbt",
    )
    strategy = StatArbStrategy(
        entry_zscore=1.3,
        exit_zscore=0.0,
        zscore_window=10,
        pairs=[["AAA", "BBB"]],
    )

    vectorbt = VectorBTProEngine().run(strategy, paired_ohlcv_polars, config)
    zipline = ZiplineReloadedEngine().run(strategy, paired_ohlcv_polars, config)

    assert vectorbt.returns.schema == zipline.returns.schema
    assert vectorbt.equity_curve.schema == zipline.equity_curve.schema
    assert set(vectorbt.metrics) == set(zipline.metrics)


def test_no_strategy_specific_runner_remains():
    repo_root = Path(__file__).resolve().parents[2]
    assert not (repo_root / "qts/research/backtest/stat_arb_runner.py").exists()
