from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pandas as pd


def _load_end2end_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "notebooks" / "end2end_stock_factor.py"
    module_name = "test_end2end_stock_factor_module"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_profitability_filter_accepts_fractional_roe_units():
    mod = _load_end2end_module()
    frame = pd.DataFrame(
        {
            "roe": [0.12, 0.03, None],
            "net_income_qoq": [0.1, 0.1, None],
        }
    )

    mask = mod.ProfitabilityFilter(min_roe=5.0).mask(frame)

    assert mask.tolist() == [True, False, False]


def test_profitability_filter_preserves_percentage_point_roe_units():
    mod = _load_end2end_module()
    frame = pd.DataFrame(
        {
            "roe": [12.0, 4.9],
            "net_income_qoq": [0.1, 0.1],
        }
    )

    mask = mod.ProfitabilityFilter(min_roe=5.0).mask(frame)

    assert mask.tolist() == [True, False]


def test_prepared_data_round_trip(tmp_path):
    mod = _load_end2end_module()
    universe = mod.UniverseConfig(
        name="unit_test_universe",
        symbols_path=tmp_path / "symbols.yml",
        runtime_root=tmp_path / "runtime",
    )
    config = mod.RunConfig(
        universe=universe,
        download_start_date=date.fromisoformat("2023-01-01"),
        optimize_end_date=date.fromisoformat("2023-01-02"),
        inference_start_date=date.fromisoformat("2023-01-03"),
        inference_end_date=date.fromisoformat("2023-01-04"),
        target_horizon=21,
        train_window=100,
        rebalance_period=21,
        max_positions=5,
        signal_class=4,
        commission=0.0015,
        initial_capital=100_000_000,
        max_drawdown_limit=0.30,
        n_trials=1,
        seed=42,
        price_interval="1d",
        batch_size=10,
        fundamental_pages=40,
        force_refresh_fundamentals=False,
        output_dir=tmp_path / "output",
        prepared_data_dir=tmp_path / "prepared",
        reuse_prepared_data=False,
    )

    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04"]),
            "symbol": ["VN:AAA", "VN:AAA", "VN:AAA"],
            "open": [10.0, 10.1, 10.2],
            "high": [10.2, 10.3, 10.4],
            "low": [9.9, 10.0, 10.1],
            "close": [10.1, 10.2, 10.3],
            "volume": [1000.0, 1100.0, 1200.0],
        }
    ).set_index(["date", "symbol"])
    featured = prices.copy()
    featured["roe"] = [0.11, 0.12, 0.13]
    featured["mom_21"] = [None, 0.01, 0.02]
    benchmark = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04"]),
            "close": [100.0, 101.0, 102.0],
        }
    )
    feature_cols = ["roe", "mom_21"]

    prepared_dir = mod.save_prepared_data(config, prices, benchmark, featured, feature_cols)
    loaded_prices, loaded_benchmark, loaded_featured, loaded_feature_cols = mod.load_prepared_data(
        mod.replace(config, reuse_prepared_data=True)
    )

    assert prepared_dir.exists()
    assert (prepared_dir / "featured_train.parquet").exists()
    assert (prepared_dir / "featured_inference.parquet").exists()
    pd.testing.assert_frame_equal(loaded_prices, prices.sort_index())
    pd.testing.assert_frame_equal(loaded_featured, featured.sort_index())
    pd.testing.assert_frame_equal(loaded_benchmark, benchmark)
    assert loaded_feature_cols == feature_cols
