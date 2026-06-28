from __future__ import annotations

import polars as pl

from qts.config.builder import Config


def test_config_build_resolves_ml_factor_target_class_transform(monkeypatch, tmp_path):
    monkeypatch.setenv("QTS_ROOT", str(tmp_path / "qts_root"))
    raw = {
        "workflow": "research",
        "asset_types": ["stock"],
        "universe": {"stock": ["AAPL"]},
        "start_date": "2024-01-01",
        "end_date": "2024-03-20",
        "initial_capital": 100000,
        "data_sources": {"stock": "fmp"},
        "storage": "duckdb",
        "features": {
            "transforms": [
                {
                    "name": "ml_factor_target_class",
                    "params": {
                        "target_column": "forward_return_{rebalance_period}",
                        "output_column": "forward_return_{rebalance_frequency}_class",
                    },
                }
            ]
        },
        "strategy": {
            "type": "ml_factor",
            "params": {
                "task": "classification",
                "model": {"name": "xgb_classifier", "params": {"n_estimators": 1}},
            },
        },
        "backtest_engine": "vectorbt",
        "rebalance_frequency": 1,
    }

    resolved = Config.build_from_mapping(raw)
    featured = resolved.feature_pipeline.fit_transform(
        pl.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "symbol": ["AAPL", "AAPL", "AAPL"],
                "open": [10.0, 11.0, 12.0],
                "high": [10.5, 11.5, 12.5],
                "low": [9.5, 10.5, 11.5],
                "close": [10.0, 11.0, 12.0],
                "volume": [1000.0, 1100.0, 1200.0],
            }
        ).with_columns(pl.col("date").str.to_date())
    )

    assert "forward_return_1_class" in featured.columns


def test_config_build_defaults_ml_factor_target_class_columns_from_rebalance(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("QTS_ROOT", str(tmp_path / "qts_root"))
    raw = {
        "workflow": "research",
        "asset_types": ["stock"],
        "universe": {"stock": ["AAPL"]},
        "start_date": "2024-01-01",
        "end_date": "2024-03-20",
        "initial_capital": 100000,
        "data_sources": {"stock": "fmp"},
        "storage": "duckdb",
        "features": {"transforms": [{"name": "ml_factor_target_class"}]},
        "strategy": {
            "type": "ml_factor",
            "params": {
                "task": "classification",
                "model": {"name": "xgb_classifier", "params": {"n_estimators": 1}},
            },
        },
        "backtest_engine": "vectorbt",
        "rebalance_frequency": 5,
    }

    resolved = Config.build_from_mapping(raw)
    featured = resolved.feature_pipeline.fit_transform(
        pl.DataFrame(
            {
                "date": [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-08",
                ],
                "symbol": ["AAPL"] * 6,
                "open": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                "high": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
                "low": [9.5, 10.5, 11.5, 12.5, 13.5, 14.5],
                "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0],
            }
        ).with_columns(pl.col("date").str.to_date())
    )

    assert "forward_return_5" in featured.columns
    assert "forward_return_5_class" in featured.columns
