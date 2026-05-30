from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from qts.config.builder import Config
from qts.config.loader import load_config_from_mapping
from qts.research.strategies.ml_factor.classification import MLFactorStrategy


def _research_mapping() -> dict[str, object]:
    return {
        "workflow": "research",
        "asset_types": ["stock"],
        "universe": {"stock": ["AAPL"]},
        "start_date": "2024-01-01",
        "end_date": "2024-03-20",
        "initial_capital": 100000,
        "data_sources": {"stock": "fmp"},
        "storage": "duckdb",
        "features": {
            "indicators": [{"name": "rsi", "params": {"periods": [14]}}],
            "forward_returns": {"periods": [1]},
        },
        "strategy": {"type": "factor", "params": {"long_quantile": 0.8}},
        "backtest_engine": "vectorbt",
    }


def test_load_config_from_mapping_matches_yaml_loader_types():
    config = load_config_from_mapping(_research_mapping())

    assert config.workflow == "research"
    assert config.start_date == date(2024, 1, 1)
    assert config.initial_capital == Decimal("100000")
    assert config.features.indicators[0].name == "rsi"
    assert config.features.forward_returns.periods == [1]


def test_config_build_from_mapping_resolves_registered_components(monkeypatch, tmp_path):
    monkeypatch.setenv("QTS_ROOT", str(tmp_path / "qts_root"))

    resolved = Config.build_from_mapping(_research_mapping())

    assert resolved.raw.universe.stock == ["AAPL"]
    assert resolved.strategy is not None
    assert resolved.engine is not None


def test_config_build_from_mapping_resolves_ml_factor_factories(monkeypatch, tmp_path):
    monkeypatch.setenv("QTS_ROOT", str(tmp_path / "qts_root"))
    raw = _research_mapping()
    raw["strategy"] = {
        "type": "ml_factor",
        "params": {
            "predictor_cols": ["rsi_14"],
            "target_col": "forward_return_1",
            "rebalance_period": 5,
            "trainer": {
                "name": "xgb_classifier",
                "params": {"model_params": {"n_estimators": 1, "random_state": 7}},
            },
            "portfolio": {
                "name": "equal_weight",
                "params": {"num_long_positions": 1},
            },
        },
    }

    resolved = Config.build_from_mapping(raw)

    assert isinstance(resolved.strategy, MLFactorStrategy)
    assert resolved.strategy.predictor_cols == ["rsi_14"]
    assert resolved.strategy.target_col == "forward_return_1"
    assert resolved.strategy.rebalance_period == 5
    assert resolved.raw.rebalance_frequency == 5
    assert callable(resolved.strategy.train_func)
    assert callable(resolved.strategy.portfolio_func)


def test_config_build_supports_ml_factor_base_config(monkeypatch, tmp_path):
    monkeypatch.setenv("QTS_ROOT", str(tmp_path / "qts_root"))

    resolved = Config.build("configs/strategies/ml_factor/base.yaml")

    assert isinstance(resolved.strategy, MLFactorStrategy)
    assert resolved.raw.strategy.type == "ml_factor"
    assert resolved.raw.universe.stock == ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    assert resolved.strategy.target_col == "forward_return_21"
    assert resolved.strategy.rebalance_period == 10
    assert resolved.raw.rebalance_frequency == 10
    assert resolved.strategy.model is not None
    assert resolved.strategy.model.xgb_params == {
        "n_estimators": 50,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": -1,
    }


def test_ml_factor_rebalance_period_rejects_strings(monkeypatch, tmp_path):
    monkeypatch.setenv("QTS_ROOT", str(tmp_path / "qts_root"))
    raw = _research_mapping()
    raw["strategy"] = {
        "type": "ml_factor",
        "params": {
            "predictor_cols": ["rsi_14"],
            "target_col": "forward_return_1",
            "rebalance_period": "10d",
            "trainer": {
                "name": "xgb_classifier",
                "params": {"model_params": {"n_estimators": 1, "random_state": 7}},
            },
            "portfolio": {
                "name": "equal_weight",
                "params": {"num_long_positions": 1},
            },
        },
    }

    with pytest.raises(ValueError, match="rebalance_period must be a positive integer"):
        Config.build_from_mapping(raw)
