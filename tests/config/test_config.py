from __future__ import annotations

from importlib import import_module

import polars as pl
import pytest

from qts.config.builder import Config
from qts.config.loader import load_config
from qts.core.errors import ConfigError
from qts.core.instrument import AssetType
from qts.core.registry import Registry, RegistryError
from qts.data.sources.dnse import DNSEDataSource
from qts.research.backtest.base import (
    BacktestConfig,
    FeaturesConfig,
    IndicatorConfig,
    UniverseConfig,
)
from qts.research.features.forward_returns import ForwardReturns
from qts.research.features.fundamentals import FundamentalFeatures
from qts.research.features.indicators.momentum import RSIFeature
from qts.research.features.pipeline import FeaturePipeline
from qts.research.features.technical import TechnicalFeatures


def test_load_typed_configs(config_dir):
    research = load_config(config_dir / "research.yaml")
    validation = load_config(config_dir / "validation.yaml")
    live = load_config(config_dir / "live.yaml")
    assert research.initial_capital is not None
    assert validation.commission is not None
    assert live.brokers is not None
    assert research.train_window == 252
    assert research.rebalance_frequency == "monthly"


def test_missing_brokers_raises(config_dir):
    path = config_dir / "bad_live.yaml"
    path.write_text(
        (config_dir / "live.yaml")
        .read_text()
        .replace(
            "brokers:\n  stock: moomoo\n  crypto: binance\n",
            "",
        )
    )
    with pytest.raises(ConfigError, match="brokers"):
        load_config(path)


def test_unknown_top_level_key_raises(config_dir):
    path = config_dir / "bad.yaml"
    path.write_text((config_dir / "research.yaml").read_text() + "\nunknown_key: true\n")
    with pytest.raises(ConfigError, match="unknown_key"):
        load_config(path)


def test_config_build_resolves_components(config_dir, monkeypatch, tmp_path):
    import_module("qts.data.sources.fmp")
    import_module("qts.data.sources.binance")
    import_module("qts.data.sources.dnse")
    monkeypatch.setenv("QTS_ROOT", str(tmp_path / "qts_root"))
    resolved = Config.build(config_dir / "validation.yaml")
    assert resolved.engine is not None
    assert resolved.strategy is not None
    assert resolved.fill_model is not None
    assert isinstance(resolved.storage, Registry.get_storage("duckdb"))
    assert isinstance(resolved.cache, Registry.get_storage("parquet"))


def test_vn_stock_config_defaults_and_resolution(config_dir, monkeypatch, tmp_path):
    import_module("qts.data.sources.fmp")
    import_module("qts.data.sources.binance")
    import_module("qts.data.sources.dnse")
    path = config_dir / "vn_research.yaml"
    path.write_text(
        """
workflow: research
asset_types: [vn_stock]
universe:
  vn_stock: [VN:VNM]
start_date: "2024-01-01"
end_date: "2024-03-20"
initial_capital: 100000
data_sources:
  vn_stock: dnse
storage: duckdb
features:
  technical: false
  fundamental: false
  onchain: false
  forward_returns:
    periods: []
strategy:
  type: factor
  params: {}
backtest_engine: vectorbt
"""
    )
    config = load_config(path)
    assert config.universe.vn_stock == ["VN:VNM"]

    no_vn_path = config_dir / "no_vn.yaml"
    no_vn_path.write_text((config_dir / "research.yaml").read_text())
    no_vn = load_config(no_vn_path)
    assert no_vn.universe.vn_stock == []

    monkeypatch.setenv("QTS_ROOT", str(tmp_path / "qts_root"))
    resolved = Config.build(path)
    assert resolved.vn_stock_source is not None
    assert (tmp_path / "qts_root" / "database" / "qts.duckdb").exists()


def test_vn_stock_dnse_prefers_from_env(config_dir, monkeypatch, tmp_path):
    import_module("qts.data.sources.dnse")
    path = config_dir / "vn_dnse_live.yaml"
    path.write_text(
        """
workflow: research
asset_types: [vn_stock]
universe:
  vn_stock: [VN:VNM]
start_date: "2024-01-01"
end_date: "2024-03-20"
initial_capital: 100000
data_sources:
  vn_stock: dnse
storage: duckdb
features:
  technical: false
  fundamental: false
  onchain: false
  forward_returns:
    periods: []
strategy:
  type: factor
  params: {}
backtest_engine: vectorbt
"""
    )
    sentinel = object()

    def fake_from_env(cls):
        return cls(client=sentinel)

    monkeypatch.setattr(DNSEDataSource, "from_env", classmethod(fake_from_env))
    monkeypatch.setenv("QTS_ROOT", str(tmp_path / "qts_root"))
    resolved = Config.build(path)

    assert resolved.vn_stock_source is not None
    assert resolved.vn_stock_source._client is sentinel


def test_backtest_config_and_features_defaults():
    config = BacktestConfig(workflow="research", asset_types=[], universe=UniverseConfig())
    assert config.train_window == 252
    assert config.rebalance_frequency == "monthly"
    assert config.backtest_engine == "vectorbt"

    features = FeaturesConfig(indicators=[IndicatorConfig("rsi", {"periods": [14]})])
    assert features.indicators[0].name == "rsi"
    assert features.indicators[0].params == {"periods": [14]}


def test_loader_parses_indicator_list_and_train_window(config_dir):
    path = config_dir / "indicator_research.yaml"
    path.write_text(
        """
workflow: research
asset_types: [stock]
universe:
  stock: [AAPL]
start_date: "2024-01-01"
end_date: "2024-03-20"
initial_capital: 100000
data_sources:
  stock: fmp
storage: duckdb
features:
  indicators:
    - name: rsi
      params:
        periods: [14]
  forward_returns:
    periods: [1]
strategy:
  type: factor
  params: {}
backtest_engine: vectorbt
train_window: 126
"""
    )
    config = load_config(path)
    assert config.features.indicators[0].name == "rsi"
    assert config.features.indicators[0].params == {"periods": [14]}
    assert config.train_window == 126
    assert config.rebalance_frequency == "monthly"


def test_config_build_resolves_registry_driven_indicators(config_dir, monkeypatch, tmp_path):
    import_module("qts.data.sources.fmp")
    monkeypatch.setenv("QTS_ROOT", str(tmp_path / "qts_root"))
    path = config_dir / "indicator_build.yaml"
    path.write_text(
        """
workflow: research
asset_types: [stock]
universe:
  stock: [AAPL]
start_date: "2024-01-01"
end_date: "2024-03-20"
initial_capital: 100000
data_sources:
  stock: fmp
storage: duckdb
features:
  indicators:
    - name: rsi
      params:
        periods: [14]
strategy:
  type: factor
  params: {}
backtest_engine: vectorbt
"""
    )
    resolved = Config.build(path)
    assert isinstance(resolved.feature_pipeline.features[0], RSIFeature)
    assert resolved.feature_pipeline.features[0].periods == [14]


def test_config_build_unknown_indicator_raises(config_dir, monkeypatch, tmp_path):
    import_module("qts.data.sources.fmp")
    monkeypatch.setenv("QTS_ROOT", str(tmp_path / "qts_root"))
    path = config_dir / "unknown_indicator.yaml"
    path.write_text(
        """
workflow: research
asset_types: [stock]
universe:
  stock: [AAPL]
start_date: "2024-01-01"
end_date: "2024-03-20"
initial_capital: 100000
data_sources:
  stock: fmp
storage: duckdb
features:
  indicators:
    - name: unknown_xyz
strategy:
  type: factor
  params: {}
backtest_engine: vectorbt
"""
    )
    with pytest.raises(RegistryError, match="unknown_xyz"):
        Config.build(path)


def test_config_build_keeps_technical_backward_compat(config_dir, monkeypatch, tmp_path):
    import_module("qts.data.sources.fmp")
    monkeypatch.setenv("QTS_ROOT", str(tmp_path / "qts_root"))
    resolved = Config.build(config_dir / "research.yaml")
    assert isinstance(resolved.feature_pipeline.features[0], TechnicalFeatures)


def test_config_build_resolves_forward_returns_via_registry(config_dir, monkeypatch, tmp_path):
    import_module("qts.data.sources.fmp")
    monkeypatch.setenv("QTS_ROOT", str(tmp_path / "qts_root"))
    resolved = Config.build(config_dir / "research.yaml")
    assert any(
        isinstance(feature, ForwardReturns) for feature in resolved.feature_pipeline.features
    )


def test_resolved_config_collects_runtime_components(config_dir, monkeypatch, tmp_path):
    import_module("qts.data.sources.fmp")
    import_module("qts.data.sources.binance")
    monkeypatch.setenv("QTS_ROOT", str(tmp_path / "qts_root"))
    resolved = Config.build(config_dir / "live.yaml")
    assert resolved.data_sources() == {
        AssetType.STOCK: resolved.stock_source,
        AssetType.CRYPTO: resolved.crypto_source,
    }
    assert resolved.brokers() == {
        AssetType.STOCK: resolved.stock_broker,
        AssetType.CRYPTO: resolved.crypto_broker,
    }


def test_resolved_config_binds_fundamentals_without_mutating_pipeline(
    config_dir,
    monkeypatch,
    tmp_path,
):
    import_module("qts.data.sources.fmp")
    monkeypatch.setenv("QTS_ROOT", str(tmp_path / "qts_root"))
    path = config_dir / "fundamental_build.yaml"
    path.write_text(
        """
workflow: research
asset_types: [stock]
universe:
  stock: [AAPL]
start_date: "2024-01-01"
end_date: "2024-03-20"
initial_capital: 100000
data_sources:
  stock: fmp
storage: duckdb
features:
  technical: false
  fundamental: true
  onchain: false
strategy:
  type: factor
  params: {}
backtest_engine: vectorbt
"""
    )
    resolved = Config.build(path)
    fundamentals = pl.DataFrame({"symbol": ["AAPL"], "pe_ratio": [21.5], "ev_ebitda": [15.2]})

    bound_pipeline = resolved.with_fundamentals(fundamentals)

    original_feature = resolved.feature_pipeline.features[0]
    bound_feature = bound_pipeline.features[0]
    assert isinstance(original_feature, FundamentalFeatures)
    assert isinstance(bound_feature, FundamentalFeatures)
    assert original_feature.fundamentals.is_empty()
    assert bound_feature.fundamentals.equals(fundamentals)


def test_feature_pipeline_with_fundamentals_returns_bound_copy():
    feature = FundamentalFeatures()
    pipeline = FeaturePipeline([feature])
    fundamentals = pl.DataFrame({"symbol": ["AAPL"], "pe_ratio": [21.5], "ev_ebitda": [15.2]})

    bound_pipeline = pipeline.with_fundamentals(fundamentals)

    assert bound_pipeline is not pipeline
    assert bound_pipeline.features[0] is not feature
    assert feature.fundamentals.is_empty()
    assert bound_pipeline.features[0].fundamentals.equals(fundamentals)


def test_package_bootstrap_registers_strategy_families():
    import_module("qts")
    assert Registry.get_strategy("factor").__name__ == "FactorStrategy"
    assert Registry.get_strategy("ml_factor").__name__ == "MLFactorStrategy"
    assert Registry.get_strategy("stat_arb").__name__ == "StatArbStrategy"
    assert Registry.get_strategy("vn100_quantamental").__name__ == "VN100QuantamentalStrategy"
