from __future__ import annotations

import pytest

from qts.config.builder import Config
from qts.config.loader import load_config
from qts.core.errors import ConfigError


def test_load_typed_configs(config_dir):
    research = load_config(config_dir / "research.yaml")
    validation = load_config(config_dir / "validation.yaml")
    live = load_config(config_dir / "live.yaml")
    assert research.initial_capital is not None
    assert validation.commission is not None
    assert live.brokers is not None


def test_missing_brokers_raises(config_dir):
    path = config_dir / "bad_live.yaml"
    path.write_text((config_dir / "live.yaml").read_text().replace("brokers:\n  stock: moomoo\n  crypto: binance\n", ""))
    with pytest.raises(ConfigError, match="brokers"):
        load_config(path)


def test_unknown_top_level_key_raises(config_dir):
    path = config_dir / "bad.yaml"
    path.write_text((config_dir / "research.yaml").read_text() + "\nunknown_key: true\n")
    with pytest.raises(ConfigError, match="unknown_key"):
        load_config(path)


def test_config_build_resolves_components(config_dir):
    resolved = Config.build(config_dir / "validation.yaml")
    assert resolved.engine is not None
    assert resolved.strategy is not None
    assert resolved.fill_model is not None
