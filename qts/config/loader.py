"""YAML configuration loader."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from qts.core.errors import ConfigError
from qts.research.backtest.base import (
    BacktestConfig,
    BrokersConfig,
    CommissionConfig,
    DataSourcesConfig,
    FeaturesConfig,
    ForwardReturnsConfig,
    IndicatorConfig,
    PromotionGateConfig,
    ScheduleConfig,
    StrategyConfig,
    UniverseConfig,
)

ALLOWED_TOP_LEVEL_KEYS = {
    "workflow",
    "asset_types",
    "universe",
    "start_date",
    "end_date",
    "initial_capital",
    "data_sources",
    "storage",
    "features",
    "strategy",
    "backtest_engine",
    "train_window",
    "rebalance_frequency",
    "fill_model",
    "slippage_model",
    "commission",
    "calendar",
    "brokers",
    "schedule",
    "promotion_gate",
}
BASE_CONFIG_KEY = "base_config"

REQUIRED_KEYS_BY_WORKFLOW = {
    "research": {
        "workflow",
        "asset_types",
        "universe",
        "start_date",
        "end_date",
        "initial_capital",
        "data_sources",
        "storage",
        "features",
        "strategy",
        "backtest_engine",
    },
    "validation": {
        "workflow",
        "asset_types",
        "universe",
        "start_date",
        "end_date",
        "initial_capital",
        "data_sources",
        "storage",
        "features",
        "strategy",
        "backtest_engine",
        "fill_model",
        "slippage_model",
        "commission",
        "calendar",
        "promotion_gate",
    },
    "live": {
        "workflow",
        "asset_types",
        "universe",
        "data_sources",
        "storage",
        "features",
        "strategy",
        "backtest_engine",
        "fill_model",
        "slippage_model",
        "commission",
        "brokers",
        "schedule",
    },
}


def _as_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _normalize_backtest_engine(value: str | None) -> str:
    if not value:
        return "vectorbt"
    normalized = value.strip().lower()
    aliases = {
        "fast": "vectorbt",
        "normal": "zipline",
    }
    return aliases.get(normalized, normalized)


def _strategy_params(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    strategy = payload.get("strategy", {})
    if not isinstance(strategy, Mapping):
        return {}
    params = strategy.get("params", {})
    return params if isinstance(params, Mapping) else {}


def _resolve_rebalance_frequency(payload: Mapping[str, Any]) -> str | int:
    default = BacktestConfig.__dataclass_fields__["rebalance_frequency"].default
    params = _strategy_params(payload)
    value = params.get(
        "rebalance_period",
        params.get("rebalance_frequency", payload.get("rebalance_frequency", default)),
    )
    return value if isinstance(value, int) and not isinstance(value, bool) else str(value)


def load_config_from_mapping(raw: Mapping[str, Any]) -> BacktestConfig:
    """Parse and validate a mapping into a typed config."""

    payload = deepcopy(dict(raw))
    unknown = sorted(set(payload) - ALLOWED_TOP_LEVEL_KEYS)
    if unknown:
        raise ConfigError(f"Unknown top-level key(s): {', '.join(unknown)}")
    workflow = payload.get("workflow")
    if workflow not in REQUIRED_KEYS_BY_WORKFLOW:
        raise ConfigError("workflow must be one of research, validation, live")
    missing = sorted(REQUIRED_KEYS_BY_WORKFLOW[workflow] - set(payload))
    if missing:
        raise ConfigError(f"Missing required key(s): {', '.join(missing)}")
    universe = UniverseConfig(**payload.get("universe", {}))
    data_sources = DataSourcesConfig(**payload.get("data_sources", {}))
    features_payload = payload.get("features", {})
    indicators = [
        IndicatorConfig(name=item["name"], params=item.get("params", {}))
        for item in features_payload.get("indicators", [])
    ]
    features_config = FeaturesConfig(
        indicators=indicators,
        technical=bool(features_payload.get("technical", False)),
        fundamental=bool(features_payload.get("fundamental", False)),
        onchain=bool(features_payload.get("onchain", False)),
        forward_returns=ForwardReturnsConfig(
            periods=list(features_payload.get("forward_returns", {}).get("periods", []))
        ),
    )
    commission = payload.get("commission")
    initial_capital = (
        Decimal(str(payload["initial_capital"])) if "initial_capital" in payload else None
    )
    train_window_default = BacktestConfig.__dataclass_fields__["train_window"].default
    return BacktestConfig(
        workflow=workflow,
        asset_types=list(payload.get("asset_types", [])),
        universe=universe,
        start_date=_as_date(payload.get("start_date")),
        end_date=_as_date(payload.get("end_date")),
        initial_capital=initial_capital,
        data_sources=data_sources,
        storage=payload.get("storage", "duckdb"),
        features=features_config,
        strategy=StrategyConfig(**payload.get("strategy", {})),
        backtest_engine=_normalize_backtest_engine(payload.get("backtest_engine")),
        train_window=payload.get("train_window", train_window_default),
        rebalance_frequency=_resolve_rebalance_frequency(payload),
        fill_model=payload.get("fill_model"),
        slippage_model=payload.get("slippage_model"),
        commission=CommissionConfig(
            model=commission["model"],
            rate=Decimal(str(commission["rate"])),
        )
        if commission
        else None,
        calendar=payload.get("calendar"),
        brokers=BrokersConfig(**payload["brokers"]) if "brokers" in payload else None,
        schedule=ScheduleConfig(**payload["schedule"]) if "schedule" in payload else None,
        promotion_gate=PromotionGateConfig(**payload["promotion_gate"])
        if "promotion_gate" in payload
        else None,
    )


def load_config(path: str | Path) -> BacktestConfig:
    """Parse and validate YAML into a typed config."""

    return load_config_from_mapping(load_config_mapping(path))


def load_config_mapping(path: str | Path) -> dict[str, Any]:
    """Load YAML config inheritance into a plain mapping."""

    return _load_config_mapping(Path(path))


def _load_config_mapping(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    seen = set(seen or set())
    if resolved in seen:
        raise ConfigError(f"Cyclic base_config reference: {resolved}")
    seen.add(resolved)
    payload = yaml.safe_load(resolved.read_text()) or {}
    if not isinstance(payload, Mapping):
        raise ConfigError(f"Expected mapping YAML: {resolved}")
    overlay = deepcopy(dict(payload))
    base_value = overlay.pop(BASE_CONFIG_KEY, None)
    if base_value is None:
        return overlay
    base_path = Path(base_value)
    if not base_path.is_absolute():
        base_path = resolved.parent / base_path
    return _deep_overlay_merge(_load_config_mapping(base_path, seen), overlay)


def _deep_overlay_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_overlay_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
