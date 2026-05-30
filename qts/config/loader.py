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
    PortfolioConstructionConfig,
    PromotionGateConfig,
    ScheduleConfig,
    StrategyConfig,
    TransformStepConfig,
    UniverseConfig,
    ValidationConfig,
)

ALLOWED_TOP_LEVEL_KEYS = {
    "workflow",
    "asset_types",
    "universe",
    "start_date",
    "end_date",
    "test_start_date",
    "initial_capital",
    "data_sources",
    "storage",
    "features",
    "strategy",
    "backtest_engine",
    "train_window",
    "rebalance",
    "rebalance_frequency",
    "portfolio_construction",
    "validation",
    "fill_model",
    "slippage_model",
    "commission",
    "calendar",
    "brokers",
    "schedule",
    "promotion_gate",
    "benchmark",
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
    return _parse_date(value)


def _parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    return date.fromisoformat(str(value))


def _strategy_params(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    strategy = payload.get("strategy", {})
    if not isinstance(strategy, Mapping):
        return {}
    params = strategy.get("params", {})
    return params if isinstance(params, Mapping) else {}


def _resolve_rebalance(raw: Mapping[str, Any]) -> str | int:
    if "rebalance" in raw:
        return _as_positive_int(raw["rebalance"], "rebalance")
    if "rebalance_frequency" in raw:
        value = raw["rebalance_frequency"]
        if isinstance(value, str) and not value.strip().isdigit():
            return value
        return _as_positive_int(value, "rebalance_frequency")
    rebalance_period = raw.get("strategy", {}).get("params", {}).get("rebalance_period")
    if rebalance_period is not None:
        return _as_positive_int(rebalance_period, "rebalance_period")
    return BacktestConfig.__dataclass_fields__["rebalance_frequency"].default


def _as_positive_int(value: object, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if isinstance(value, bool) or result < 1:
        raise ValueError(f"{label} must be a positive integer")
    return result


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
    raw_transforms = payload.get("features", {}).get("transforms", [])
    transforms = [
        TransformStepConfig(name=str(item["name"]), params=dict(item.get("params", {})))
        for item in raw_transforms
    ]
    indicators = [
        IndicatorConfig(name=item["name"], params=item.get("params", {}))
        for item in features_payload.get("indicators", [])
    ]
    features_config = FeaturesConfig(
        transforms=transforms,
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
    pc_raw = payload.get("portfolio_construction")
    portfolio_construction: PortfolioConstructionConfig | None = None
    if pc_raw:
        portfolio_construction = PortfolioConstructionConfig(
            name=str(pc_raw["name"]),
            params=dict(pc_raw.get("params", {})),
        )

    val_raw = payload.get("validation")
    validation: ValidationConfig | None = None
    if val_raw:
        validation = ValidationConfig(
            method=str(val_raw.get("method", "single_split")),
            test_start_date=_parse_date(val_raw.get("test_start_date")),
            n_folds=int(val_raw.get("n_folds", 5)),
            fold_size_days=int(val_raw.get("fold_size_days", 252)),
            embargo_days=int(val_raw.get("embargo_days", 0)),
        )
    elif payload.get("test_start_date"):
        validation = ValidationConfig(test_start_date=_parse_date(payload["test_start_date"]))

    train_window_default = BacktestConfig.__dataclass_fields__["train_window"].default
    return BacktestConfig(
        workflow=workflow,
        asset_types=list(payload.get("asset_types", [])),
        universe=universe,
        start_date=_as_date(payload.get("start_date")),
        end_date=_as_date(payload.get("end_date")),
        test_start_date=_as_date(payload.get("test_start_date")),
        initial_capital=initial_capital,
        data_sources=data_sources,
        storage=payload.get("storage", "duckdb"),
        features=features_config,
        strategy=StrategyConfig(**payload.get("strategy", {})),
        backtest_engine=str(payload.get("backtest_engine", "vectorbt")).strip().lower(),
        train_window=payload.get("train_window", train_window_default),
        rebalance_frequency=_resolve_rebalance(payload),
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
        portfolio_construction=portfolio_construction,
        validation=validation,
        benchmark=payload.get("benchmark"),
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
