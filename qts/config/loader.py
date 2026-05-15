"""YAML configuration loader."""

from __future__ import annotations

from dataclasses import fields
from datetime import date
from decimal import Decimal
from pathlib import Path

import yaml

from qts.core.errors import ConfigError
from qts.research.backtest.base import (
    BacktestConfig,
    BrokersConfig,
    CommissionConfig,
    DataSourcesConfig,
    FeaturesConfig,
    ForwardReturnsConfig,
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
    "fill_model",
    "slippage_model",
    "commission",
    "calendar",
    "brokers",
    "schedule",
    "promotion_gate",
}

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


def load_config(path: str | Path) -> BacktestConfig:
    """Parse and validate YAML into a typed config."""

    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown = sorted(set(raw) - ALLOWED_TOP_LEVEL_KEYS)
    if unknown:
        raise ConfigError(f"Unknown top-level key(s): {', '.join(unknown)}")
    workflow = raw.get("workflow")
    if workflow not in REQUIRED_KEYS_BY_WORKFLOW:
        raise ConfigError("workflow must be one of research, validation, live")
    missing = sorted(REQUIRED_KEYS_BY_WORKFLOW[workflow] - set(raw))
    if missing:
        raise ConfigError(f"Missing required key(s): {', '.join(missing)}")
    universe = UniverseConfig(**raw.get("universe", {}))
    data_sources = DataSourcesConfig(**raw.get("data_sources", {}))
    features_payload = raw.get("features", {})
    features_config = FeaturesConfig(
        technical=bool(features_payload.get("technical", False)),
        fundamental=bool(features_payload.get("fundamental", False)),
        onchain=bool(features_payload.get("onchain", False)),
        forward_returns=ForwardReturnsConfig(
            periods=list(features_payload.get("forward_returns", {}).get("periods", []))
        ),
    )
    commission = raw.get("commission")
    return BacktestConfig(
        workflow=workflow,
        asset_types=list(raw.get("asset_types", [])),
        universe=universe,
        start_date=_as_date(raw.get("start_date")),
        end_date=_as_date(raw.get("end_date")),
        initial_capital=Decimal(str(raw["initial_capital"])) if "initial_capital" in raw else None,
        data_sources=data_sources,
        storage=raw.get("storage", "duckdb"),
        features=features_config,
        strategy=StrategyConfig(**raw.get("strategy", {})),
        backtest_engine=raw.get("backtest_engine", "fast"),
        fill_model=raw.get("fill_model"),
        slippage_model=raw.get("slippage_model"),
        commission=CommissionConfig(
            model=commission["model"],
            rate=Decimal(str(commission["rate"])),
        )
        if commission
        else None,
        calendar=raw.get("calendar"),
        brokers=BrokersConfig(**raw["brokers"]) if "brokers" in raw else None,
        schedule=ScheduleConfig(**raw["schedule"]) if "schedule" in raw else None,
        promotion_gate=PromotionGateConfig(**raw["promotion_gate"])
        if "promotion_gate" in raw
        else None,
    )
