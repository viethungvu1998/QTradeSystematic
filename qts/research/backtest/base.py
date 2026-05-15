"""Backtest base models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any

import polars as pl

from qts.research.strategies.base import BaseStrategy


@dataclass(slots=True)
class UniverseConfig:
    stock: list[str] = field(default_factory=list)
    crypto: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DataSourcesConfig:
    stock: str | None = None
    crypto: str | None = None


@dataclass(slots=True)
class ForwardReturnsConfig:
    periods: list[int] = field(default_factory=list)


@dataclass(slots=True)
class FeaturesConfig:
    technical: bool = False
    fundamental: bool = False
    onchain: bool = False
    forward_returns: ForwardReturnsConfig = field(default_factory=ForwardReturnsConfig)


@dataclass(slots=True)
class StrategyConfig:
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CommissionConfig:
    model: str
    rate: Decimal


@dataclass(slots=True)
class BrokersConfig:
    stock: str | None = None
    crypto: str | None = None
    binance_mode: str = "demo"  # demo (testnet) | live (production)


@dataclass(slots=True)
class ScheduleConfig:
    stock: str | None = None
    crypto: str | None = None


@dataclass(slots=True)
class PromotionGateConfig:
    max_sharpe_degradation: float = 0.0


@dataclass(slots=True)
class BacktestConfig:
    workflow: str
    asset_types: list[str]
    universe: UniverseConfig
    start_date: date | None = None
    end_date: date | None = None
    initial_capital: Decimal | None = None
    data_sources: DataSourcesConfig = field(default_factory=DataSourcesConfig)
    storage: str = "duckdb"
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    strategy: StrategyConfig = field(default_factory=lambda: StrategyConfig(type="factor"))
    backtest_engine: str = "fast"
    fill_model: str | None = None
    slippage_model: str | None = None
    commission: CommissionConfig | None = None
    calendar: str | None = None
    brokers: BrokersConfig | None = None
    schedule: ScheduleConfig | None = None
    promotion_gate: PromotionGateConfig | None = None


@dataclass(frozen=True, slots=True)
class BacktestResult:
    engine_name: str
    metrics: dict[str, float]
    returns: pl.DataFrame
    equity_curve: pl.DataFrame
    signals: pl.DataFrame


class BaseEngine:
    """Backtest engine contract."""

    def run(self, strategy: BaseStrategy, data: pl.DataFrame, config: BacktestConfig) -> BacktestResult:
        raise NotImplementedError
