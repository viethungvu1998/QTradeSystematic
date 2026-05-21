"""Backtest base models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any

import polars as pl

from qts.research.strategies.base import BaseStrategy

TRADE_LOG_SCHEMA = {
    "ticker": pl.String,
    "entry_time": pl.Datetime(time_unit="us"),
    "exit_time": pl.Datetime(time_unit="us"),
    "start_price": pl.Float64,
    "end_price": pl.Float64,
    "quantity": pl.Float64,
    "profit_pct": pl.Float64,
    "fee": pl.Float64,
    "side": pl.String,
}
TOKEN_SNAPSHOT_SCHEMA = pl.Struct(
    {
        "token": pl.String,
        "quantity": pl.Float64,
        "avg_buy_price": pl.Float64,
        "current_price": pl.Float64,
    }
)
PORTFOLIO_SNAPSHOTS_SCHEMA = {
    "timestamp": pl.Datetime(time_unit="us"),
    "tokens": pl.List(TOKEN_SNAPSHOT_SCHEMA),
    "equity": pl.Float64,
}


def empty_trade_log_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=TRADE_LOG_SCHEMA)


def empty_portfolio_snapshots_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=PORTFOLIO_SNAPSHOTS_SCHEMA)


def empty_backtest_result(
    engine_name: str = "vectorbt",
    signals: pl.DataFrame | None = None,
) -> BacktestResult:
    """Return a zero-metric BacktestResult with empty Polars frames."""
    empty_returns = pl.DataFrame(schema={"date": pl.Date, "portfolio_return": pl.Float64})
    empty_equity = pl.DataFrame(schema={"date": pl.Date, "equity": pl.Float64})
    return BacktestResult(
        engine_name=engine_name,
        metrics={"sharpe": 0.0, "sortino": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "win_rate": 0.0},
        returns=empty_returns,
        equity_curve=empty_equity,
        signals=signals if signals is not None else pl.DataFrame(),
        trade_log=empty_trade_log_frame(),
        portfolio_snapshots=empty_portfolio_snapshots_frame(),
    )


@dataclass(slots=True)
class UniverseConfig:
    stock: list[str] = field(default_factory=list)
    vn_stock: list[str] = field(default_factory=list)
    vn_warrant: list[str] = field(default_factory=list)
    vn_futures: list[str] = field(default_factory=list)
    crypto: list[str] = field(default_factory=list)
    crypto_futures: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DataSourcesConfig:
    stock: str | None = None
    vn_stock: str | None = None
    vn_warrant: str | None = None
    vn_futures: str | None = None
    crypto: str | None = None
    crypto_futures: str | None = None


@dataclass(slots=True)
class ForwardReturnsConfig:
    periods: list[int] = field(default_factory=list)


@dataclass(slots=True)
class IndicatorConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TransformStepConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FeaturesConfig:
    transforms: list[TransformStepConfig] = field(default_factory=list)
    indicators: list[IndicatorConfig] = field(default_factory=list)
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
    vn_stock: str | None = None
    vn_warrant: str | None = None
    vn_futures: str | None = None
    crypto: str | None = None
    crypto_futures: str | None = None
    binance_mode: str = "demo"  # demo (testnet) | live (production)


@dataclass(slots=True)
class ScheduleConfig:
    stock: str | None = None
    vn_stock: str | None = None
    vn_warrant: str | None = None
    vn_futures: str | None = None
    crypto: str | None = None
    crypto_futures: str | None = None


@dataclass(slots=True)
class PromotionGateConfig:
    max_sharpe_degradation: float = 0.0


@dataclass(slots=True)
class PortfolioConstructionConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationConfig:
    method: str = "single_split"
    test_start_date: date | None = None
    n_folds: int = 5
    fold_size_days: int = 252
    embargo_days: int = 0


@dataclass(slots=True)
class BacktestConfig:
    workflow: str
    asset_types: list[str]
    universe: UniverseConfig
    start_date: date | None = None
    end_date: date | None = None
    test_start_date: date | None = None  # first bar of the out-of-sample period
    initial_capital: Decimal | None = None
    data_sources: DataSourcesConfig = field(default_factory=DataSourcesConfig)
    storage: str = "duckdb"
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    strategy: StrategyConfig = field(default_factory=lambda: StrategyConfig(type="factor"))
    backtest_engine: str = "vectorbt"
    train_window: int = 252
    rebalance_frequency: str | int = "monthly"
    fill_model: str | None = None
    slippage_model: str | None = None
    commission: CommissionConfig | None = None
    calendar: str | None = None
    brokers: BrokersConfig | None = None
    schedule: ScheduleConfig | None = None
    promotion_gate: PromotionGateConfig | None = None
    portfolio_construction: PortfolioConstructionConfig | None = None
    validation: ValidationConfig | None = None


@dataclass(frozen=True, slots=True)
class BacktestResult:
    engine_name: str
    metrics: dict[str, float]
    returns: pl.DataFrame
    equity_curve: pl.DataFrame
    signals: pl.DataFrame
    trade_log: pl.DataFrame = field(default_factory=empty_trade_log_frame)
    portfolio_snapshots: pl.DataFrame = field(default_factory=empty_portfolio_snapshots_frame)
    metrics_is: dict[str, float] = field(default_factory=dict)
    metrics_oos: dict[str, float] = field(default_factory=dict)


class BaseEngine:
    """Backtest engine contract."""

    def run(
        self,
        strategy: BaseStrategy,
        data: pl.DataFrame,
        config: BacktestConfig,
        *,
        pipeline=None,
        ohlcv: pl.DataFrame | None = None,
    ) -> BacktestResult:
        raise NotImplementedError
