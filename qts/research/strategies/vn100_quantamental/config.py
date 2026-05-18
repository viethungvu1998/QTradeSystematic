"""Dataclasses and constants for the VN100 quantamental strategy."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any


TECHNICAL_BASE_COLUMNS: list[str] = [
    "roc_21",
    "roc_63",
    "roc_126",
    "roc_252",
    "rsi_14",
    "rsi_63",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "atr_14",
    "hist_vol_21",
    "hist_vol_63",
    "hist_vol_126",
    "close_to_sma_50",
    "close_to_sma_200",
    "volume_ratio_20",
    "dollar_volume_zscore_63",
    "intraday_range",
    "close_location_value",
]

MODEL_PARAMS: dict[str, Any] = {
    "random_state": 123,
    "objective": "reg:squarederror",
    "n_estimators": 120,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "n_jobs": -1,
}


@dataclass(frozen=True)
class FeatureConfig:
    min_trading_days: int = 252
    volume_top_n: int | None = 80
    min_avg_volume: float = 50_000
    remove_large_gaps: bool = False
    max_gap_days: int = 21
    remove_low_volume: bool = False
    qsmom_fast: int = 21
    qsmom_slow: int = 252
    qsmom_returns: int = 126
    forward_period: int = 21
    fundamental_termtype: int = 1


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    feature: FeatureConfig
    predictor_cols: list[str] | None = None
    train_window: int = 504
    rebalance_period: str | int = "monthly"
    num_long_positions: int = 10
    num_short_positions: int = 0
    long_threshold: float | None = None
    short_threshold: float | None = None
    model_params: dict[str, Any] | None = None
    initial_capital: Decimal = Decimal("1000000000")


def qsmom_column(config: FeatureConfig) -> str:
    return f"close_qsmom_{config.qsmom_fast}_{config.qsmom_slow}_{config.qsmom_returns}"
