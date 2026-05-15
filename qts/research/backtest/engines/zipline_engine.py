"""Bar-by-bar backtest engine."""

from __future__ import annotations

import polars as pl

from qts.core.registry import Registry
from qts.research.backtest._runner import run_backtest_frame
from qts.research.backtest.base import BacktestConfig, BacktestResult, BaseEngine
from qts.research.strategies.base import BaseStrategy


@Registry.register_engine("normal")
class ZiplineEngine(BaseEngine):
    """Sequential engine with the same output schema."""

    def run(self, strategy: BaseStrategy, data: pl.DataFrame, config: BacktestConfig) -> BacktestResult:
        return run_backtest_frame("normal", strategy, data, config)
