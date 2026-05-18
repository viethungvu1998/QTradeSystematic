"""VN100 quantamental ML factor strategy."""

from __future__ import annotations

from decimal import Decimal
from itertools import product
from typing import Any

import polars as pl

from qts.core.registry import Registry
from qts.research.backtest.base import (
    BacktestConfig,
    BacktestResult,
    CommissionConfig,
    UniverseConfig,
)
from qts.research.backtest._runner import run_backtest_frame
from qts.research.strategies.factor.base import BaseFactorStrategy

from .config import MODEL_PARAMS, ExperimentConfig, FeatureConfig
from .features import build_model_frame
from .signals import choose_predictors, walk_forward_ml_signals


_SWEEP_REBALANCE_PERIODS: list[int] = [10, 21, 63]  # bi-weekly, monthly, quarterly


@Registry.register_strategy("vn100_quantamental")
class VN100QuantamentalStrategy(BaseFactorStrategy):
    """Walk-forward ML factor strategy for the VN100 universe.

    Builds quantamental features (QSMOM + technicals + fundamentals), trains an
    XGBoost regressor on rolling windows, and produces long-only signals.
    """

    def __init__(self, experiment: ExperimentConfig) -> None:
        self.experiment = experiment

    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """Generate walk-forward ML signals from a fully-featured model frame.

        ``data`` must already have been processed through ``build_model_frame``.
        """
        target_col = f"forward_return_{self.experiment.feature.forward_period}"
        if target_col not in data.columns:
            return self.empty_signal_frame()

        predictor_cols = choose_predictors(data, self.experiment.feature, self.experiment.predictor_cols)
        if not predictor_cols:
            return self.empty_signal_frame()

        return walk_forward_ml_signals(data, self.experiment, predictor_cols, target_col)

    def run_experiment(self, raw_ohlcv: pl.DataFrame) -> dict[str, Any]:
        """Full pipeline: feature engineering → walk-forward signals → backtest.

        Returns a dict with keys: result, model_frame, feature_diagnostics,
        signals, predictor_cols, target_col, factor_sources.
        """
        model_frame, factor_source_map, diagnostics = build_model_frame(
            raw_ohlcv,
            self.experiment.feature,
            return_diagnostics=True,
        )
        target_col = f"forward_return_{self.experiment.feature.forward_period}"
        predictor_cols = choose_predictors(
            model_frame, self.experiment.feature, self.experiment.predictor_cols
        )
        if not predictor_cols:
            raise RuntimeError(f"{self.experiment.name}: no predictor columns are available")

        signals = self.generate_signals(model_frame)
        if signals.is_empty():
            raise RuntimeError(f"{self.experiment.name}: no signals generated")

        bt_config = BacktestConfig(
            workflow="research",
            asset_types=["vn_stock"],
            universe=UniverseConfig(vn_stock=sorted(model_frame["symbol"].unique().to_list())),
            start_date=model_frame["date"].min(),
            end_date=model_frame["date"].max(),
            initial_capital=self.experiment.initial_capital,
            backtest_engine="notebook",
            rebalance_frequency=self.experiment.rebalance_period,
            commission=CommissionConfig(model="percentage", rate=Decimal("0.0015")),
            calendar="XHOSE",
        )
        result = run_backtest_frame(
            engine_name="notebook_walk_forward",
            strategy=self,
            data=model_frame,
            config=bt_config,
            prebuilt_signals=signals,
        )
        return {
            "result": result,
            "model_frame": model_frame,
            "feature_diagnostics": diagnostics,
            "signals": signals,
            "predictor_cols": predictor_cols,
            "target_col": target_col,
            "factor_sources": factor_source_map,
        }


def make_sweep_arms(base: ExperimentConfig) -> list[ExperimentConfig]:
    """Enumerate hyperparameter sweep arms from a base experiment config."""
    arms: list[ExperimentConfig] = []
    for top_n, fast, slow, num_long, depth, n_estimators, rebal in product(
        [50, 80],
        [21],
        [126, 252],
        [8, 12],
        [2, 3],
        [80],
        _SWEEP_REBALANCE_PERIODS,
    ):
        feature = FeatureConfig(
            min_trading_days=base.feature.min_trading_days,
            volume_top_n=top_n,
            min_avg_volume=base.feature.min_avg_volume,
            remove_large_gaps=base.feature.remove_large_gaps,
            max_gap_days=base.feature.max_gap_days,
            remove_low_volume=base.feature.remove_low_volume,
            qsmom_fast=fast,
            qsmom_slow=slow,
            qsmom_returns=base.feature.qsmom_returns,
            forward_period=base.feature.forward_period,
            fundamental_termtype=base.feature.fundamental_termtype,
        )
        model_params = {
            **(base.model_params or MODEL_PARAMS),
            "max_depth": depth,
            "n_estimators": n_estimators,
        }
        arms.append(
            ExperimentConfig(
                name=f"vn100_top{top_n}_qsmom{fast}_{slow}_long{num_long}_depth{depth}_rb{rebal}",
                feature=feature,
                predictor_cols=base.predictor_cols,
                train_window=base.train_window,
                rebalance_period=rebal,
                num_long_positions=num_long,
                num_short_positions=base.num_short_positions,
                long_threshold=base.long_threshold,
                short_threshold=base.short_threshold,
                model_params=model_params,
                initial_capital=base.initial_capital,
            )
        )
    return arms


__all__ = [
    "VN100QuantamentalStrategy",
    "make_sweep_arms",
]
