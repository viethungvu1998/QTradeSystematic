"""Vectorized backtest engine using vectorbtpro.Portfolio.from_orders."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import polars as pl

from qts.core.registry import Registry
from qts.research.backtest._runner import walk_forward_signals
from qts.research.backtest.base import (
    BacktestConfig,
    BacktestResult,
    BaseEngine,
    empty_backtest_result,
    empty_portfolio_snapshots_frame,
    empty_trade_log_frame,
)
from qts.research.backtest.engines._targets import build_target_schedule
from qts.research.backtest.metrics import build_metrics
from qts.research.backtest.observability import vectorbt_observability
from qts.research.strategies.base import BaseStrategy
from qts.research.strategies.stat_arb.base import BaseStatArbStrategy


def _pivot_wide(df: pl.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot Polars long frame → pandas wide (DatetimeIndex × symbol)."""
    wide = (
        df.select(["date", "symbol", value_col])
        .to_pandas()
        .pivot(index="date", columns="symbol", values=value_col)
        .rename_axis(index=None, columns=None)
        .sort_index()
        .sort_index(axis=1)
    )
    wide.index = pd.to_datetime(wide.index)
    return wide


def _pivot_close(df: pl.DataFrame) -> pd.DataFrame:
    return _pivot_wide(df, "close")


def _pivot_open(df: pl.DataFrame) -> pd.DataFrame:
    return _pivot_wide(df, "open")


def _pivot_high(df: pl.DataFrame) -> pd.DataFrame:
    return _pivot_wide(df, "high")


def _pivot_low(df: pl.DataFrame) -> pd.DataFrame:
    return _pivot_wide(df, "low")


def _extract_fees(config: BacktestConfig) -> float:
    if config.commission is None:
        return 0.001  # Binance taker default 0.1 %
    return float(config.commission.rate)


def _extract_slippage(config: BacktestConfig) -> float:
    if config.slippage_model is None:
        return 0.0
    slippage_map = {
        "fixed": 0.0005,
        "volatility_scaled": 0.001,
    }
    return slippage_map.get(config.slippage_model, 0.0)


def _vbt_pf_to_result(pf, signals: pl.DataFrame) -> BacktestResult:
    """Extract BacktestResult from a vbt Portfolio object."""
    val_series = pf.get_value(group_by=True)
    ret_series = pf.get_returns(group_by=True)
    metrics = build_metrics(ret_series.to_list(), val_series.to_list())

    returns_df = pl.from_pandas(
        pd.DataFrame(
            {
                "date": ret_series.index.date,
                "portfolio_return": ret_series.to_numpy(),
            }
        )
    ).with_columns(pl.col("date").cast(pl.Date))

    equity_df = pl.from_pandas(
        pd.DataFrame(
            {
                "date": val_series.index.date,
                "equity": val_series.to_numpy(),
            }
        )
    ).with_columns(pl.col("date").cast(pl.Date))
    try:
        trade_log, portfolio_snapshots = vectorbt_observability(pf, val_series)
    except Exception:
        trade_log = empty_trade_log_frame()
        portfolio_snapshots = empty_portfolio_snapshots_frame()

    return BacktestResult(
        engine_name="vectorbt",
        metrics=metrics,
        returns=returns_df,
        equity_curve=equity_df,
        signals=signals,
        trade_log=trade_log,
        portfolio_snapshots=portfolio_snapshots,
    )


@Registry.register_engine("vectorbt")
class VectorBTProEngine(BaseEngine):
    """Vectorized engine backed by vectorbtpro.Portfolio.from_orders (TargetPercent)."""

    def run(
        self,
        strategy: BaseStrategy,
        data: pl.DataFrame,
        config: BacktestConfig,
        *,
        pipeline=None,
        ohlcv: pl.DataFrame | None = None,
    ) -> BacktestResult:
        import vectorbtpro as vbt  # noqa: PLC0415 — vendor SDK confined to this file

        # 1. Build walk-forward signals or one-shot signals
        if pipeline is not None and ohlcv is not None:
            signals = walk_forward_signals(pipeline, strategy, ohlcv, config)
        else:
            signals = strategy.generate_signals(data).sort(["symbol", "date"])

        # 2. Pivot close prices to wide pandas
        close_wide = _pivot_close(data)
        open_wide = _pivot_open(data)
        high_wide = _pivot_high(data)
        low_wide = _pivot_low(data)

        # Guard: nothing to simulate if data was filtered away (e.g. too-short history)
        if close_wide.empty or close_wide.shape[1] == 0:
            return empty_backtest_result(engine_name="vectorbt", signals=signals)

        # 3. Build sparse target-change orders from the shared signal frame.
        shift_to_next_session = isinstance(strategy, BaseStatArbStrategy)
        target_schedule = build_target_schedule(
            signals,
            close_wide.index,
            close_wide.columns.tolist(),
            shift_by_one_bar=shift_to_next_session,
        )
        size_matrix = target_schedule.events

        # 4. Run vectorized simulation via from_orders (supports TargetPercent)
        pf = vbt.Portfolio.from_orders(
            close=close_wide,
            size=size_matrix,
            size_type="TargetPercent",
            price=open_wide if shift_to_next_session else close_wide,
            val_price=close_wide if shift_to_next_session else None,
            open=open_wide,
            high=high_wide,
            low=low_wide,
            fees=_extract_fees(config),
            slippage=_extract_slippage(config),
            cash_sharing=True,
            group_by=True,
            init_cash=float(config.initial_capital or Decimal("100000")),
            freq="1D",
        )

        # 5. Package into BacktestResult
        return _vbt_pf_to_result(pf, signals)
