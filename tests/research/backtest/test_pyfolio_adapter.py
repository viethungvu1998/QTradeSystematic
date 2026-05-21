"""Unit tests for pyfolio adapter conversions."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import polars as pl
import pytest

from qts.research.backtest.base import (
    BacktestResult,
    empty_portfolio_snapshots_frame,
    empty_trade_log_frame,
)
from qts.research.backtest.pyfolio_adapter import (
    positions_frame,
    returns_series,
    transactions_frame,
)


def _make_result(
    returns_data=None,
    trade_log_data: pl.DataFrame | None = None,
    snapshots_data: pl.DataFrame | None = None,
) -> BacktestResult:
    if returns_data is None:
        returns_data = {
            "date": [date(2024, 1, 2), date(2024, 1, 3)],
            "portfolio_return": [0.01, -0.005],
        }
    returns = pl.DataFrame(returns_data).with_columns(pl.col("date").cast(pl.Date))
    equity = pl.DataFrame(
        {
            "date": returns_data["date"],
            "equity": [100_000.0, 101_000.0],
        }
    ).with_columns(pl.col("date").cast(pl.Date))
    signals = pl.DataFrame({"date": [], "symbol": [], "signal": [], "weight": []}).cast(
        {"date": pl.Date, "symbol": pl.String, "signal": pl.Int32, "weight": pl.Float64}
    )
    return BacktestResult(
        engine_name="vectorbt",
        metrics={
            "sharpe": 1.2,
            "sortino": 1.5,
            "cagr": 0.15,
            "max_drawdown": 0.1,
            "win_rate": 0.55,
        },
        returns=returns,
        equity_curve=equity,
        signals=signals,
        trade_log=trade_log_data if trade_log_data is not None else empty_trade_log_frame(),
        portfolio_snapshots=(
            snapshots_data if snapshots_data is not None else empty_portfolio_snapshots_frame()
        ),
    )


def test_returns_series_shape() -> None:
    series = returns_series(_make_result())

    assert isinstance(series, pd.Series)
    assert len(series) == 2
    assert series.index.tz is not None


def test_returns_series_values() -> None:
    series = returns_series(_make_result())

    assert abs(series.iloc[0] - 0.01) < 1e-9
    assert abs(series.iloc[1] - (-0.005)) < 1e-9


def test_positions_frame_empty() -> None:
    assert positions_frame(_make_result()) is None


def test_positions_frame_cash_calculation() -> None:
    snapshots = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 2, 0, 0, 0)],
            "tokens": [
                [
                    {
                        "token": "VNM",
                        "quantity": 1000.0,
                        "avg_buy_price": 80.0,
                        "current_price": 85.0,
                    }
                ]
            ],
            "equity": [190_000.0],
        }
    )

    positions = positions_frame(_make_result(snapshots_data=snapshots))

    assert positions is not None
    assert "VNM" in positions.columns
    assert "cash" in positions.columns
    assert abs(positions.iloc[0]["VNM"] - 85_000.0) < 1e-3
    assert abs(positions.iloc[0]["cash"] - 105_000.0) < 1e-3


def test_transactions_frame_empty() -> None:
    assert transactions_frame(_make_result()) is None


@pytest.mark.parametrize("side", ["BUY", "buy"])
def test_transactions_frame_long_roundtrip(side: str) -> None:
    trade_log = pl.DataFrame(
        {
            "ticker": ["VNM"],
            "entry_time": [datetime(2024, 1, 2, 9, 0)],
            "exit_time": [datetime(2024, 1, 10, 9, 0)],
            "start_price": [80.0],
            "end_price": [90.0],
            "quantity": [1000.0],
            "profit_pct": [0.125],
            "fee": [200.0],
            "side": [side],
        }
    )

    transactions = transactions_frame(_make_result(trade_log_data=trade_log))

    assert transactions is not None
    assert len(transactions) == 2
    assert transactions[transactions["order_id"] == "0_entry"].iloc[0]["amount"] > 0
    assert transactions[transactions["order_id"] == "0_exit"].iloc[0]["amount"] < 0
