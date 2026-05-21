from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal

import polars as pl
import pytest

from qts.core.instrument import AssetType, Instrument
from qts.core.portfolio import Portfolio, Position
from qts.research.backtest.base import (
    PORTFOLIO_SNAPSHOTS_SCHEMA,
    TRADE_LOG_SCHEMA,
    BacktestResult,
    empty_backtest_result,
)
from qts.utils.export import (
    export_live_portfolio,
    export_portfolio_snapshots,
    export_trade_log,
)


def test_export_trade_log_writes_readable_csv(tmp_path):
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    trade_log = pl.DataFrame(
        {
            "ticker": ["AAPL"],
            "entry_time": [timestamp],
            "exit_time": [timestamp],
            "start_price": [100.0],
            "end_price": [110.0],
            "quantity": [2.0],
            "profit_pct": [0.1],
            "fee": [1.0],
            "side": ["buy"],
        },
        schema=TRADE_LOG_SCHEMA,
    )
    result = _result(trade_log=trade_log)
    path = tmp_path / "trade_log.csv"

    export_trade_log(result, path)

    exported = pl.read_csv(path)
    assert exported.columns == list(TRADE_LOG_SCHEMA)
    assert exported["ticker"].to_list() == ["AAPL"]


def test_export_trade_log_rejects_wrong_schema(tmp_path):
    result = _result(trade_log=pl.DataFrame({"ticker": ["AAPL"]}))

    with pytest.raises(ValueError, match="trade_log schema mismatch"):
        export_trade_log(result, tmp_path / "bad.csv")


def test_export_portfolio_snapshots_writes_readable_csv(tmp_path):
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    snapshots = pl.DataFrame(
        {
            "timestamp": [timestamp],
            "tokens": [
                [
                    {
                        "token": "AAPL",
                        "quantity": 2.0,
                        "avg_buy_price": 45.0,
                        "current_price": 50.0,
                    },
                    {
                        "token": "MSFT",
                        "quantity": 1.0,
                        "avg_buy_price": 20.0,
                        "current_price": 25.0,
                    },
                ]
            ],
            "equity": [100_000.0],
        },
        schema=PORTFOLIO_SNAPSHOTS_SCHEMA,
    )
    result = _result(portfolio_snapshots=snapshots)
    path = tmp_path / "portfolio_snapshots.csv"

    export_portfolio_snapshots(result, path)

    exported = pl.read_csv(path)
    assert exported.columns == list(PORTFOLIO_SNAPSHOTS_SCHEMA)
    assert json.loads(exported["tokens"][0]) == [
        {
            "token": "AAPL",
            "quantity": 2.0,
            "avg_buy_price": 45.0,
            "current_price": 50.0,
        },
        {
            "token": "MSFT",
            "quantity": 1.0,
            "avg_buy_price": 20.0,
            "current_price": 25.0,
        },
    ]


def test_export_live_portfolio_writes_cash_column(tmp_path):
    instrument = Instrument("AAPL", AssetType.STOCK, "NASDAQ", "USD")
    portfolio = Portfolio(
        positions=[
            Position(
                instrument,
                Decimal("2"),
                Decimal("50"),
                average_cost=Decimal("45"),
            )
        ],
        cash=Decimal("10"),
    )
    path = tmp_path / "live_portfolio.csv"

    export_live_portfolio(
        portfolio,
        datetime(2024, 1, 1, tzinfo=UTC),
        path,
    )

    exported = pl.read_csv(path)
    assert exported.columns == ["timestamp", "tokens", "equity", "cash"]
    assert exported.height == 1
    assert exported["cash"].to_list() == [10.0]
    assert json.loads(exported["tokens"][0]) == [
        {
            "token": "AAPL",
            "quantity": 2.0,
            "avg_buy_price": 45.0,
            "current_price": 50.0,
        }
    ]


def test_empty_backtest_result_has_observability_schemas():
    result = empty_backtest_result()

    assert result.trade_log.height == 0
    assert result.trade_log.schema == TRADE_LOG_SCHEMA
    assert result.portfolio_snapshots.height == 0
    assert result.portfolio_snapshots.schema == PORTFOLIO_SNAPSHOTS_SCHEMA


def _result(
    *,
    trade_log: pl.DataFrame | None = None,
    portfolio_snapshots: pl.DataFrame | None = None,
) -> BacktestResult:
    return BacktestResult(
        engine_name="test",
        metrics={},
        returns=pl.DataFrame(schema={"date": pl.Date, "portfolio_return": pl.Float64}),
        equity_curve=pl.DataFrame(schema={"date": pl.Date, "equity": pl.Float64}),
        signals=pl.DataFrame(),
        trade_log=trade_log if trade_log is not None else pl.DataFrame(schema=TRADE_LOG_SCHEMA),
        portfolio_snapshots=(
            portfolio_snapshots
            if portfolio_snapshots is not None
            else pl.DataFrame(schema=PORTFOLIO_SNAPSHOTS_SCHEMA)
        ),
    )
