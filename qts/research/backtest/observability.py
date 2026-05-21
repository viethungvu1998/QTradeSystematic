"""Backtest observability extraction helpers."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import polars as pl

from qts.core.order import OrderSide
from qts.research.backtest.base import PORTFOLIO_SNAPSHOTS_SCHEMA, TRADE_LOG_SCHEMA


def backtest_frame_observability(
    joined: pl.DataFrame,
    daily: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    return _runner_trade_log(joined), _runner_portfolio_snapshots(joined, daily)


def vectorbt_observability(pf: Any, value_series: pd.Series) -> tuple[pl.DataFrame, pl.DataFrame]:
    return _vbt_trade_log(pf), _vbt_portfolio_snapshots(pf, value_series)


def _runner_portfolio_snapshots(joined: pl.DataFrame, daily: pl.DataFrame) -> pl.DataFrame:
    tokens_by_date: dict[date, list[dict[str, float | str]]] = {}
    for row in joined.sort(["date", "symbol"]).iter_rows(named=True):
        target = float(row["signal"]) * float(row["weight"])
        if target == 0.0:
            continue
        close = float(row["close"])
        tokens_by_date.setdefault(row["date"], []).append(
            {
                "token": str(row["symbol"]),
                "quantity": target,
                # The sequential runner has weights, not fills or cost basis.
                "avg_buy_price": close,
                "current_price": close,
            }
        )
    rows = [
        {
            "timestamp": _as_datetime(row["date"]),
            "tokens": tokens_by_date.get(row["date"], []),
            "equity": float(row["equity"]),
        }
        for row in daily.iter_rows(named=True)
    ]
    return _frame(rows, PORTFOLIO_SNAPSHOTS_SCHEMA)


def _runner_trade_log(joined: pl.DataFrame) -> pl.DataFrame:
    # The sequential runner has signals only, so trades are inferred from transitions.
    rows: list[dict[str, object]] = []
    for symbol in sorted(joined["symbol"].unique().to_list()):
        open_trade: dict[str, object] | None = None
        symbol_rows = joined.filter(pl.col("symbol") == symbol).sort("date")
        for row in symbol_rows.iter_rows(named=True):
            target = float(row["signal"]) * float(row["weight"])
            if target == 0.0:
                if open_trade is not None:
                    rows.append(_close_runner_trade(open_trade, row))
                    open_trade = None
                continue
            if open_trade is None:
                open_trade = _open_runner_trade(symbol, row, target)
            elif _sign(target) != _sign(float(open_trade["target"])):
                rows.append(_close_runner_trade(open_trade, row))
                open_trade = _open_runner_trade(symbol, row, target)
    return _frame(rows, TRADE_LOG_SCHEMA)


def _open_runner_trade(symbol: str, row: dict[str, object], target: float) -> dict[str, object]:
    return {
        "ticker": symbol,
        "entry_time": _as_datetime(row["date"]),
        "start_price": float(row["close"]),
        "quantity": abs(target),
        "side": OrderSide.BUY.value if target > 0 else OrderSide.SELL.value,
        "target": target,
    }


def _close_runner_trade(open_trade: dict[str, object], row: dict[str, object]) -> dict[str, object]:
    start_price = float(open_trade["start_price"])
    end_price = float(row["close"])
    return {
        "ticker": open_trade["ticker"],
        "entry_time": open_trade["entry_time"],
        "exit_time": _as_datetime(row["date"]),
        "start_price": start_price,
        "end_price": end_price,
        "quantity": float(open_trade["quantity"]),
        "profit_pct": 0.0 if start_price == 0.0 else (end_price - start_price) / start_price,
        # This signal-transition approximation has no commission model.
        "fee": 0.0,
        "side": open_trade["side"],
    }


def _vbt_trade_log(pf: Any) -> pl.DataFrame:
    trades = pd.DataFrame(pf.trades.records_readable)
    rows = []
    for _, row in trades.iterrows():
        direction = str(row.get("Direction", "Long"))
        rows.append(
            {
                "ticker": str(row["Column"]),
                "entry_time": _as_datetime(row["Entry Index"]),
                "exit_time": _as_datetime(row["Exit Index"]),
                "start_price": float(row["Entry Price"]),
                "end_price": float(row["Exit Price"]),
                "quantity": float(row["Size"]),
                "profit_pct": float(row["Return"]),
                # vbt net returns already reflect simulation fees.
                "fee": 0.0,
                "side": OrderSide.BUY.value if direction == "Long" else OrderSide.SELL.value,
            }
        )
    return _frame(rows, TRADE_LOG_SCHEMA)


def _vbt_portfolio_snapshots(pf: Any, value_series: pd.Series) -> pl.DataFrame:
    asset_values = _as_dataframe(pf.get_asset_value(group_by=False))
    close = _as_dataframe(pf.close)
    rows = []
    for timestamp, asset_row in asset_values.iterrows():
        tokens = []
        for symbol, asset_value in asset_row.items():
            value = float(asset_value)
            if value == 0.0 or pd.isna(value):
                continue
            price = float(close.loc[timestamp, symbol])
            tokens.append(
                {
                    "token": str(symbol),
                    "quantity": 0.0 if price == 0.0 else value / price,
                    "avg_buy_price": price,
                    "current_price": price,
                }
            )
        rows.append(
            {
                "timestamp": _as_datetime(timestamp),
                "tokens": tokens,
                "equity": float(value_series.loc[timestamp]),
            }
        )
    return _frame(rows, PORTFOLIO_SNAPSHOTS_SCHEMA)


def _as_datetime(value: object) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(UTC)
    else:
        timestamp = timestamp.tz_convert(UTC)
    return timestamp.to_pydatetime()


def _as_dataframe(value: object) -> pd.DataFrame:
    if isinstance(value, pd.Series):
        return value.to_frame()
    return pd.DataFrame(value)


def _frame(records: list[dict[str, object]], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(records, schema=schema) if records else pl.DataFrame(schema=schema)


def _sign(value: float) -> float:
    return 1.0 if value > 0 else -1.0
