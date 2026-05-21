"""Zipline observability extraction helpers."""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime

import pandas as pd
import polars as pl

from qts.core.order import OrderSide
from qts.research.backtest.base import PORTFOLIO_SNAPSHOTS_SCHEMA, TRADE_LOG_SCHEMA


def zipline_observability(
    perf: pd.DataFrame,
    symbol_map: dict[str, str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    return _trade_log(perf, symbol_map), _portfolio_snapshots(perf, symbol_map)


def _trade_log(perf: pd.DataFrame, symbol_map: dict[str, str] | None) -> pl.DataFrame:
    reverse_map = _reverse_symbol_map(symbol_map)
    open_fills: dict[str, deque[dict[str, object]]] = {}
    rows: list[dict[str, object]] = []
    for fills in perf.get("transactions", pd.Series(dtype=object)):
        if not isinstance(fills, list | tuple):
            continue
        for fill in fills or []:
            amount = float(fill.get("amount", 0.0))
            if amount == 0.0:
                continue
            symbol = _mapped_symbol(fill.get("symbol"), reverse_map)
            queue = open_fills.setdefault(symbol, deque())
            remaining = abs(amount)
            while queue and _sign(amount) != _sign(float(queue[0]["amount"])) and remaining > 0.0:
                entry = queue[0]
                matched = min(remaining, abs(float(entry["amount"])))
                rows.append(_closed_trade(symbol, entry, fill, matched))
                entry["amount"] = _sign(float(entry["amount"])) * (
                    abs(float(entry["amount"])) - matched
                )
                remaining -= matched
                if abs(float(entry["amount"])) == 0.0:
                    queue.popleft()
            if remaining > 0.0:
                queue.append({**fill, "amount": _sign(amount) * remaining})
    return _frame(rows, TRADE_LOG_SCHEMA)


def _portfolio_snapshots(
    perf: pd.DataFrame,
    symbol_map: dict[str, str] | None,
) -> pl.DataFrame:
    reverse_map = _reverse_symbol_map(symbol_map)
    rows = []
    for timestamp, row in perf.iterrows():
        positions = row.get("positions", {}) or {}
        if not isinstance(positions, dict):
            positions = {}
        rows.append(
            {
                "timestamp": _as_datetime(timestamp),
                "tokens": _tokens(positions, reverse_map),
                "equity": float(row["portfolio_value"]),
            }
        )
    return _frame(rows, PORTFOLIO_SNAPSHOTS_SCHEMA)


def _tokens(
    positions: dict[object, dict[str, object]],
    reverse_map: dict[str, str],
) -> list[dict[str, float | str]]:
    rows = []
    for raw_symbol, position in positions.items():
        amount = float(position.get("amount", 0.0))
        if amount == 0.0:
            continue
        current_price = float(position.get("last_sale_price", 0.0))
        rows.append(
            {
                "token": _mapped_symbol(raw_symbol, reverse_map),
                "quantity": amount,
                "avg_buy_price": float(position.get("cost_basis", current_price)),
                "current_price": current_price,
            }
        )
    return rows


def _closed_trade(
    symbol: str,
    entry: dict[str, object],
    exit_fill: dict[str, object],
    quantity: float,
) -> dict[str, object]:
    start_price = float(entry["price"])
    end_price = float(exit_fill["price"])
    return {
        "ticker": symbol,
        "entry_time": _as_datetime(entry["dt"]),
        "exit_time": _as_datetime(exit_fill["dt"]),
        "start_price": start_price,
        "end_price": end_price,
        "quantity": quantity,
        "profit_pct": 0.0 if start_price == 0.0 else (end_price - start_price) / start_price,
        # Zipline transactions are fills, so FIFO pairing is only an approximation.
        "fee": 0.0,
        "side": OrderSide.BUY.value if float(entry["amount"]) > 0 else OrderSide.SELL.value,
    }


def _as_datetime(value: object) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(UTC)
    else:
        timestamp = timestamp.tz_convert(UTC)
    return timestamp.to_pydatetime()


def _frame(records: list[dict[str, object]], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(records, schema=schema) if records else pl.DataFrame(schema=schema)


def _mapped_symbol(value: object, reverse_map: dict[str, str]) -> str:
    raw = getattr(value, "symbol", value)
    symbol = str(raw)
    return reverse_map.get(symbol, symbol)


def _reverse_symbol_map(symbol_map: dict[str, str] | None) -> dict[str, str]:
    return {mapped: original for original, mapped in (symbol_map or {}).items()}


def _sign(value: float) -> float:
    return 1.0 if value > 0 else -1.0
