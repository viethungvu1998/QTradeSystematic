"""Convert BacktestResult Polars frames to pyfolio-ready pandas objects."""

from __future__ import annotations

from datetime import UTC

import pandas as pd

from qts.research.backtest.base import BacktestResult


def returns_series(result: BacktestResult) -> pd.Series:
    """Daily portfolio returns as a UTC-indexed pandas Series."""

    df = result.returns.to_pandas()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    series = df.set_index("date")["portfolio_return"]
    series.name = "portfolio"
    return series


def positions_frame(result: BacktestResult) -> pd.DataFrame | None:
    """Wide UTC-indexed position market values plus cash."""

    snapshots = result.portfolio_snapshots
    if snapshots.is_empty():
        return None

    records: list[dict[str, object]] = []
    for row in snapshots.iter_rows(named=True):
        equity = float(row["equity"])
        tokens = row["tokens"] or []

        pos_row: dict[str, object] = {"timestamp": _utc_timestamp(row["timestamp"])}
        total_market_value = 0.0
        for token in tokens:
            market_value = float(token["quantity"]) * float(token["current_price"])
            pos_row[str(token["token"])] = market_value
            total_market_value += market_value
        pos_row["cash"] = equity - total_market_value
        records.append(pos_row)

    return pd.DataFrame(records).set_index("timestamp").fillna(0.0)


def transactions_frame(result: BacktestResult) -> pd.DataFrame | None:
    """pyfolio transactions DataFrame derived from round-trip trade logs."""

    trade_log = result.trade_log
    if trade_log.is_empty():
        return None

    rows: list[dict[str, object]] = []
    for index, row in enumerate(trade_log.iter_rows(named=True)):
        symbol = row["ticker"]
        fee = float(row["fee"] or 0.0)
        quantity = float(row["quantity"])
        is_long = str(row["side"]).upper() == "BUY"
        entry_amount = quantity if is_long else -quantity
        exit_amount = -quantity if is_long else quantity

        rows.append(
            {
                "timestamp": _utc_timestamp(row["entry_time"]),
                "sid": symbol,
                "symbol": symbol,
                "price": float(row["start_price"]),
                "order_id": f"{index}_entry",
                "amount": entry_amount,
                "commission": fee / 2.0,
            }
        )

        if row["exit_time"] is not None:
            rows.append(
                {
                    "timestamp": _utc_timestamp(row["exit_time"]),
                    "sid": symbol,
                    "symbol": symbol,
                    "price": float(row["end_price"]),
                    "order_id": f"{index}_exit",
                    "amount": exit_amount,
                    "commission": fee / 2.0,
                }
            )

    return pd.DataFrame(rows).set_index("timestamp").sort_index()


def _utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(UTC)
    return timestamp.tz_convert(UTC)
