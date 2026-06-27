"""Export helpers for backtest and live artifacts."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from qts.core.observability import TokenSnapshot, snapshot_portfolio
from qts.core.portfolio import Portfolio
from qts.research.backtest.base import (
    PORTFOLIO_SNAPSHOTS_SCHEMA,
    TOKEN_SNAPSHOT_SCHEMA,
    TRADE_LOG_SCHEMA,
    BacktestResult,
)
from qts.research.backtest.tearsheet import save_tearsheet
from qts.utils.dataframe import serialise_list_columns
from qts.utils.date_intervals import DateInterval

LIVE_PORTFOLIO_SCHEMA = {
    "timestamp": pl.Datetime(time_unit="us"),
    "tokens": pl.List(TOKEN_SNAPSHOT_SCHEMA),
    "equity": pl.Float64,
    "cash": pl.Float64,
}


def export_trade_log(result: BacktestResult, path: Path) -> None:
    _validate_schema(result.trade_log, TRADE_LOG_SCHEMA, "trade_log")
    result.trade_log.write_csv(path)


def export_portfolio_snapshots(result: BacktestResult, path: Path) -> None:
    _validate_schema(
        result.portfolio_snapshots,
        PORTFOLIO_SNAPSHOTS_SCHEMA,
        "portfolio_snapshots",
    )
    _write_csv(result.portfolio_snapshots, path)


def export_backtest_report_pdf(
    result: BacktestResult,
    *,
    run_id: str,
    output_dir: Path,
    benchmark_rets: Any = None,
    tearsheet_out_dir: Path | None = None,
    filename: str | None = None,
    test_intervals: list[DateInterval] | None = None,
) -> Path | None:
    """Generate a tearsheet PDF and copy it to a stable report output path."""

    source_dir = tearsheet_out_dir if tearsheet_out_dir is not None else output_dir
    save_kwargs = {"benchmark_rets": benchmark_rets}
    if test_intervals is not None:
        save_kwargs["test_intervals"] = test_intervals
    tearsheet_path = save_tearsheet(result, source_dir, run_id, **save_kwargs)
    if tearsheet_path is None:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / (filename or f"{run_id}_report.pdf")
    if tearsheet_path.resolve() != report_path.resolve():
        shutil.copy2(tearsheet_path, report_path)
    return report_path


def export_live_portfolio(portfolio: Portfolio, ts: datetime, path: Path) -> None:
    snapshot = snapshot_portfolio(portfolio, ts)
    frame = pl.DataFrame(
        {
            "timestamp": [snapshot.timestamp],
            "tokens": [[_token_snapshot_to_record(token) for token in snapshot.tokens]],
            "equity": [float(snapshot.equity)],
            "cash": [float(portfolio.cash)],
        },
        schema=LIVE_PORTFOLIO_SCHEMA,
    )
    _validate_schema(frame, LIVE_PORTFOLIO_SCHEMA, "live_portfolio")
    _write_csv(frame, path)


def _validate_schema(
    frame: pl.DataFrame,
    expected_schema: Mapping[str, pl.DataType],
    name: str,
) -> None:
    if list(frame.schema.items()) != list(expected_schema.items()):
        raise ValueError(
            f"{name} schema mismatch: expected {dict(expected_schema)}, got {dict(frame.schema)}"
        )


def _write_csv(frame: pl.DataFrame, path: Path) -> None:
    csv_frame = serialise_list_columns(frame)
    csv_frame.write_csv(path)


def _token_snapshot_to_record(token: TokenSnapshot) -> dict[str, str | float]:
    return {
        "token": token.token,
        "quantity": float(token.quantity),
        "avg_buy_price": float(token.avg_buy_price),
        "current_price": float(token.current_price),
    }
