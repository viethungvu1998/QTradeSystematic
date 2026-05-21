"""Tests for tearsheet PDF generation."""

from __future__ import annotations

from datetime import date

import pandas as pd
import polars as pl

from qts.research.backtest.base import (
    BacktestResult,
    empty_portfolio_snapshots_frame,
    empty_trade_log_frame,
)


def _minimal_result() -> BacktestResult:
    returns = pl.DataFrame(
        {
            "date": [date(2024, 1, 2), date(2024, 1, 3)],
            "portfolio_return": [0.01, -0.005],
        }
    ).with_columns(pl.col("date").cast(pl.Date))
    equity = pl.DataFrame(
        {
            "date": [date(2024, 1, 2), date(2024, 1, 3)],
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
        trade_log=empty_trade_log_frame(),
        portfolio_snapshots=empty_portfolio_snapshots_frame(),
    )


def test_save_tearsheet_creates_pdf(tmp_path) -> None:
    from qts.research.backtest.tearsheet import save_tearsheet

    pdf = save_tearsheet(_minimal_result(), tmp_path, "test_run_001")

    assert pdf is not None
    assert pdf.exists()
    assert pdf.suffix == ".pdf"
    assert pdf.stat().st_size > 0


def test_save_tearsheet_returns_none_when_no_pyfolio(tmp_path, monkeypatch) -> None:
    import builtins
    import importlib

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "pyfolio":
            raise ImportError("pyfolio not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    from qts.research.backtest import tearsheet as tearsheet_module

    importlib.reload(tearsheet_module)

    pdf = tearsheet_module.save_tearsheet(_minimal_result(), tmp_path, "test_run_002")

    assert pdf is None


def test_save_tearsheet_with_benchmark(tmp_path) -> None:
    from qts.research.backtest.tearsheet import save_tearsheet

    benchmark = pd.Series(
        [0.005, -0.003],
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]).tz_localize("UTC"),
        name="VNINDEX",
    )

    pdf = save_tearsheet(
        _minimal_result(),
        tmp_path,
        "test_run_benchmark",
        benchmark_rets=benchmark,
    )

    assert pdf is not None
    assert pdf.exists()
