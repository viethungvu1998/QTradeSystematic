from __future__ import annotations

import pandas as pd
import polars as pl

from qts.research.backtest.engines.zipline_engine import _perf_to_result


def test_perf_to_result_builds_observability_with_fifo_pairing():
    index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    perf = pd.DataFrame(
        {
            "returns": [0.0, 0.01, 0.02],
            "portfolio_value": [100.0, 110.0, 120.0],
            "transactions": [
                [_fill(1.0, 100.0, index[0])],
                [_fill(2.0, 105.0, index[1])],
                [_fill(-2.0, 110.0, index[2])],
            ],
            "positions": [
                {"AAPL_Z": {"amount": 1.0, "last_sale_price": 100.0, "cost_basis": 100.0}},
                {"AAPL_Z": {"amount": 3.0, "last_sale_price": 105.0, "cost_basis": 103.3}},
                {"AAPL_Z": {"amount": 1.0, "last_sale_price": 110.0, "cost_basis": 105.0}},
            ],
        },
        index=index,
    )

    result = _perf_to_result(perf, _signals(), symbol_map={"AAPL": "AAPL_Z"})

    assert result.trade_log.height == 2
    assert result.trade_log["ticker"].to_list() == ["AAPL", "AAPL"]
    assert result.trade_log["start_price"].to_list() == [100.0, 105.0]
    assert result.trade_log["quantity"].to_list() == [1.0, 1.0]
    assert result.portfolio_snapshots.height == 3
    assert result.portfolio_snapshots["tokens"][0][0]["token"] == "AAPL"
    assert result.portfolio_snapshots["tokens"][1][0]["quantity"] == 3.0


def _fill(amount: float, price: float, timestamp: pd.Timestamp) -> dict[str, object]:
    return {
        "amount": amount,
        "price": price,
        "dt": timestamp,
        "symbol": "AAPL_Z",
    }


def _signals() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01").date()],
            "symbol": ["AAPL"],
            "signal": [1],
            "weight": [1.0],
        }
    )
