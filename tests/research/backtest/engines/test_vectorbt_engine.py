from __future__ import annotations

import pandas as pd
import polars as pl

from qts.research.backtest.engines.vectorbtpro_engine import _vbt_pf_to_result


def test_vbt_pf_to_result_populates_observability_frames():
    index = pd.date_range("2024-01-01", periods=2, freq="D")
    pf = MockPortfolio(index)

    result = _vbt_pf_to_result(pf, _signals())

    assert result.trade_log.height == 1
    trade = result.trade_log.row(0, named=True)
    assert trade["ticker"] == "AAPL"
    assert trade["quantity"] == 2.0
    assert trade["side"] == "buy"
    assert result.portfolio_snapshots.height == 2
    assert result.portfolio_snapshots["tokens"][0][0] == {
        "token": "AAPL",
        "quantity": 10.0,
        "avg_buy_price": 10.0,
        "current_price": 10.0,
    }


def test_vbt_pf_to_result_observability_falls_back_to_empty_frames():
    index = pd.date_range("2024-01-01", periods=2, freq="D")
    pf = MockPortfolio(index, broken_trades=True)

    result = _vbt_pf_to_result(pf, _signals())

    assert result.trade_log.height == 0
    assert result.portfolio_snapshots.height == 0


class MockPortfolio:
    def __init__(self, index: pd.DatetimeIndex, *, broken_trades: bool = False) -> None:
        self._index = index
        self.trades = BrokenTrades() if broken_trades else Trades(index)
        self.close = pd.DataFrame({"AAPL": [10.0, 11.0]}, index=index)

    def get_value(self, group_by: bool = True) -> pd.Series:
        return pd.Series([100.0, 110.0], index=self._index)

    def get_returns(self, group_by: bool = True) -> pd.Series:
        return pd.Series([0.0, 0.1], index=self._index)

    def get_asset_value(self, group_by: bool = False) -> pd.DataFrame:
        return pd.DataFrame({"AAPL": [100.0, 0.0]}, index=self._index)


class Trades:
    def __init__(self, index: pd.DatetimeIndex) -> None:
        self.records_readable = pd.DataFrame(
            {
                "Column": ["AAPL"],
                "Entry Index": [index[0]],
                "Exit Index": [index[1]],
                "Entry Price": [10.0],
                "Exit Price": [11.0],
                "Return": [0.1],
                "Size": [2.0],
                "Direction": ["Long"],
            }
        )


class BrokenTrades:
    @property
    def records_readable(self):
        raise RuntimeError("vbt trade extraction failed")


def _signals() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01").date()],
            "symbol": ["AAPL"],
            "signal": [1],
            "weight": [1.0],
        }
    )
