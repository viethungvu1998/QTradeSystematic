from __future__ import annotations

import json
from datetime import date
from decimal import Decimal

import polars as pl
import pytest

from qts.research.backtest.base import BacktestConfig, CommissionConfig, UniverseConfig
from qts.research.backtest.vanilla import (
    export_vanilla_backtest_artifacts,
    run_vanilla_backtest_from_signals,
    save_vanilla_comparison_report_pdf,
)


def test_vanilla_backtest_rebalances_from_signals_with_hand_computed_equity() -> None:
    data = pl.DataFrame(
        {
            "date": [
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 4),
            ]
            * 2,
            "symbol": ["AAA"] * 4 + ["BBB"] * 4,
            "open": [100.0, 110.0, 120.0, 130.0, 200.0, 190.0, 180.0, 170.0],
            "high": [100.0, 110.0, 120.0, 130.0, 200.0, 190.0, 180.0, 170.0],
            "low": [100.0, 110.0, 120.0, 130.0, 200.0, 190.0, 180.0, 170.0],
            "close": [100.0, 110.0, 120.0, 130.0, 200.0, 190.0, 180.0, 170.0],
            "volume": [1_000.0] * 8,
        }
    )
    signals = pl.DataFrame(
        {
            "date": [
                date(2024, 1, 1),
                date(2024, 1, 1),
                date(2024, 1, 3),
                date(2024, 1, 3),
            ],
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "signal": [1, 1, 0, 1],
            "weight": [0.5, 0.5, 0.0, 1.0],
        }
    )
    config = _config(initial_capital=Decimal("1000"), commission_rate=Decimal("0"))

    artifacts = run_vanilla_backtest_from_signals(data=data, signals=signals, config=config)

    equity = artifacts.result.equity_curve.to_pandas()["equity"].tolist()
    assert equity == pytest.approx([1000.0, 1025.0, 1050.0, 991.6666666667])
    assert artifacts.portfolio_log["date"].tolist() == ["2024-01-01", "2024-01-03"]
    assert artifacts.entry_log["date"].tolist() == ["2024-01-01", "2024-01-03"]
    assert json.loads(artifacts.entry_log.loc[0, "entries"]) == ["AAA", "BBB"]
    assert json.loads(artifacts.entry_log.loc[1, "exits"]) == ["AAA"]
    assert artifacts.portfolio_log.loc[1, "portfolio_value"] == pytest.approx(1050.0)


def test_vanilla_backtest_applies_fee_and_keeps_cash_non_negative() -> None:
    data = pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "symbol": ["AAA", "AAA"],
            "open": [100.0, 100.0],
            "high": [100.0, 100.0],
            "low": [100.0, 100.0],
            "close": [100.0, 100.0],
            "volume": [1_000.0, 1_000.0],
        }
    )
    signals = pl.DataFrame(
        {
            "date": [date(2024, 1, 1)],
            "symbol": ["AAA"],
            "signal": [1],
            "weight": [1.0],
        }
    )
    config = _config(initial_capital=Decimal("1000"), commission_rate=Decimal("0.001"))

    artifacts = run_vanilla_backtest_from_signals(data=data, signals=signals, config=config)

    row = artifacts.portfolio_log.iloc[0]
    first_return = artifacts.result.returns.to_pandas()["portfolio_return"].iloc[0]
    assert row["fee"] == pytest.approx(0.9990009990)
    assert row["cash_after_rebalance"] == pytest.approx(0.0, abs=1e-9)
    assert row["portfolio_value"] == pytest.approx(999.000999001)
    assert first_return == pytest.approx(-0.000999000999)
    assert row["cash_scaled_buy_orders"]


def test_vanilla_exports_and_comparison_report(tmp_path) -> None:
    data = pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "symbol": ["AAA", "AAA"],
            "open": [100.0, 101.0],
            "high": [100.0, 101.0],
            "low": [100.0, 101.0],
            "close": [100.0, 101.0],
            "volume": [1_000.0, 1_000.0],
        }
    )
    signals = pl.DataFrame(
        {
            "date": [date(2024, 1, 1)],
            "symbol": ["AAA"],
            "signal": [1],
            "weight": [1.0],
        }
    )
    artifacts = run_vanilla_backtest_from_signals(
        data=data,
        signals=signals,
        config=_config(initial_capital=Decimal("1000"), commission_rate=Decimal("0")),
    )

    paths = export_vanilla_backtest_artifacts(artifacts, out_dir=tmp_path, run_id="unit")
    report_path = save_vanilla_comparison_report_pdf(
        vectorbt_result=artifacts.result,
        vanilla_artifacts=artifacts,
        run_id="unit",
        output_dir=tmp_path,
        exported_paths=paths,
    )

    assert paths["portfolio_log"].exists()
    assert paths["entry_log"].exists()
    assert pl.read_csv(paths["portfolio_log"]).height == 1
    assert pl.read_csv(paths["entry_log"]).height == 1
    assert report_path.exists()
    assert report_path.stat().st_size > 0


def _config(*, initial_capital: Decimal, commission_rate: Decimal) -> BacktestConfig:
    return BacktestConfig(
        workflow="research",
        asset_types=["stock"],
        universe=UniverseConfig(stock=["AAA", "BBB"]),
        initial_capital=initial_capital,
        commission=CommissionConfig(model="percentage", rate=commission_rate),
    )
