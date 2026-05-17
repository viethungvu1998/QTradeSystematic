from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import polars as pl
import pytest

from qts.research.backtest._runner import (
    _rebalance_dates,
    run_backtest_frame,
    walk_forward_signals,
)
from qts.research.backtest.base import (
    BacktestConfig,
    DataSourcesConfig,
    FeaturesConfig,
    StrategyConfig,
    UniverseConfig,
)
from qts.research.backtest.engines.vectorbtpro_engine import VectorBTProEngine
from qts.research.backtest.engines.zipline_engine import ZiplineEngine
from qts.research.features.pipeline import FeaturePipeline


def _ohlcv_frame(start: date, days: int, symbol: str = "AAPL") -> pl.DataFrame:
    rows = []
    for index in range(days):
        current = start + timedelta(days=index)
        price = 100.0 + index
        rows.append(
            {
                "date": current,
                "symbol": symbol,
                "open": price,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price + 0.5,
                "volume": 1_000.0 + index,
            }
        )
    return pl.DataFrame(rows)


def _config(train_window: int = 20, rebalance_frequency: str = "monthly") -> BacktestConfig:
    return BacktestConfig(
        workflow="research",
        asset_types=["stock"],
        universe=UniverseConfig(stock=["AAPL"]),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 31),
        initial_capital=Decimal("100000"),
        data_sources=DataSourcesConfig(stock="fmp"),
        features=FeaturesConfig(),
        strategy=StrategyConfig(type="factor"),
        backtest_engine="vectorbt",
        train_window=train_window,
        rebalance_frequency=rebalance_frequency,
    )


class RecordingPipeline:
    def __init__(self) -> None:
        self.max_dates: list[date] = []

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        self.max_dates.append(df["date"].max())
        return df.with_columns(pl.lit(1.0).alias("feature_a"))


class FlatStrategy:
    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.select("date", "symbol").with_columns(
            pl.lit(1).alias("signal"),
            pl.lit(1.0).alias("weight"),
        )


class FailingStrategy:
    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        raise AssertionError("generate_signals should not be called")


def test_rebalance_dates_daily_weekly_monthly():
    dates = [date(2024, 1, 29) + timedelta(days=index) for index in range(10)]
    assert _rebalance_dates(dates, "daily") == dates
    assert _rebalance_dates(dates, "weekly") == [date(2024, 1, 29), date(2024, 2, 5)]
    assert _rebalance_dates(dates, "monthly") == [date(2024, 1, 29), date(2024, 2, 1)]
    assert _rebalance_dates(dates, 3) == [
        date(2024, 1, 29),
        date(2024, 2, 1),
        date(2024, 2, 4),
        date(2024, 2, 7),
    ]
    with pytest.raises(ValueError, match="Unsupported rebalance frequency"):
        _rebalance_dates(dates, "3d")


def test_walk_forward_signals_monthly_uses_prior_window_only():
    ohlcv = _ohlcv_frame(date(2024, 1, 15), 60)
    pipeline = RecordingPipeline()
    strategy = FlatStrategy()
    signals = walk_forward_signals(pipeline, strategy, ohlcv, _config())

    assert signals.height == 2
    assert signals["date"].to_list() == [date(2024, 2, 1), date(2024, 3, 1)]
    assert pipeline.max_dates == [date(2024, 1, 31), date(2024, 2, 29)]
    assert all(
        train_date < rebalance_date
        for train_date, rebalance_date in zip(
            pipeline.max_dates,
            signals["date"].to_list(),
            strict=True,
        )
    )


def test_run_backtest_frame_uses_prebuilt_signals_and_forward_fills():
    data = _ohlcv_frame(date(2024, 1, 15), 60).with_columns(pl.lit(1.0).alias("feature_a"))
    prebuilt_signals = pl.DataFrame(
        {
            "date": [date(2024, 2, 1), date(2024, 3, 1)],
            "symbol": ["AAPL", "AAPL"],
            "signal": [1, 1],
            "weight": [1.0, 1.0],
        }
    )

    result = run_backtest_frame(
        "vectorbt",
        FailingStrategy(),
        data,
        _config(),
        prebuilt_signals=prebuilt_signals,
    )

    between = result.returns.filter(
        (pl.col("date") > date(2024, 2, 1)) & (pl.col("date") < date(2024, 3, 1))
    )
    assert result.signals.height == 2
    assert between.filter(pl.col("portfolio_return") > 0).height == between.height


def test_run_backtest_frame_sums_weighted_returns_across_symbols():
    data = pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 1), date(2024, 1, 2)],
            "symbol": ["AAPL", "AAPL", "MSFT", "MSFT"],
            "open": [100.0, 110.0, 100.0, 90.0],
            "high": [100.0, 110.0, 100.0, 90.0],
            "low": [100.0, 110.0, 100.0, 90.0],
            "close": [100.0, 110.0, 100.0, 90.0],
            "volume": [1_000.0, 1_000.0, 1_000.0, 1_000.0],
        }
    )
    prebuilt_signals = pl.DataFrame(
        {
            "date": [date(2024, 1, 2), date(2024, 1, 2)],
            "symbol": ["AAPL", "MSFT"],
            "signal": [1, -1],
            "weight": [0.6, 0.4],
        }
    )

    result = run_backtest_frame(
        "vectorbt",
        FailingStrategy(),
        data,
        _config(),
        prebuilt_signals=prebuilt_signals,
    )

    jan_2_return = result.returns.filter(pl.col("date") == date(2024, 1, 2))["portfolio_return"][0]
    assert jan_2_return == pytest.approx(0.10)


def test_backtest_engines_accept_walk_forward_inputs():
    fixture = _ohlcv_frame(date(2023, 1, 1), 365)
    pipeline = FeaturePipeline([])
    data = pipeline.fit_transform(fixture)
    strategy = FlatStrategy()

    vectorbt = VectorBTProEngine().run(strategy, data, _config(), pipeline=pipeline, ohlcv=fixture)
    zipline = ZiplineEngine().run(strategy, data, _config(), pipeline=pipeline, ohlcv=fixture)

    for result in (vectorbt, zipline):
        assert result.metrics["sharpe"] == result.metrics["sharpe"]
        assert result.metrics["cagr"] == result.metrics["cagr"]
        assert result.metrics["max_drawdown"] == result.metrics["max_drawdown"]
    assert vectorbt.returns.schema == zipline.returns.schema
    assert vectorbt.equity_curve.schema == zipline.equity_curve.schema
    assert vectorbt.signals.schema == zipline.signals.schema
