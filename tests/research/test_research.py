from __future__ import annotations

from datetime import date
from decimal import Decimal

import polars as pl

from qts.research.backtest.base import BacktestConfig, DataSourcesConfig, FeaturesConfig, StrategyConfig, UniverseConfig
from qts.research.backtest.engines.vectorbtpro_engine import VectorBTProEngine
from qts.research.backtest.engines.zipline_engine import ZiplineEngine
from qts.research.backtest.metrics import cagr, max_drawdown, sharpe_ratio, sortino_ratio, win_rate
from qts.research.backtest.simulation.calendar import CryptoCalendar, NYSECalendar
from qts.research.backtest.simulation.commission import PercentageCommission
from qts.research.backtest.simulation.fills import NextOpenFill
from qts.research.backtest.simulation.slippage import VolatilityScaledSlippage
from qts.research.features.forward_returns import ForwardReturns
from qts.research.features.fundamentals import FundamentalFeatures
from qts.research.features.onchain import OnchainFeatures
from qts.research.features.pipeline import FeaturePipeline
from qts.research.features.technical import TechnicalFeatures
from qts.research.strategies.factor.model import FactorStrategy
from qts.research.strategies.stat_arb.model import StatArbStrategy


def _config(engine: str = "fast") -> BacktestConfig:
    return BacktestConfig(
        workflow="research",
        asset_types=["stock"],
        universe=UniverseConfig(stock=["AAPL"]),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 3, 20),
        initial_capital=Decimal("100000"),
        data_sources=DataSourcesConfig(stock="fmp"),
        features=FeaturesConfig(),
        strategy=StrategyConfig(type="factor"),
        backtest_engine=engine,
    )


def test_feature_pipeline(stock_ohlcv, fundamentals_frame):
    pipeline = FeaturePipeline(
        [
            TechnicalFeatures(),
            FundamentalFeatures(fundamentals_frame),
            ForwardReturns([1]),
        ]
    )
    featured = pipeline.fit_transform(stock_ohlcv)
    assert "rsi_14" in featured.columns
    assert "pe_ratio" in featured.columns
    assert "forward_return_1" in featured.columns


def test_onchain_noop_for_stock(stock_ohlcv, onchain_frame):
    featured = OnchainFeatures(onchain_frame).fit_transform(stock_ohlcv)
    assert featured.columns == stock_ohlcv.columns


def test_forward_returns_last_row_null(stock_ohlcv):
    featured = ForwardReturns([1, 5]).fit_transform(stock_ohlcv)
    assert featured.tail(1)["forward_return_1"][0] is None
    assert featured.tail(5)["forward_return_5"].null_count() == 5


def test_factor_and_stat_arb_strategies(stock_ohlcv, paired_ohlcv):
    factor_data = TechnicalFeatures().fit_transform(stock_ohlcv)
    factor_signals = FactorStrategy().generate_signals(factor_data)
    stat_signals = StatArbStrategy().generate_signals(paired_ohlcv)
    assert factor_signals.filter(pl.col("signal") != 0).height > 0
    assert stat_signals.columns == ["date", "symbol", "signal", "weight"]


def test_simulation_models():
    assert NextOpenFill().get_fill_price(Decimal("10"), Decimal("11"), Decimal("12")) == Decimal("12")
    high = VolatilityScaledSlippage().apply(Decimal("100"), Decimal("10"))
    low = VolatilityScaledSlippage().apply(Decimal("100"), Decimal("1"))
    assert high > low
    assert PercentageCommission(Decimal("0.001")).calculate(Decimal("1000")) == Decimal("1.000")
    assert not NYSECalendar().is_session(date(2024, 1, 1))
    assert CryptoCalendar().is_session(date(2024, 1, 6))


def test_metrics():
    returns = [0.01, -0.005, 0.02, 0.0]
    equity = [100.0, 101.0, 100.5, 102.5]
    assert sharpe_ratio(returns) == sharpe_ratio(returns)
    assert sortino_ratio(returns) == sortino_ratio(returns)
    assert cagr(equity) > -1
    assert max_drawdown(equity) <= 0
    assert 0 <= win_rate(returns) <= 1


def test_backtest_engines_share_schema(stock_ohlcv):
    data = FeaturePipeline([TechnicalFeatures()]).fit_transform(stock_ohlcv)
    strategy = FactorStrategy()
    fast = VectorBTProEngine().run(strategy, data, _config("fast"))
    normal = ZiplineEngine().run(strategy, data, _config("normal"))
    assert set(fast.metrics) == set(normal.metrics)
    assert fast.signals.columns == normal.signals.columns
