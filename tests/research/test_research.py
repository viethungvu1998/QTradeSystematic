from __future__ import annotations

import dataclasses
from datetime import date, timedelta
from decimal import Decimal

import polars as pl

from qts.research.backtest.base import (
    BacktestConfig,
    DataSourcesConfig,
    FeaturesConfig,
    StrategyConfig,
    UniverseConfig,
)
from qts.research.backtest.engines.vectorbtpro_engine import (
    VectorBTProEngine,
    _pivot_close,
)
from qts.research.backtest.metrics import cagr, max_drawdown, sharpe_ratio, sortino_ratio, win_rate
from qts.research.backtest.simulation.calendar import CryptoCalendar, NYSECalendar
from qts.research.backtest.simulation.commission import PercentageCommission
from qts.research.backtest.simulation.fills import NextOpenFill
from qts.research.backtest.simulation.slippage import VolatilityScaledSlippage
from qts.research.features.forward_returns import ForwardReturns
from qts.research.features.fundamentals import FundamentalFeatures
from qts.research.features.indicators.momentum import ROCFeature, RSIFeature
from qts.research.features.onchain import OnchainFeatures
from qts.research.features.pipeline import FeaturePipeline
from qts.research.features.preprocessor import EPSILON, preprocess_ohlcv
from qts.research.features.technical import TechnicalFeatures
from qts.research.strategies.factor.rank import FactorStrategy
from qts.research.strategies.stat_arb.mean_reversion import StatArbStrategy


def _config(engine: str = "vectorbt") -> BacktestConfig:
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


def test_feature_pipeline(fundamentals_frame):
    rows = []
    for index in range(300):
        current = date(2023, 1, 1) + timedelta(days=index)
        price = 100.0 + index
        rows.append(
            {
                "date": current,
                "symbol": "AAPL",
                "open": price,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price + 0.5,
                "volume": 1_000.0 + index,
            }
        )
    long_stock_ohlcv = pl.DataFrame(rows)
    pipeline = FeaturePipeline(
        [
            TechnicalFeatures(),
            FundamentalFeatures(fundamentals_frame),
            ForwardReturns([1]),
        ]
    )
    featured = pipeline.fit_transform(long_stock_ohlcv)
    assert "rsi_14" in featured.columns
    assert "pe_ratio" in featured.columns
    assert "forward_return_1" in featured.columns


def test_preprocess_ohlcv_deduplicates_last_row():
    frame = pl.DataFrame(
        [
            {
                "date": date(2024, 1, 1),
                "symbol": "AAPL",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000.0,
            },
            {
                "date": date(2024, 1, 1),
                "symbol": "AAPL",
                "open": 110.0,
                "high": 111.0,
                "low": 109.0,
                "close": 110.5,
                "volume": 1100.0,
            },
        ]
    )
    result = preprocess_ohlcv(frame, min_trading_days=1)
    assert result.height == 1
    assert result["close"][0] == 110.5


def test_preprocess_ohlcv_replaces_non_positive_values():
    frame = pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "symbol": ["AAPL", "AAPL"],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 0.0],
            "volume": [1000.0, 1010.0],
        }
    )
    result = preprocess_ohlcv(frame, min_trading_days=1)
    assert result.filter(pl.col("date") == date(2024, 1, 2))["close"][0] == EPSILON


def test_preprocess_ohlcv_corrects_high_low_band():
    frame = pl.DataFrame(
        {
            "date": [date(2024, 1, 1)],
            "symbol": ["AAPL"],
            "open": [100.0],
            "high": [99.0],
            "low": [98.0],
            "close": [101.0],
            "volume": [1000.0],
        }
    )
    result = preprocess_ohlcv(frame, min_trading_days=1)
    assert result["high"][0] == 101.0
    assert result["low"][0] == 98.0


def test_preprocess_ohlcv_drops_short_history_symbols():
    frame = pl.concat(
        [
            pl.DataFrame(
                {
                    "date": [date(2024, 1, 1)],
                    "symbol": ["AAPL"],
                    "open": [100.0],
                    "high": [101.0],
                    "low": [99.0],
                    "close": [100.5],
                    "volume": [1000.0],
                }
            ),
            pl.DataFrame(
                {
                    "date": [date(2024, 1, 1) + timedelta(days=index) for index in range(10)],
                    "symbol": ["MSFT"] * 10,
                    "open": [100.0 + index for index in range(10)],
                    "high": [101.0 + index for index in range(10)],
                    "low": [99.0 + index for index in range(10)],
                    "close": [100.5 + index for index in range(10)],
                    "volume": [1000.0 + index for index in range(10)],
                }
            ),
        ],
        how="vertical",
    )
    result = preprocess_ohlcv(frame, min_trading_days=252)
    assert "MSFT" not in result["symbol"].to_list()


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


def test_factor_strategy_uses_cross_sectional_zscores():
    frame = pl.DataFrame(
        {
            "date": [date(2024, 1, 1)] * 3,
            "symbol": ["AAA", "BBB", "CCC"],
            "open": [10.0, 10.0, 10.0],
            "high": [11.0, 11.0, 11.0],
            "low": [9.0, 9.0, 9.0],
            "close": [10.0, 10.0, 10.0],
            "volume": [1000.0, 1000.0, 1000.0],
            "col_a": [100.0, 50.0, 0.0],
            "col_b": [0.0, 1.0, 0.9],
        }
    )
    signals = FactorStrategy(long_quantile=2 / 3, short_quantile=1 / 3).generate_signals(frame)
    raw_top_symbol = frame.with_columns(
        ((pl.col("col_a") + pl.col("col_b")) / 2).alias("raw_score")
    ).sort("raw_score", descending=True)["symbol"][0]
    long_symbols = signals.filter(pl.col("signal") == 1)["symbol"].to_list()
    assert raw_top_symbol == "AAA"
    assert long_symbols == ["BBB"]
    assert signals.columns == ["date", "symbol", "signal", "weight"]
    assert signals.filter((pl.col("weight") < 0) | (pl.col("weight") > 1)).is_empty()


def test_factor_strategy_handles_empty_input():
    signals = FactorStrategy().generate_signals(
        pl.DataFrame(
            schema={
                "date": pl.Date,
                "symbol": pl.String,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )
    )
    assert signals.columns == ["date", "symbol", "signal", "weight"]
    assert signals.is_empty()


def test_simulation_models():
    assert NextOpenFill().get_fill_price(Decimal("10"), Decimal("11"), Decimal("12")) == Decimal(
        "12"
    )
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
    assert max_drawdown(equity) >= 0
    assert 0 <= win_rate(returns) <= 1


# ---------------------------------------------------------------------------
# VectorBTProEngine — real vbt integration tests
# ---------------------------------------------------------------------------


def _long_crypto_ohlcv(days: int = 550) -> pl.DataFrame:
    """Multi-symbol crypto fixture long enough to survive preprocess_ohlcv (≥252 bars).

    550 days gives ~10 rebalance windows once train_window=260 is satisfied.
    """
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    rows = []
    for i, sym in enumerate(symbols):
        base = 30_000.0 + i * 1_000
        for day in range(days):
            price = base + day * 0.5 + (day % 7) * 10  # gentle trend + weekly cycle
            rows.append(
                {
                    "date": date(2022, 1, 1) + timedelta(days=day),
                    "symbol": sym,
                    "open": price,
                    "high": price + 50.0,
                    "low": price - 50.0,
                    "close": price + 10.0,
                    "volume": 1_000.0 + day * 5,
                }
            )
    return pl.DataFrame(rows)


def _crypto_config(engine: str = "vectorbt") -> BacktestConfig:
    return BacktestConfig(
        workflow="crypto_test",
        asset_types=["crypto"],
        universe=UniverseConfig(crypto=["BTC/USDT", "ETH/USDT", "SOL/USDT"]),
        start_date=date(2022, 1, 1),
        end_date=date(2023, 7, 5),
        initial_capital=Decimal("100000"),
        data_sources=DataSourcesConfig(crypto="binance"),
        features=FeaturesConfig(),
        strategy=StrategyConfig(type="factor"),
        backtest_engine=engine,
        rebalance_frequency="monthly",
        # must be >= preprocess_ohlcv min_trading_days=252 so walk-forward windows
        # are long enough to survive the preprocessing filter
        train_window=260,
    )


def test_pivot_close_produces_wide_datetime_indexed_frame():
    ohlcv = _long_crypto_ohlcv(10)
    wide = _pivot_close(ohlcv)
    import pandas as pd

    assert isinstance(wide.index, pd.DatetimeIndex)
    assert set(wide.columns) == {"BTC/USDT", "ETH/USDT", "SOL/USDT"}
    assert wide.shape == (10, 3)


def test_vbt_engine_crypto_produces_valid_result():
    """VectorBTProEngine calls real vbt and returns a correctly shaped BacktestResult."""
    ohlcv = _long_crypto_ohlcv(400)
    pipeline = FeaturePipeline([RSIFeature(periods=[14]), ROCFeature(periods=[1, 7])])
    data = pipeline.fit_transform(ohlcv)
    strategy = FactorStrategy()
    config = _crypto_config()

    result = VectorBTProEngine().run(strategy, data, config)

    assert result.engine_name == "vectorbt"
    assert set(result.metrics) == {"sharpe", "sortino", "cagr", "max_drawdown", "win_rate"}
    import math

    assert all(math.isfinite(v) for v in result.metrics.values()), "non-finite in metrics"
    assert result.returns.columns == ["date", "portfolio_return"]
    assert result.equity_curve.columns == ["date", "equity"]
    assert result.equity_curve.height > 0
    # equity starts near initial capital
    first_equity = result.equity_curve["equity"][0]
    assert 80_000 < first_equity < 120_000


def test_vbt_engine_walk_forward_crypto():
    """Walk-forward signals are used by VectorBTProEngine for crypto multi-asset backtest."""
    ohlcv = _long_crypto_ohlcv(400)
    pipeline = FeaturePipeline([ROCFeature(periods=[7, 30])])
    data = pipeline.fit_transform(ohlcv)
    strategy = FactorStrategy()
    config = _crypto_config()

    result = VectorBTProEngine().run(strategy, data, config, pipeline=pipeline, ohlcv=ohlcv)

    assert result.returns.schema == {"date": pl.Date, "portfolio_return": pl.Float64}
    assert result.equity_curve.schema == {"date": pl.Date, "equity": pl.Float64}
    assert result.signals.columns == ["date", "symbol", "signal", "weight"]
    # signals should only contain rebalance dates (monthly → ~12 per year × 1 year)
    assert result.signals.height > 0


def test_vbt_engine_with_commission_and_slippage():
    """Commission and slippage round-trip from BacktestConfig into vbt fees/slippage args."""
    from qts.research.backtest.base import CommissionConfig
    from qts.research.backtest.engines.vectorbtpro_engine import _extract_fees, _extract_slippage

    config_with = dataclasses.replace(
        _crypto_config(),
        commission=CommissionConfig(model="percentage", rate=Decimal("0.002")),
        slippage_model="fixed",
    )
    config_none = _crypto_config()

    assert _extract_fees(config_with) == 0.002
    assert _extract_fees(config_none) == 0.001  # Binance taker default
    assert _extract_slippage(config_with) == 0.0005  # fixed model default
    assert _extract_slippage(config_none) == 0.0
