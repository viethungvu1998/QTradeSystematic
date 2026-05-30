from __future__ import annotations

import importlib.util
from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd
import polars as pl
import pytest

from qts.data.bundles.zipline_bundle import ZiplineBundleAdapter
from qts.research.backtest.base import BacktestConfig, BacktestResult, UniverseConfig
from qts.research.backtest.engines._targets import build_target_schedule, schedule_to_lookup
from qts.research.backtest.engines.vectorbtpro_engine import VectorBTProEngine
from qts.research.backtest.engines.zipline_engine import (
    CRYPTO_CALENDAR,
    ZiplineReloadedEngine,
    _build_price_scale_map,
    _build_zipline_symbol_map,
    _infer_calendar_name,
    build_zipline_preflight_report,
    filter_zipline_compatible_data,
)
from qts.research.strategies.base import BaseStrategy
from qts.research.strategies.stat_arb.mean_reversion import StatArbStrategy

ZIPLINE_AVAILABLE = importlib.util.find_spec("zipline") is not None


@pytest.fixture(autouse=True)
def isolated_qts_root(tmp_path, monkeypatch):
    monkeypatch.setenv("QTS_ROOT", str(tmp_path / "qts-root"))


def _synthetic_ohlcv(
    symbols: list[str], *, freq: str = "B", periods: int = 252, start: str = "2022-01-03"
) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range(start, periods=periods, freq=freq)
    rows: list[dict[str, object]] = []
    for offset, symbol in enumerate(symbols):
        close = 100.0 + offset * 10
        for current in dates:
            close *= 1 + rng.normal(0.0005, 0.01)
            open_ = close * (1 + rng.normal(0.0, 0.002))
            high = max(open_, close) * (1 + abs(rng.normal(0.001, 0.001)))
            low = min(open_, close) * (1 - abs(rng.normal(0.001, 0.001)))
            volume = float(1_000_000 + rng.integers(0, 50_000))
            rows.append(
                {
                    "date": current.date(),
                    "symbol": symbol,
                    "open": float(open_),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close),
                    "volume": volume,
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "date": pl.Date,
            "symbol": pl.Utf8,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        },
    )


class MockStrategy(BaseStrategy):
    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        unique_dates = sorted(data["date"].unique().to_list())
        unique_symbols = sorted(data["symbol"].unique().to_list())
        rows = []
        for current_date in unique_dates:
            for symbol in unique_symbols:
                rows.append(
                    {
                        "date": current_date,
                        "symbol": symbol,
                        "signal": 1,
                        "weight": 1 / 3,
                    }
                )
        return pl.DataFrame(
            rows,
            schema={
                "date": pl.Date,
                "symbol": pl.Utf8,
                "signal": pl.Int32,
                "weight": pl.Float64,
            },
        )


@pytest.fixture
def stock_ohlcv_fixture() -> pl.DataFrame:
    return _synthetic_ohlcv(["AAPL", "MSFT", "GOOGL"])


@pytest.fixture
def crypto_ohlcv_fixture() -> pl.DataFrame:
    return _synthetic_ohlcv(["BTC/USDT"], freq="D", periods=180, start="2022-01-01")


@pytest.fixture
def crypto_pair_ohlcv_fixture() -> pl.DataFrame:
    dates = pd.date_range("2022-01-01", periods=240, freq="D")
    rng = np.random.default_rng(7)
    base = np.cumsum(rng.normal(0.0, 1.0, len(dates))) + 20000.0
    spread = 200.0 * np.sin(np.linspace(0.0, 18.0 * np.pi, len(dates)))
    btc = base + spread
    eth = base * 0.075 + spread * 0.01 + 30.0
    rows: list[dict[str, object]] = []
    for current, btc_close, eth_close in zip(dates, btc, eth, strict=True):
        rows.extend(
            [
                {
                    "date": current.date(),
                    "symbol": "BTC/USDT",
                    "open": float(btc_close * 0.998),
                    "high": float(btc_close * 1.002),
                    "low": float(btc_close * 0.996),
                    "close": float(btc_close),
                    "volume": 10_000.0,
                },
                {
                    "date": current.date(),
                    "symbol": "ETH/USDT",
                    "open": float(eth_close * 0.998),
                    "high": float(eth_close * 1.002),
                    "low": float(eth_close * 0.996),
                    "close": float(eth_close),
                    "volume": 20_000.0,
                },
            ]
        )
    return pl.DataFrame(
        rows,
        schema={
            "date": pl.Date,
            "symbol": pl.Utf8,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        },
    )


@pytest.fixture
def crypto_futures_overflow_fixture() -> pl.DataFrame:
    dates = pd.date_range("2024-01-01", periods=240, freq="D")
    rng = np.random.default_rng(19)

    avax_base = np.cumsum(rng.normal(0.02, 0.45, len(dates))) + 25.0
    avax_spread = 1.8 * np.sin(np.linspace(0.0, 18.0 * np.pi, len(dates)))
    avax = avax_base + avax_spread
    link = avax_base - 0.8 * avax_spread + 4.0

    ada_base = np.cumsum(rng.normal(0.001, 0.01, len(dates))) + 0.75
    ada_spread = 0.08 * np.sin(np.linspace(0.0, 16.0 * np.pi, len(dates)))
    ada = ada_base + ada_spread
    doge = (
        0.22 + 0.12 * (ada - ada.mean()) + 0.03 * np.cos(np.linspace(0.0, 16.0 * np.pi, len(dates)))
    )

    rows: list[dict[str, object]] = []
    for current, avax_close, link_close, ada_close, doge_close in zip(
        dates, avax, link, ada, doge, strict=True
    ):
        rows.extend(
            [
                {
                    "date": current.date(),
                    "symbol": "PERP:AVAX/USDT",
                    "open": float(avax_close * 0.998),
                    "high": float(avax_close * 1.002),
                    "low": float(avax_close * 0.996),
                    "close": float(avax_close),
                    "volume": 12_000_000.0,
                },
                {
                    "date": current.date(),
                    "symbol": "PERP:LINK/USDT",
                    "open": float(link_close * 0.998),
                    "high": float(link_close * 1.002),
                    "low": float(link_close * 0.996),
                    "close": float(link_close),
                    "volume": 18_000_000.0,
                },
                {
                    "date": current.date(),
                    "symbol": "PERP:ADA/USDT",
                    "open": float(ada_close * 0.998),
                    "high": float(ada_close * 1.002),
                    "low": float(ada_close * 0.996),
                    "close": float(ada_close),
                    "volume": 800_000_000.0,
                },
                {
                    "date": current.date(),
                    "symbol": "PERP:DOGE/USDT",
                    "open": float(doge_close * 0.998),
                    "high": float(doge_close * 1.002),
                    "low": float(doge_close * 0.996),
                    "close": float(doge_close),
                    "volume": 5_500_000_000.0,
                },
            ]
        )

    return pl.DataFrame(
        rows,
        schema={
            "date": pl.Date,
            "symbol": pl.Utf8,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        },
    )


@pytest.fixture
def mock_strategy() -> BaseStrategy:
    return MockStrategy()


@pytest.fixture
def research_config() -> BacktestConfig:
    return BacktestConfig(
        workflow="research",
        asset_types=["stock"],
        universe=UniverseConfig(stock=["AAPL", "MSFT", "GOOGL"]),
        start_date=date(2022, 1, 3),
        end_date=date(2022, 12, 30),
        initial_capital=Decimal("100000"),
        backtest_engine="zipline",
        rebalance_frequency="monthly",
    )


def test_infer_calendar_name_for_supported_asset_types():
    assert _infer_calendar_name(["AAPL"], None) == "NYSE"
    assert _infer_calendar_name(["VN:VNM"], None) == "XHOSE"
    assert _infer_calendar_name(["BTC/USDT"], None) == CRYPTO_CALENDAR
    assert _infer_calendar_name(["PERP:BTC/USDT"], None) == "CMES"


def test_infer_calendar_name_requires_explicit_calendar_for_mixed_universe():
    with pytest.raises(ValueError, match="explicit calendar"):
        _infer_calendar_name(["AAPL", "BTC/USDT"], None)

    assert _infer_calendar_name(["AAPL", "BTC/USDT"], "cmes") == "CMES"
    assert _infer_calendar_name(["BTC/USDT"], "crypto") == CRYPTO_CALENDAR
    assert _infer_calendar_name(["BTC/USDT"], "always_open") == CRYPTO_CALENDAR


def test_build_zipline_symbol_map_sanitizes_non_equity_symbols():
    mapping = _build_zipline_symbol_map(["BTC/USDT", "PERP:ETH/USDT", "VN:VNM"])
    assert mapping["BTC/USDT"].startswith("BTC_USDT_")
    assert mapping["PERP:ETH/USDT"].startswith("PERP_ETH_USDT_")
    assert mapping["VN:VNM"].startswith("VN_VNM_")
    assert len(set(mapping.values())) == 3


@pytest.mark.skipif(not ZIPLINE_AVAILABLE, reason="zipline extra not installed")
def test_zipline_result_schema(stock_ohlcv_fixture, mock_strategy, research_config):
    result = ZiplineReloadedEngine().run(mock_strategy, stock_ohlcv_fixture, research_config)

    assert isinstance(result, BacktestResult)
    assert result.returns.schema == {"date": pl.Date, "portfolio_return": pl.Float64}
    assert result.equity_curve.schema == {"date": pl.Date, "equity": pl.Float64}
    assert set(result.metrics) == {"sharpe", "sortino", "cagr", "max_drawdown", "win_rate"}
    assert all(isinstance(value, float) for value in result.metrics.values())
    assert result.engine_name == "zipline"


@pytest.mark.skipif(not ZIPLINE_AVAILABLE, reason="zipline extra not installed")
def test_zipline_accepts_crypto_with_default_always_open_calendar(
    crypto_ohlcv_fixture, mock_strategy
):
    config = BacktestConfig(
        workflow="research",
        asset_types=["crypto"],
        universe=UniverseConfig(crypto=["BTC/USDT"]),
        start_date=date(2022, 1, 1),
        end_date=date(2022, 6, 29),
        initial_capital=Decimal("100000"),
        backtest_engine="zipline",
        rebalance_frequency="monthly",
    )
    result = ZiplineReloadedEngine().run(mock_strategy, crypto_ohlcv_fixture, config)
    assert result.engine_name == "zipline"
    assert result.returns.schema == {"date": pl.Date, "portfolio_return": pl.Float64}


@pytest.mark.skipif(not ZIPLINE_AVAILABLE, reason="zipline extra not installed")
def test_engine_schema_parity(stock_ohlcv_fixture, mock_strategy, research_config):
    r_zip = ZiplineReloadedEngine().run(mock_strategy, stock_ohlcv_fixture, research_config)
    r_vbt = VectorBTProEngine().run(mock_strategy, stock_ohlcv_fixture, research_config)

    assert r_zip.returns.schema == r_vbt.returns.schema
    assert r_zip.equity_curve.schema == r_vbt.equity_curve.schema
    assert set(r_zip.metrics) == set(r_vbt.metrics)


def test_build_target_schedule_holds_missing_days_and_emits_zero_exits():
    sessions = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=5, freq="D"))
    signals = pl.DataFrame(
        {
            "date": [date(2024, 1, 2), date(2024, 1, 4)],
            "symbol": ["BTC/USDT", "BTC/USDT"],
            "signal": [1, 0],
            "weight": [0.4, 0.0],
        }
    )

    schedule = build_target_schedule(
        signals,
        sessions,
        ["BTC/USDT"],
        shift_by_one_bar=True,
    )
    lookup = schedule_to_lookup(schedule.events)

    assert schedule.targets.loc[pd.Timestamp("2024-01-01"), "BTC/USDT"] == 0.0
    assert schedule.targets.loc[pd.Timestamp("2024-01-02"), "BTC/USDT"] == 0.0
    assert schedule.targets.loc[pd.Timestamp("2024-01-03"), "BTC/USDT"] == 0.4
    assert schedule.targets.loc[pd.Timestamp("2024-01-04"), "BTC/USDT"] == 0.4
    assert schedule.targets.loc[pd.Timestamp("2024-01-05"), "BTC/USDT"] == 0.0
    assert lookup == {
        date(2024, 1, 3): {"BTC/USDT": 0.4},
        date(2024, 1, 5): {"BTC/USDT": 0.0},
    }


def test_build_price_scale_map_scales_fractional_crypto_targets():
    base_fixture = _synthetic_ohlcv(
        ["BTC/USDT", "ETH/USDT"], freq="D", periods=12, start="2024-01-01"
    )
    scale_expr = (
        pl.when(pl.col("symbol") == "BTC/USDT")
        .then(pl.lit(1000.0))
        .when(pl.col("symbol") == "ETH/USDT")
        .then(pl.lit(30.0))
        .otherwise(pl.lit(1.0))
    )
    fixture = base_fixture.with_columns(
        (pl.col("open") * scale_expr).alias("open"),
        (pl.col("high") * scale_expr).alias("high"),
        (pl.col("low") * scale_expr).alias("low"),
        (pl.col("close") * scale_expr).alias("close"),
    )
    sessions = pd.DatetimeIndex(pd.to_datetime(sorted(fixture["date"].unique().to_list())))
    signals = pl.DataFrame(
        {
            "date": [date(2024, 1, 5), date(2024, 1, 5), date(2024, 1, 8), date(2024, 1, 8)],
            "symbol": ["BTC/USDT", "ETH/USDT", "BTC/USDT", "ETH/USDT"],
            "signal": [1, -1, 0, 0],
            "weight": [0.0001, 0.9999, 0.0, 0.0],
        }
    )
    schedule = build_target_schedule(
        signals, sessions, ["BTC/USDT", "ETH/USDT"], shift_by_one_bar=True
    )
    config = BacktestConfig(
        workflow="research",
        asset_types=["crypto"],
        universe=UniverseConfig(crypto=["BTC/USDT", "ETH/USDT"]),
        initial_capital=Decimal("100000"),
    )

    scale_map = _build_price_scale_map(fixture, schedule, config)

    assert scale_map["BTC/USDT"] > 1.0
    assert scale_map["ETH/USDT"] > 1.0


def test_build_price_scale_map_scales_fractional_crypto_futures_targets():
    base_fixture = _synthetic_ohlcv(
        ["PERP:BTC/USDT", "PERP:ETH/USDT"],
        freq="D",
        periods=12,
        start="2024-01-01",
    )
    scale_expr = (
        pl.when(pl.col("symbol") == "PERP:BTC/USDT").then(pl.lit(1500.0)).otherwise(pl.lit(50.0))
    )
    fixture = base_fixture.with_columns(
        (pl.col("open") * scale_expr).alias("open"),
        (pl.col("high") * scale_expr).alias("high"),
        (pl.col("low") * scale_expr).alias("low"),
        (pl.col("close") * scale_expr).alias("close"),
    )
    sessions = pd.DatetimeIndex(pd.to_datetime(sorted(fixture["date"].unique().to_list())))
    signals = pl.DataFrame(
        {
            "date": [date(2024, 1, 5), date(2024, 1, 5), date(2024, 1, 8), date(2024, 1, 8)],
            "symbol": ["PERP:BTC/USDT", "PERP:ETH/USDT", "PERP:BTC/USDT", "PERP:ETH/USDT"],
            "signal": [1, -1, 0, 0],
            "weight": [0.0001, 0.9999, 0.0, 0.0],
        }
    )
    schedule = build_target_schedule(
        signals,
        sessions,
        ["PERP:BTC/USDT", "PERP:ETH/USDT"],
        shift_by_one_bar=True,
    )
    config = BacktestConfig(
        workflow="research",
        asset_types=["crypto_futures"],
        universe=UniverseConfig(crypto_futures=["PERP:BTC/USDT", "PERP:ETH/USDT"]),
        initial_capital=Decimal("100000"),
    )

    scale_map = _build_price_scale_map(fixture, schedule, config)

    assert scale_map["PERP:BTC/USDT"] > 1.0
    assert scale_map["PERP:ETH/USDT"] > 1.0


@pytest.mark.skipif(not ZIPLINE_AVAILABLE, reason="zipline extra not installed")
def test_zipline_preflight_report_flags_overflowing_futures_leg(crypto_futures_overflow_fixture):
    strategy = StatArbStrategy(
        entry_zscore=1.0,
        exit_zscore=0.0,
        zscore_window=20,
        pairs=[["PERP:ADA/USDT", "PERP:DOGE/USDT"]],
    )
    config = BacktestConfig(
        workflow="research",
        asset_types=["crypto_futures"],
        universe=UniverseConfig(crypto_futures=["PERP:ADA/USDT", "PERP:DOGE/USDT"]),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 8, 27),
        initial_capital=Decimal("100000"),
        backtest_engine="zipline",
        rebalance_frequency="daily",
        calendar="crypto",
    )
    signals = strategy.generate_signals(crypto_futures_overflow_fixture)

    report = build_zipline_preflight_report(
        crypto_futures_overflow_fixture,
        signals,
        config,
        shift_by_one_bar=True,
        calendar_name=CRYPTO_CALENDAR,
    )

    doge_row = report.filter(pl.col("symbol") == "PERP:DOGE/USDT").row(0, named=True)
    avax_row = report.filter(pl.col("symbol") == "PERP:AVAX/USDT").row(0, named=True)

    assert doge_row["valid_for_zipline"] is False
    assert doge_row["volume_overflow_rows"] > 0
    assert "volume_overflow" in doge_row["failure_reason"]
    assert avax_row["valid_for_zipline"] is True


@pytest.mark.skipif(not ZIPLINE_AVAILABLE, reason="zipline extra not installed")
def test_zipline_filters_invalid_futures_leg_before_backtest(crypto_futures_overflow_fixture):
    symbols = [
        "PERP:AVAX/USDT",
        "PERP:LINK/USDT",
        "PERP:ADA/USDT",
        "PERP:DOGE/USDT",
    ]
    config = BacktestConfig(
        workflow="research",
        asset_types=["crypto_futures"],
        universe=UniverseConfig(crypto_futures=symbols),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 8, 27),
        initial_capital=Decimal("100000"),
        backtest_engine="zipline",
        rebalance_frequency="daily",
        calendar="crypto",
    )
    strategy = StatArbStrategy(
        entry_zscore=1.0,
        exit_zscore=0.0,
        zscore_window=20,
        pairs=[["PERP:AVAX/USDT", "PERP:LINK/USDT"], ["PERP:ADA/USDT", "PERP:DOGE/USDT"]],
    )
    provisional_signals = strategy.generate_signals(crypto_futures_overflow_fixture)
    filtered_data, _, report = filter_zipline_compatible_data(
        crypto_futures_overflow_fixture,
        provisional_signals,
        config,
        shift_by_one_bar=True,
        calendar_name=CRYPTO_CALENDAR,
    )

    zipline_engine = ZiplineReloadedEngine()
    zipline = zipline_engine.run(strategy, crypto_futures_overflow_fixture, config)
    surviving_symbols = sorted(zipline.signals["symbol"].unique().to_list())
    vectorbt = VectorBTProEngine().run(
        strategy,
        filtered_data.filter(pl.col("symbol").is_in(surviving_symbols)),
        config,
    )

    assert "PERP:DOGE/USDT" not in set(zipline.signals["symbol"].unique().to_list())
    assert "PERP:ADA/USDT" not in set(zipline.signals["symbol"].unique().to_list())
    assert set(surviving_symbols) == {"PERP:AVAX/USDT", "PERP:LINK/USDT"}

    doge_row = zipline_engine.last_preflight_report.filter(
        pl.col("symbol") == "PERP:DOGE/USDT"
    ).row(0, named=True)
    assert doge_row["valid_for_zipline"] is False
    assert doge_row["volume_overflow_rows"] > 0
    assert report.filter(pl.col("valid_for_zipline"))["symbol"].to_list() == [
        "PERP:ADA/USDT",
        "PERP:AVAX/USDT",
        "PERP:LINK/USDT",
    ]
    assert vectorbt.signals.sort(["date", "symbol"]).equals(
        zipline.signals.sort(["date", "symbol"])
    )
    assert vectorbt.returns.height == zipline.returns.height


@pytest.mark.skipif(not ZIPLINE_AVAILABLE, reason="zipline extra not installed")
def test_zipline_crypto_bundle_retains_weekends(tmp_path, crypto_ohlcv_fixture):
    import os

    from zipline.data import bundles as zd_bundles

    adapter = ZiplineBundleAdapter(root=tmp_path / "zipline-root")
    bundle_name = "crypto_weekends"
    adapter.ingest(
        bundle_name,
        crypto_ohlcv_fixture.with_columns(pl.lit("BTC_USDT").alias("symbol")),
        start=date(2022, 1, 1),
        end=date(2022, 6, 29),
        calendar_name=CRYPTO_CALENDAR,
    )
    bundle = zd_bundles.load(bundle_name, environ=os.environ)
    sessions = [timestamp.date() for timestamp in bundle.equity_daily_bar_reader.sessions]
    expected = sorted(crypto_ohlcv_fixture["date"].unique().to_list())
    assert sessions == expected


@pytest.mark.skipif(not ZIPLINE_AVAILABLE, reason="zipline extra not installed")
def test_crypto_stat_arb_parity_across_engines(crypto_pair_ohlcv_fixture):
    config = BacktestConfig(
        workflow="research",
        asset_types=["crypto"],
        universe=UniverseConfig(crypto=["BTC/USDT", "ETH/USDT"]),
        start_date=date(2022, 1, 1),
        end_date=date(2022, 8, 28),
        initial_capital=Decimal("100000"),
        backtest_engine="vectorbt",
        rebalance_frequency="daily",
    )
    strategy = StatArbStrategy(
        entry_zscore=1.0,
        exit_zscore=0.0,
        zscore_window=20,
        pairs=[["BTC/USDT", "ETH/USDT"]],
    )

    vectorbt = VectorBTProEngine().run(strategy, crypto_pair_ohlcv_fixture, config)
    zipline = ZiplineReloadedEngine().run(strategy, crypto_pair_ohlcv_fixture, config)

    assert vectorbt.signals.sort(["date", "symbol"]).equals(
        zipline.signals.sort(["date", "symbol"])
    )
    assert vectorbt.returns.height == zipline.returns.height
    assert vectorbt.returns["date"].to_list() == zipline.returns["date"].to_list()
    assert vectorbt.metrics["sharpe"] * zipline.metrics["sharpe"] >= 0
    assert vectorbt.metrics["sortino"] * zipline.metrics["sortino"] >= 0
    assert abs(vectorbt.metrics["cagr"] - zipline.metrics["cagr"]) < 0.02
    assert abs(vectorbt.metrics["max_drawdown"] - zipline.metrics["max_drawdown"]) < 0.02
    assert abs(vectorbt.metrics["win_rate"] - zipline.metrics["win_rate"]) < 0.1
    final_vectorbt = float(vectorbt.equity_curve["equity"].to_list()[-1])
    final_zipline = float(zipline.equity_curve["equity"].to_list()[-1])
    assert (
        abs(final_vectorbt - final_zipline) / max(abs(final_vectorbt), abs(final_zipline), 1.0)
        < 0.02
    )
