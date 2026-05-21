"""Bar-by-bar backtest engine."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pandas as pd
import polars as pl

from qts.core.instrument import AssetType
from qts.core.registry import Registry
from qts.research.backtest._runner import run_backtest_frame, walk_forward_signals
from qts.research.backtest.base import (
    BacktestConfig,
    BacktestResult,
    BaseEngine,
    empty_backtest_result,
    empty_portfolio_snapshots_frame,
    empty_trade_log_frame,
)
from qts.research.backtest.engines._targets import build_target_schedule, schedule_to_lookup
from qts.research.backtest.metrics import build_metrics
from qts.research.backtest.zipline_observability import zipline_observability
from qts.research.strategies.base import BaseStrategy
from qts.research.strategies.stat_arb.base import BaseStatArbStrategy


@Registry.register_engine("normal")
class ZiplineEngine(BaseEngine):
    """Legacy sequential engine kept for compatibility with older tests."""

    def run(
        self,
        strategy: BaseStrategy,
        data: pl.DataFrame,
        config: BacktestConfig,
        *,
        pipeline=None,
        ohlcv: pl.DataFrame | None = None,
    ) -> BacktestResult:
        prebuilt = (
            walk_forward_signals(pipeline, strategy, ohlcv, config)
            if pipeline is not None and ohlcv is not None
            else None
        )
        return run_backtest_frame("zipline", strategy, data, config, prebuilt_signals=prebuilt)


CRYPTO_CALENDAR = "24/7"
_CRYPTO_CALENDAR_ALIASES = {"24/7", "24x7", "always_open", "crypto"}
_MIN_SYNTHETIC_UNITS = 100.0
_MAX_PRICE_SCALE = 1_000_000.0
_ZIPLINE_UINT32_MAX = int(np.iinfo(np.uint32).max)
_ZIPLINE_OHLC_COLUMNS = ("open", "high", "low", "close")


def _zipline_symbol_name(symbol: str) -> str:
    base = "".join(character if character.isalnum() else "_" for character in symbol.upper()).strip("_")
    if not base or not base[0].isalpha():
        base = f"ASSET_{base}"
    digest = hashlib.md5(symbol.encode(), usedforsecurity=False).hexdigest()[:8].upper()
    return f"{base[:32]}_{digest}"


def _build_zipline_symbol_map(symbols: list[str]) -> dict[str, str]:
    return {symbol: _zipline_symbol_name(symbol) for symbol in sorted(symbols)}


def _normalize_calendar_name(name: str) -> str:
    normalized = name.strip()
    if normalized.lower() in _CRYPTO_CALENDAR_ALIASES:
        return CRYPTO_CALENDAR
    return normalized.upper()


def _infer_calendar_name(symbols: list[str], explicit_calendar: str | None) -> str:
    if explicit_calendar:
        return _normalize_calendar_name(explicit_calendar)

    asset_types = {AssetType.from_symbol(symbol) for symbol in symbols}
    if len(asset_types) != 1:
        raise ValueError(
            "ZiplineReloadedEngine requires an explicit calendar for mixed asset-type universes."
        )

    asset_type = next(iter(asset_types))
    return {
        AssetType.STOCK: "NYSE",
        AssetType.VN_STOCK: "XHOSE",
        AssetType.CRYPTO: CRYPTO_CALENDAR,
        AssetType.CRYPTO_FUTURES: "CMES",
        AssetType.COMMODITY: "CMES",
    }[asset_type]


def _extract_slippage_bps(config: BacktestConfig) -> float:
    if config.slippage_model is None:
        return 0.0
    basis_points_map = {
        "fixed": 5.0,
        "volatility_scaled": 10.0,
    }
    return float(basis_points_map.get(config.slippage_model, 0.0))


def _pivot_wide(df: pl.DataFrame, value_col: str) -> pd.DataFrame:
    wide = (
        df.select(["date", "symbol", value_col])
        .to_pandas()
        .pivot(index="date", columns="symbol", values=value_col)
        .rename_axis(index=None, columns=None)
        .sort_index()
        .sort_index(axis=1)
    )
    wide.index = pd.to_datetime(wide.index)
    return wide


def _build_price_scale_map(
    data: pl.DataFrame,
    target_schedule,
    config: BacktestConfig,
) -> dict[str, float]:
    open_wide = _pivot_wide(data, "open").reindex(target_schedule.targets.index, columns=target_schedule.targets.columns)
    target_deltas = target_schedule.targets.diff().fillna(target_schedule.targets).abs()
    capital = float(config.initial_capital or 100000)
    scale_map: dict[str, float] = {}

    for symbol in target_schedule.targets.columns:
        if AssetType.from_symbol(symbol) not in {AssetType.CRYPTO, AssetType.CRYPTO_FUTURES}:
            scale_map[symbol] = 1.0
            continue

        prices = open_wide[symbol].replace(0.0, pd.NA).ffill().bfill()
        units = (target_deltas[symbol] * capital / prices).replace([math.inf, -math.inf], pd.NA).dropna()
        units = units[units > 0]
        if units.empty:
            scale_map[symbol] = 1.0
            continue

        min_units = float(units.min())
        scale = math.ceil(_MIN_SYNTHETIC_UNITS / min_units)
        scale_map[symbol] = float(min(max(scale, 1), _MAX_PRICE_SCALE))

    return scale_map


def _calendar_valid_dates(calendar_name: str, start, end) -> set[object]:
    from zipline.utils.calendar_utils import get_calendar

    trading_calendar = get_calendar(calendar_name)
    valid_sessions = trading_calendar.sessions_in_range(
        pd.Timestamp(start),
        pd.Timestamp(end),
    )
    return {session.date() for session in valid_sessions}


def _apply_price_scale(symbol_rows: pd.DataFrame, scale: float) -> pd.DataFrame:
    adjusted = symbol_rows.copy()
    if scale != 1.0:
        adjusted[list(_ZIPLINE_OHLC_COLUMNS)] = adjusted[list(_ZIPLINE_OHLC_COLUMNS)] / scale
        adjusted["volume"] = adjusted["volume"] * scale
    return adjusted


def build_zipline_preflight_report(
    data: pl.DataFrame,
    signals: pl.DataFrame,
    config: BacktestConfig,
    *,
    shift_by_one_bar: bool = False,
    calendar_name: str | None = None,
) -> pl.DataFrame:
    if data.is_empty():
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                "row_count": pl.Int64,
                "aligned_row_count": pl.Int64,
                "calendar_loss_ratio": pl.Float64,
                "price_scale": pl.Float64,
                "max_scaled_price": pl.Float64,
                "max_scaled_volume": pl.Float64,
                "price_overflow_rows": pl.Int64,
                "volume_overflow_rows": pl.Int64,
                "valid_for_zipline": pl.Boolean,
                "failure_reason": pl.String,
            }
        )

    symbols = sorted(data["symbol"].unique().to_list())
    resolved_calendar = calendar_name or _infer_calendar_name(symbols, config.calendar)
    sessions = pd.DatetimeIndex(pd.to_datetime(sorted(data["date"].unique().to_list())))
    target_schedule = build_target_schedule(
        signals,
        sessions,
        symbols,
        shift_by_one_bar=shift_by_one_bar,
    )
    scale_map = _build_price_scale_map(data, target_schedule, config)
    valid_dates = _calendar_valid_dates(
        resolved_calendar,
        config.start_date or data["date"].min(),
        config.end_date or data["date"].max(),
    )

    pdf = (
        data.select(["date", "symbol", "open", "high", "low", "close", "volume"])
        .sort(["symbol", "date"])
        .to_pandas()
    )
    pdf["date"] = pd.DatetimeIndex(pd.to_datetime(pdf["date"])).tz_localize(None)
    rows: list[dict[str, object]] = []

    for symbol in symbols:
        symbol_rows = pdf.loc[pdf["symbol"] == symbol].copy()
        aligned_rows = symbol_rows.loc[symbol_rows["date"].dt.date.isin(valid_dates)].copy()
        scale = float(scale_map.get(symbol, 1.0))
        adjusted_rows = _apply_price_scale(aligned_rows, scale)
        price_overflow_mask = (
            adjusted_rows[list(_ZIPLINE_OHLC_COLUMNS)] * 1000.0 > _ZIPLINE_UINT32_MAX
        ).any(axis=1)
        volume_overflow_mask = adjusted_rows["volume"] > _ZIPLINE_UINT32_MAX
        failures: list[str] = []
        if aligned_rows.empty:
            failures.append("calendar_alignment")
        if int(price_overflow_mask.sum()) > 0:
            failures.append("price_overflow")
        if int(volume_overflow_mask.sum()) > 0:
            failures.append("volume_overflow")

        row_count = int(len(symbol_rows))
        aligned_row_count = int(len(aligned_rows))
        calendar_loss_ratio = 0.0 if row_count == 0 else float((row_count - aligned_row_count) / row_count)
        rows.append(
            {
                "symbol": symbol,
                "row_count": row_count,
                "aligned_row_count": aligned_row_count,
                "calendar_loss_ratio": calendar_loss_ratio,
                "price_scale": scale,
                "max_scaled_price": float(adjusted_rows[list(_ZIPLINE_OHLC_COLUMNS)].max().max())
                if not adjusted_rows.empty
                else 0.0,
                "max_scaled_volume": float(adjusted_rows["volume"].max()) if not adjusted_rows.empty else 0.0,
                "price_overflow_rows": int(price_overflow_mask.sum()),
                "volume_overflow_rows": int(volume_overflow_mask.sum()),
                "valid_for_zipline": not failures,
                "failure_reason": ",".join(failures),
            }
        )

    return pl.DataFrame(rows).sort("symbol")


def filter_zipline_compatible_data(
    data: pl.DataFrame,
    signals: pl.DataFrame,
    config: BacktestConfig,
    *,
    shift_by_one_bar: bool = False,
    calendar_name: str | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    report = build_zipline_preflight_report(
        data,
        signals,
        config,
        shift_by_one_bar=shift_by_one_bar,
        calendar_name=calendar_name,
    )
    if report.is_empty():
        return data, signals, report

    valid_symbols = report.filter(pl.col("valid_for_zipline"))["symbol"].to_list()
    filtered_data = data.filter(pl.col("symbol").is_in(valid_symbols))
    filtered_signals = signals.filter(pl.col("symbol").is_in(valid_symbols))
    return filtered_data, filtered_signals, report


def _shift_dates_to_previous_session(
    index: pd.Index,
    trading_calendar,
) -> list[object]:
    shifted_dates: list[object] = []
    for timestamp in index:
        session = pd.Timestamp(timestamp)
        if session.tz is not None:
            session = session.tz_localize(None)
        session = session.normalize()
        shifted_dates.append(trading_calendar.previous_session(session).date())
    return shifted_dates


def _perf_to_result(
    perf,
    signals: pl.DataFrame,
    *,
    trading_calendar=None,
    shift_to_previous_session: bool = False,
    symbol_map: dict[str, str] | None = None,
) -> BacktestResult:
    returns_series = perf["returns"].fillna(0.0).astype(float)
    equity_series = perf["portfolio_value"].astype(float)
    if shift_to_previous_session:
        if trading_calendar is None:
            raise ValueError("trading_calendar is required when shifting dates to the previous session")
        dates = _shift_dates_to_previous_session(perf.index, trading_calendar)
    else:
        dates = [timestamp.date() for timestamp in perf.index]

    returns_df = pl.DataFrame({"date": dates, "portfolio_return": returns_series.to_list()})
    returns_df = returns_df.with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("portfolio_return").cast(pl.Float64),
    )
    equity_df = pl.DataFrame({"date": dates, "equity": equity_series.to_list()})
    equity_df = equity_df.with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("equity").cast(pl.Float64),
    )

    returns_list = returns_series.to_list()
    equity_list = equity_series.to_list()
    metrics = build_metrics(returns_list, equity_list)
    try:
        trade_log, portfolio_snapshots = zipline_observability(perf, symbol_map)
    except Exception:
        trade_log = empty_trade_log_frame()
        portfolio_snapshots = empty_portfolio_snapshots_frame()

    return BacktestResult(
        engine_name="zipline",
        metrics=metrics,
        returns=returns_df,
        equity_curve=equity_df,
        signals=signals,
        trade_log=trade_log,
        portfolio_snapshots=portfolio_snapshots,
    )


@Registry.register_engine("zipline")
class ZiplineReloadedEngine(BaseEngine):
    """Calendar-aware multi-asset backtest engine backed by zipline-reloaded."""

    def __init__(self) -> None:
        self.last_preflight_report = pl.DataFrame()

    def run(
        self,
        strategy: BaseStrategy,
        data: pl.DataFrame,
        config: BacktestConfig,
        *,
        pipeline=None,
        ohlcv: pl.DataFrame | None = None,
    ) -> BacktestResult:
        from zipline import run_algorithm
        from zipline.api import (
            date_rules,
            get_datetime,
            order_target_percent,
            schedule_function,
            set_commission,
            set_slippage,
            time_rules,
        )
        from zipline.api import (
            symbol as zl_symbol,
        )
        from zipline.finance import commission as zlc
        from zipline.finance import slippage as zls
        from zipline.utils.calendar_utils import get_calendar

        from qts.data.bundles.zipline_bundle import ZiplineBundleAdapter
        from qts.utils.paths import bundle_dir

        symbols = sorted(data["symbol"].unique().to_list())
        calendar_name = _infer_calendar_name(symbols, config.calendar)
        current_data = data
        current_ohlcv = ohlcv
        symbol_map = _build_zipline_symbol_map(symbols)
        initial_report: pl.DataFrame | None = None

        shift_to_next_session = isinstance(strategy, BaseStatArbStrategy)
        while True:
            if pipeline is not None and current_ohlcv is not None:
                signals = walk_forward_signals(pipeline, strategy, current_ohlcv, config)
            else:
                signals = strategy.generate_signals(current_data).sort(["symbol", "date"])

            filtered_data, filtered_signals, report = filter_zipline_compatible_data(
                current_data,
                signals,
                config,
                shift_by_one_bar=shift_to_next_session,
                calendar_name=calendar_name,
            )
            if initial_report is None:
                initial_report = report
            filtered_symbols = sorted(filtered_data["symbol"].unique().to_list()) if not filtered_data.is_empty() else []
            current_symbols = sorted(current_data["symbol"].unique().to_list())
            if not filtered_symbols:
                self.last_preflight_report = initial_report if initial_report is not None else report
                return empty_backtest_result(engine_name="zipline", signals=filtered_signals)
            if filtered_symbols == current_symbols:
                self.last_preflight_report = initial_report if initial_report is not None else report
                data = filtered_data
                signals = filtered_signals
                symbols = filtered_symbols
                symbol_map = _build_zipline_symbol_map(symbols)
                break

            current_data = filtered_data
            if current_ohlcv is not None:
                current_ohlcv = current_ohlcv.filter(pl.col("symbol").is_in(filtered_symbols))

        zipline_data = data.with_columns(pl.col("symbol").replace(symbol_map))

        bundle_name = "qts_" + hashlib.md5(
            json.dumps({"calendar": calendar_name, "symbols": symbols}).encode(),
            usedforsecurity=False,
        ).hexdigest()[:8]
        adapter = ZiplineBundleAdapter(root=bundle_dir())
        start = config.start_date or data["date"].min()
        end = config.end_date or data["date"].max()

        sessions = pd.DatetimeIndex(pd.to_datetime(sorted(data["date"].unique().to_list())))
        target_schedule = build_target_schedule(
            signals,
            sessions,
            symbols,
            shift_by_one_bar=shift_to_next_session,
        )
        price_scale_map = {
            symbol_map[symbol]: scale
            for symbol, scale in _build_price_scale_map(data, target_schedule, config).items()
        }
        adapter.ingest(
            bundle_name,
            zipline_data,
            start,
            end,
            calendar_name=calendar_name,
            price_scale_map=price_scale_map,
        )
        signals_lookup = schedule_to_lookup(target_schedule.events)

        def rebalance(context, data_portal):
            today = get_datetime().date()
            targets = signals_lookup.get(today, {})
            for symbol, weight in targets.items():
                try:
                    order_target_percent(zl_symbol(symbol_map[symbol]), weight)
                except Exception:
                    pass

        def initialize(context):
            rate = float(config.commission.rate) if config.commission else 0.001
            slippage_bps = _extract_slippage_bps(config)
            set_commission(us_equities=zlc.PerDollar(cost=rate))
            set_slippage(us_equities=zls.FixedBasisPointsSlippage(basis_points=slippage_bps, volume_limit=1.0))
            rule = {
                "daily": date_rules.every_day(),
                "weekly": date_rules.week_start(),
                "monthly": date_rules.month_start(),
            }[config.rebalance_frequency]
            schedule_function(rebalance, rule, time_rules.market_open())

        def handle_data(context, data_portal):
            return None

        trading_calendar = get_calendar(calendar_name)
        perf = run_algorithm(
            start=pd.Timestamp(start),
            end=pd.Timestamp(end),
            initialize=initialize,
            handle_data=handle_data,
            capital_base=float(config.initial_capital or 100000),
            bundle=bundle_name,
            trading_calendar=trading_calendar,
        )
        return _perf_to_result(
            perf,
            signals,
            trading_calendar=trading_calendar,
            shift_to_previous_session=shift_to_next_session and calendar_name == CRYPTO_CALENDAR,
            symbol_map=symbol_map,
        )
