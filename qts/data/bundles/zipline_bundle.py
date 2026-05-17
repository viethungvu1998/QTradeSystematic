"""Zipline bundle adapter."""

from __future__ import annotations

import os
from contextlib import suppress
from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl

from qts.data.bundles.base import BaseBundleAdapter


def _as_naive_datetime_index(values: pd.Series) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(values))
    if index.tz is not None:
        index = index.tz_localize(None)
    return index


class ZiplineBundleAdapter(BaseBundleAdapter):
    """Stores bundle data on disk as a zipline-reloaded bundle."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _set_zipline_root(self) -> None:
        os.environ["ZIPLINE_ROOT"] = str(self.root)

    def ingest(
        self,
        name: str,
        data: pl.DataFrame,
        start: date,
        end: date,
        *,
        calendar_name: str = "NYSE",
        price_scale_map: dict[str, float] | None = None,
    ) -> Path:
        from zipline.data import bundles as zd_bundles
        from zipline.utils.calendar_utils import get_calendar

        self._set_zipline_root()
        trading_calendar = get_calendar(calendar_name)

        ohlcv = (
            data.select(["date", "symbol", "open", "high", "low", "close", "volume"])
            .with_columns(
                pl.col("date").cast(pl.Date),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
            )
            .sort(["symbol", "date"])
        )
        pdf = ohlcv.to_pandas()
        pdf["date"] = _as_naive_datetime_index(pdf["date"])
        valid_sessions = trading_calendar.sessions_in_range(
            pd.Timestamp(start),
            pd.Timestamp(end),
        )
        valid_dates = {session.date() for session in valid_sessions}
        pdf = pdf.loc[pdf["date"].dt.date.isin(valid_dates)].copy()

        sid_frames: dict[int, pd.DataFrame] = {}
        meta_rows: list[dict[str, object]] = []
        symbols = sorted(ohlcv["symbol"].unique().to_list())
        sid_map = {symbol: index for index, symbol in enumerate(symbols, start=1)}
        scale_map = price_scale_map or {}

        for symbol in symbols:
            symbol_rows = pdf.loc[pdf["symbol"] == symbol].copy()
            if symbol_rows.empty:
                continue
            symbol_rows["date"] = _as_naive_datetime_index(symbol_rows["date"])
            sid = sid_map[symbol]
            bars = (
                symbol_rows.drop(columns=["symbol"])
                .set_index("date")
                .sort_index()[["open", "high", "low", "close", "volume"]]
                .astype("float64")
            )
            scale = float(scale_map.get(symbol, 1.0))
            if scale <= 0:
                raise ValueError(f"price scale must be positive for symbol {symbol!r}")
            if scale != 1.0:
                bars[["open", "high", "low", "close"]] = bars[["open", "high", "low", "close"]] / scale
                bars["volume"] = bars["volume"] * scale
            bars.index = pd.DatetimeIndex(bars.index).tz_localize("UTC")
            sid_frames[sid] = bars

            symbol_start = pd.Timestamp(symbol_rows["date"].min(), tz="UTC")
            symbol_end = pd.Timestamp(symbol_rows["date"].max(), tz="UTC")
            if calendar_name == "24/7":
                auto_close_date = symbol_end + pd.Timedelta(days=1)
            else:
                auto_close_date = symbol_end + pd.offsets.BDay(1)
            meta_rows.append(
                {
                    "sid": sid,
                    "symbol": symbol,
                    "start_date": symbol_start,
                    "end_date": symbol_end,
                    "exchange": calendar_name,
                    "auto_close_date": auto_close_date,
                }
            )

        if not meta_rows:
            raise ValueError(
                "No rows remained after aligning data to the Zipline trading calendar. "
                f"calendar={calendar_name!r}"
            )

        equities_meta = pd.DataFrame(meta_rows).set_index("sid").sort_index()

        def ingest_func(
            environ,
            asset_db_writer,
            minute_bar_writer,
            daily_bar_writer,
            adjustment_writer,
            calendar,
            start_session,
            end_session,
            cache,
            show_progress,
            output_dir,
        ):
            sid_data_iter = ((sid, bars) for sid, bars in sid_frames.items())
            daily_bar_writer.write(
                sid_data_iter,
                assets=equities_meta.index,
                show_progress=False,
            )
            asset_db_writer.write(equities=equities_meta)
            adjustment_writer.write(
                splits=pd.DataFrame(columns=["sid", "effective_date", "ratio"]),
                dividends=pd.DataFrame(
                    columns=[
                        "sid",
                        "ex_date",
                        "declared_date",
                        "record_date",
                        "pay_date",
                        "amount",
                    ]
                ),
            )

        with suppress(Exception):
            zd_bundles.unregister(name)
        zd_bundles.register(
            name,
            ingest_func,
            calendar_name=calendar_name,
            start_session=pd.Timestamp(start),
            end_session=pd.Timestamp(end),
        )
        zd_bundles.ingest(name, environ=os.environ, show_progress=False)
        return Path(os.environ.get("ZIPLINE_ROOT", Path.home() / ".zipline")) / "data" / name

    def load(self, name: str) -> pl.DataFrame:
        from zipline.data import bundles as zd_bundles
        from zipline.data.data_portal import DataPortal
        from zipline.utils.calendar_utils import get_calendar

        self._set_zipline_root()

        bundle = zd_bundles.load(name, environ=os.environ)
        daily_reader = bundle.equity_daily_bar_reader
        trading_calendar = get_calendar("NYSE")
        data_portal = DataPortal(
            asset_finder=bundle.asset_finder,
            trading_calendar=trading_calendar,
            first_trading_day=daily_reader.first_trading_day,
            equity_daily_reader=daily_reader,
            adjustment_reader=bundle.adjustment_reader,
            last_available_session=daily_reader.last_available_dt,
        )

        assets = list(bundle.asset_finder.retrieve_all(bundle.asset_finder.equities_sids))
        rows: list[dict[str, object]] = []

        for session in daily_reader.sessions:
            opens = data_portal.get_spot_value(assets, "open", session, "daily")
            highs = data_portal.get_spot_value(assets, "high", session, "daily")
            lows = data_portal.get_spot_value(assets, "low", session, "daily")
            closes = data_portal.get_spot_value(assets, "close", session, "daily")
            volumes = data_portal.get_spot_value(assets, "volume", session, "daily")
            for asset, open_, high_, low_, close_, volume_ in zip(
                assets,
                opens,
                highs,
                lows,
                closes,
                volumes,
                strict=True,
            ):
                if pd.isna(close_):
                    continue
                rows.append(
                    {
                        "date": session.date(),
                        "symbol": asset.symbol,
                        "open": float(open_),
                        "high": float(high_),
                        "low": float(low_),
                        "close": float(close_),
                        "volume": float(volume_),
                    }
                )

        if not rows:
            return pl.DataFrame(
                schema={
                    "date": pl.Date,
                    "symbol": pl.Utf8,
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Float64,
                }
            )

        return pl.from_dicts(
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
        ).sort(["date", "symbol"])

    def exists(self, name: str) -> bool:
        from zipline.data import bundles as zd_bundles

        self._set_zipline_root()

        try:
            zd_bundles.load(name, environ=os.environ)
            return True
        except Exception:
            return False
