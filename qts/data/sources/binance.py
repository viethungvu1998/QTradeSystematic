"""Binance data source adapter."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime
from decimal import Decimal

import polars as pl

from qts.core.errors import DataSourceError
from qts.core.events import Tick
from qts.core.instrument import AssetType, Instrument
from qts.core.registry import Registry
from qts.data._schemas import OHLCV_COLUMNS
from qts.data.base import BaseDataSource

_DEMO_BASE_URL = "https://testnet.binance.vision"


def _to_binance_symbol(symbol: str) -> str:
    return symbol.replace("/", "")


def _date_to_ms(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp() * 1000)


def _klines_to_frame(symbol: str, rows: list) -> pl.DataFrame:
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
    dates = [datetime.fromtimestamp(r[0] / 1000, tz=UTC).date() for r in rows]
    return pl.DataFrame({
        "date": dates,
        "symbol": [symbol] * len(rows),
        "open": [float(r[1]) for r in rows],
        "high": [float(r[2]) for r in rows],
        "low": [float(r[3]) for r in rows],
        "close": [float(r[4]) for r in rows],
        "volume": [float(r[5]) for r in rows],
    })


@Registry.register_data_source("binance")
class BinanceDataSource(BaseDataSource):
    """Binance REST adapter. Use from_env() for real credentials; pass ohlcv_payloads for tests."""

    def __init__(
        self,
        client=None,
        ohlcv_payloads: dict[str, pl.DataFrame] | None = None,
        tick_payloads: dict[str, list[dict[str, str]]] | None = None,
    ) -> None:
        self._client = client
        self.ohlcv_payloads = ohlcv_payloads or {}
        self.tick_payloads = tick_payloads or {}

    @classmethod
    def from_env(cls, mode: str = "demo") -> BinanceDataSource:
        """Build from env vars. mode='demo' uses testnet; mode='live' uses production."""
        from binance.spot import Spot  # noqa: PLC0415

        if mode == "live":
            client = Spot(
                api_key=os.environ["BINANCE_TRADING_KEY"],
                api_secret=os.environ["BINANCE_TRADING_SECRET_KEY"],
            )
        else:
            client = Spot(
                api_key=os.environ["BINANCE_DEMO_TRADING_API_KEY"],
                api_secret=os.environ["BINANCE_DEMO_TRADING_SECRET_KEY"],
                base_url=_DEMO_BASE_URL,
            )
        return cls(client=client)

    async def get_ohlcv(
        self,
        symbol: str,
        start: date | None,
        end: date | None,
        interval: str,
    ) -> pl.DataFrame:
        if self._client is None:
            try:
                frame = self.ohlcv_payloads[symbol]
            except KeyError as exc:
                raise DataSourceError("Unknown Binance symbol", symbol, (start, end)) from exc
            frame = frame.select(OHLCV_COLUMNS)
            if start is None or end is None:
                return frame
            return frame.filter(pl.col("date").is_between(start, end))

        binance_sym = _to_binance_symbol(symbol)
        kwargs: dict = {"limit": 1000}
        if start:
            kwargs["startTime"] = _date_to_ms(start)
        if end:
            kwargs["endTime"] = _date_to_ms(end)
        try:
            rows = self._client.klines(binance_sym, interval, **kwargs)
        except Exception as exc:
            raise DataSourceError(str(exc), symbol, (start, end)) from exc
        return _klines_to_frame(symbol, rows)

    async def get_fundamentals(self, symbol: str) -> pl.DataFrame:
        raise NotImplementedError("Binance fundamentals are not supported.")

    async def stream_ticks(self, symbols: list[str]) -> AsyncIterator[Tick]:
        for symbol in symbols:
            for payload in self.tick_payloads.get(symbol, []):
                yield Tick(
                    instrument=Instrument(
                        symbol=symbol,
                        asset_type=AssetType.CRYPTO,
                        exchange="BINANCE",
                        currency=symbol.split("/")[-1],
                    ),
                    price=Decimal(payload["price"]),
                    volume=Decimal(payload.get("volume", "0")),
                    timestamp=datetime.fromisoformat(payload.get("timestamp"))
                    if payload.get("timestamp")
                    else datetime.now(UTC),
                )
