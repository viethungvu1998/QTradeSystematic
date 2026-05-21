"""Binance data source adapters (spot and USDT-margined perpetual futures)."""

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
from qts.data._schemas import DataType, OHLCV_COLUMNS
from qts.data.base import BaseDataSource

_SPOT_DEMO_BASE_URL = "https://testnet.binance.vision"
_FUTURES_BASE_URL = "https://fapi.binance.com"
_FUTURES_DEMO_BASE_URL = "https://testnet.binancefuture.com"


class _FapiClient:
    """Minimal httpx wrapper for the Binance USDT-margined futures REST API (fapi/v1).

    Presents the same .klines() interface as binance.spot.Spot so _fetch_klines
    can be used unchanged for both spot and futures data.
    """

    def __init__(self, base_url: str = _FUTURES_BASE_URL) -> None:
        import httpx  # noqa: PLC0415

        self._http = httpx.Client(base_url=base_url, timeout=30)

    def klines(self, symbol: str, interval: str, **kwargs) -> list:
        params = {"symbol": symbol, "interval": interval, **kwargs}
        resp = self._http.get("/fapi/v1/klines", params=params)
        resp.raise_for_status()
        return resp.json()


def _to_binance_symbol(symbol: str) -> str:
    return symbol.replace("/", "")


def _to_binance_futures_symbol(symbol: str) -> str:
    return symbol.replace("PERP:", "").replace("/", "")


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


def _fetch_klines(client, binance_sym: str, interval: str, start_ms: int | None, end_ms: int | None) -> list:
    """Paginate through Binance klines (1 000-row limit per request)."""
    all_rows: list = []
    next_start_ms = start_ms
    while True:
        kwargs: dict = {"limit": 1000}
        if next_start_ms is not None:
            kwargs["startTime"] = next_start_ms
        if end_ms is not None:
            kwargs["endTime"] = end_ms
        batch = client.klines(binance_sym, interval, **kwargs)
        if not batch:
            break
        all_rows.extend(batch)
        if len(batch) < 1000:
            break
        next_start_ms = batch[-1][0] + 1
    return all_rows


@Registry.register_data_source("binance")
class BinanceDataSource(BaseDataSource):
    """Binance spot REST adapter. Use from_env() for real credentials; pass ohlcv_payloads for tests."""

    CAPABILITIES = frozenset({DataType.OHLCV})

    def __init__(
        self,
        client=None,
        ohlcv_payloads: dict[str, pl.DataFrame] | None = None,
        tick_payloads: dict[str, list[dict[str, str]]] | None = None,
    ) -> None:
        self._client = client
        self.ohlcv_payloads = ohlcv_payloads or {}
        self.tick_payloads = tick_payloads or {}

    async def fetch(self, data_type: DataType, symbol: str, **kwargs) -> pl.DataFrame:
        if data_type is not DataType.OHLCV:
            raise NotImplementedError(f"Binance does not support {data_type.value}.")
        return await self.get_ohlcv(
            symbol,
            kwargs.get("start"),
            kwargs.get("end"),
            kwargs.get("interval", "1d"),
        )

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
                base_url=_SPOT_DEMO_BASE_URL,
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
        try:
            rows = _fetch_klines(
                self._client,
                binance_sym,
                interval,
                _date_to_ms(start) if start else None,
                _date_to_ms(end) if end else None,
            )
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


@Registry.register_data_source("binance_futures")
class BinanceFuturesDataSource(BaseDataSource):
    """Binance USDT-margined perpetual futures REST adapter (fapi). Use from_env() for credentials."""

    CAPABILITIES = frozenset({DataType.FUTURES_OHLCV})

    def __init__(
        self,
        client=None,
        ohlcv_payloads: dict[str, pl.DataFrame] | None = None,
        tick_payloads: dict[str, list[dict[str, str]]] | None = None,
    ) -> None:
        self._client = client
        self.ohlcv_payloads = ohlcv_payloads or {}
        self.tick_payloads = tick_payloads or {}

    async def fetch(self, data_type: DataType, symbol: str, **kwargs) -> pl.DataFrame:
        if data_type is not DataType.FUTURES_OHLCV:
            raise NotImplementedError(f"BinanceFutures does not support {data_type.value}.")
        return await self.get_ohlcv(
            symbol,
            kwargs.get("start"),
            kwargs.get("end"),
            kwargs.get("interval", "1d"),
        )

    @classmethod
    def from_env(cls, mode: str = "demo") -> BinanceFuturesDataSource:
        """Build from env vars. mode='demo' uses testnet; mode='live' uses production."""
        base_url = _FUTURES_BASE_URL if mode == "live" else _FUTURES_DEMO_BASE_URL
        return cls(client=_FapiClient(base_url=base_url))

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
                raise DataSourceError("Unknown Binance futures symbol", symbol, (start, end)) from exc
            frame = frame.select(OHLCV_COLUMNS)
            if start is None or end is None:
                return frame
            return frame.filter(pl.col("date").is_between(start, end))

        binance_sym = _to_binance_futures_symbol(symbol)
        try:
            rows = _fetch_klines(
                self._client,
                binance_sym,
                interval,
                _date_to_ms(start) if start else None,
                _date_to_ms(end) if end else None,
            )
        except Exception as exc:
            raise DataSourceError(str(exc), symbol, (start, end)) from exc
        return _klines_to_frame(symbol, rows)

    async def get_fundamentals(self, symbol: str) -> pl.DataFrame:
        raise NotImplementedError("Binance futures fundamentals are not supported.")

    async def stream_ticks(self, symbols: list[str]) -> AsyncIterator[Tick]:
        for symbol in symbols:
            for payload in self.tick_payloads.get(symbol, []):
                yield Tick(
                    instrument=Instrument(
                        symbol=symbol,
                        asset_type=AssetType.CRYPTO_FUTURES,
                        exchange="BINANCE",
                        currency=symbol.split("/")[-1],
                    ),
                    price=Decimal(payload["price"]),
                    volume=Decimal(payload.get("volume", "0")),
                    timestamp=datetime.fromisoformat(payload.get("timestamp"))
                    if payload.get("timestamp")
                    else datetime.now(UTC),
                )
