"""DNSE OpenAPI v2 data source adapter (Vietnamese equities, covered warrants, and VN30 futures).

Base URL: https://openapi.dnse.com.vn
Auth:     X-API-Key + X-Aux-Date + X-Signature (HMAC-SHA256)
OHLCV:    GET /price/ohlc  — type=STOCK (equities + warrants), type=DERIVATIVE (futures)
"""

from __future__ import annotations

import hashlib
import hmac
import os
import uuid
from datetime import UTC, date, datetime
from email.utils import formatdate

import polars as pl

from qts.core.errors import DataSourceError
from qts.core.registry import Registry
from qts.data._schemas import OHLCV_COLUMNS, DataType
from qts.data.base import BaseDataSource

_BASE_URL = "https://openapi.dnse.com.vn"
_OHLC_PATH = "/price/ohlc"

# Maps QTS interval strings to DNSE resolution codes.
_RESOLUTION_MAP: dict[str, str] = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "1h",
    "1d": "1D",
    "1w": "1W",
}

_EMPTY_SCHEMA = {
    "date": pl.Date,
    "symbol": pl.Utf8,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}


def _to_dnse_resolution(interval: str) -> str:
    return _RESOLUTION_MAP.get(interval.lower(), "1D")


def _strip_vn_prefix(symbol: str) -> str:
    for prefix in ("VNF:", "VNW:", "VN:"):
        if symbol.startswith(prefix):
            return symbol[len(prefix):]
    return symbol


def _dnse_market_type(symbol: str) -> str:
    """Map QTS symbol prefix to the DNSE 'type' query parameter."""
    return "DERIVATIVE" if symbol.startswith("VNF:") else "STOCK"


def _epoch(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp())


def _sign(api_secret: str, path: str, date_str: str, nonce: str, query: str) -> str:
    # DNSE OpenAPI v2 signature: HMAC-SHA256(secret, path + date + nonce + query)
    message = path + date_str + nonce + query
    return hmac.new(api_secret.encode(), message.encode(), hashlib.sha256).hexdigest()


def _columnar_to_frame(symbol: str, pages: list[dict]) -> pl.DataFrame:
    """Merge DNSE columnar pages {t,o,h,l,c,v} into canonical OHLCV DataFrame."""
    t_all: list = []
    o_all: list[float] = []
    h_all: list[float] = []
    l_all: list[float] = []
    c_all: list[float] = []
    v_all: list[float] = []
    for page in pages:
        t_all.extend(page.get("t") or [])
        o_all.extend(float(x) for x in (page.get("o") or []))
        h_all.extend(float(x) for x in (page.get("h") or []))
        l_all.extend(float(x) for x in (page.get("l") or []))
        c_all.extend(float(x) for x in (page.get("c") or []))
        v_all.extend(float(x) for x in (page.get("v") or []))
    if not t_all:
        return pl.DataFrame(schema=_EMPTY_SCHEMA)
    return pl.DataFrame({
        "date": [datetime.fromtimestamp(ts, tz=UTC).date() for ts in t_all],
        "symbol": [symbol] * len(t_all),
        "open": o_all,
        "high": h_all,
        "low": l_all,
        "close": c_all,
        "volume": v_all,
    })


class _DNSEClient:
    """Thin httpx wrapper for DNSE OpenAPI v2 (market data endpoints only)."""

    def __init__(self, api_key: str, api_secret: str) -> None:
        import httpx  # noqa: PLC0415

        self._http = httpx.Client(base_url=_BASE_URL, timeout=30)
        self._api_key = api_key
        self._api_secret = api_secret

    def _auth_headers(self, path: str, query: str) -> dict[str, str]:
        date_str = formatdate(usegmt=True)
        nonce = str(uuid.uuid4())
        sig = _sign(self._api_secret, path, date_str, nonce, query)
        return {"X-API-Key": self._api_key, "X-Aux-Date": date_str, "X-Signature": sig}

    def get_ohlc_pages(
        self,
        symbol: str,
        market_type: str,
        resolution: str,
        from_ts: int,
        to_ts: int,
    ) -> list[dict]:
        """Fetch all pages of OHLCV data, following nextTime for pagination."""
        pages: list[dict] = []
        current_from = from_ts
        while True:
            params: dict[str, str | int] = {
                "symbol": symbol,
                "type": market_type,
                "resolution": resolution,
                "from": current_from,
                "to": to_ts,
            }
            query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            headers = self._auth_headers(_OHLC_PATH, query)
            resp = self._http.get(_OHLC_PATH, params=params, headers=headers)
            resp.raise_for_status()
            page = resp.json()
            pages.append(page)
            next_time = page.get("nextTime", 0)
            if not next_time:
                break
            current_from = next_time
        return pages


@Registry.register_data_source("dnse")
class DNSEDataSource(BaseDataSource):
    """DNSE OpenAPI v2 adapter for VN equities (VN:), warrants (VNW:), and futures (VNF:).

    Use from_env() for live data; pass ohlcv_payloads for tests.
    Requires DNSE_API_KEY and DNSE_API_SECRET environment variables.
    """

    CAPABILITIES = frozenset({DataType.OHLCV, DataType.FUTURES_OHLCV})

    def __init__(
        self,
        client: _DNSEClient | None = None,
        ohlcv_payloads: dict[str, pl.DataFrame] | None = None,
    ) -> None:
        self._client = client
        self.ohlcv_payloads = ohlcv_payloads or {}

    @classmethod
    def from_env(cls) -> DNSEDataSource:
        """Build from DNSE_API_KEY and DNSE_API_SECRET env vars."""
        return cls(
            client=_DNSEClient(
                api_key=os.environ["DNSE_API_KEY"],
                api_secret=os.environ["DNSE_API_SECRET"],
            )
        )

    async def fetch(self, data_type: DataType, symbol: str, **kwargs) -> pl.DataFrame:
        if data_type not in {DataType.OHLCV, DataType.FUTURES_OHLCV}:
            raise NotImplementedError(f"DNSE does not support {data_type.value}.")
        return await self.get_ohlcv(
            symbol,
            kwargs.get("start"),
            kwargs.get("end"),
            kwargs.get("interval", "1d"),
        )

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
                raise DataSourceError("Unknown DNSE symbol", symbol, (start, end)) from exc
            frame = frame.select(OHLCV_COLUMNS)
            if start is None or end is None:
                return frame
            return frame.filter(pl.col("date").is_between(start, end))

        if start is None or end is None:
            raise DataSourceError("start and end are required for live DNSE OHLCV", symbol)
        try:
            pages = self._client.get_ohlc_pages(
                symbol=_strip_vn_prefix(symbol),
                market_type=_dnse_market_type(symbol),
                resolution=_to_dnse_resolution(interval),
                from_ts=_epoch(start),
                to_ts=_epoch(end),
            )
        except Exception as exc:
            raise DataSourceError(str(exc), symbol, (start, end)) from exc
        return _columnar_to_frame(symbol, pages)

    async def get_fundamentals(self, symbol: str) -> pl.DataFrame:
        raise NotImplementedError("DNSE does not provide fundamental data.")

    async def stream_ticks(self, symbols: list[str]):  # pragma: no cover - unsupported
        raise NotImplementedError("DNSE tick streaming requires the WebSocket datafeed.")
