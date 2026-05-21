"""DNSE OpenAPI v2 data source adapter (Vietnamese equities, covered warrants, and VN30 futures).

Base URL: https://openapi.dnse.com.vn
Auth:     X-API-Key + X-Aux-Date + X-Signature (HMAC-SHA256)
OHLCV:    GET /price/ohlc  — type=STOCK (equities + warrants), type=DERIVATIVE (futures)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import urllib.parse
import uuid
import warnings
from datetime import UTC, date, datetime

import polars as pl

from qts.core.errors import DataSourceError, DataSourceWarning
from qts.core.registry import Registry
from qts.data._schemas import OHLCV_COLUMNS, DataType
from qts.data.base import BaseDataSource
from qts.data.vn_symbols import (
    is_vn_warrant_code,
    strip_vn_prefix,
    to_dnse_futures_alias,
    to_vn_futures_symbol,
)

_BASE_URL = "https://openapi.dnse.com.vn"
_OHLC_PATH = "/price/ohlc"
_DEFAULT_API_VERSION = "2026-05-07"

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


def _to_dnse_symbol(symbol: str) -> str:
    """Normalize QTS symbols to the DNSE symbols accepted by /price/ohlc.

    DNSE market-data currently accepts the rolling VN30 futures aliases such as
    ``VN30F1M`` rather than specific contract codes like ``VN30F2503`` or the
    newer KRX-style codes. We map specific VN30 contract requests to the front-
    month alias so the data source can still return a live derivative series.
    """
    raw = strip_vn_prefix(symbol)
    if not symbol.startswith("VNF:"):
        return raw
    return to_dnse_futures_alias(raw)


def _dnse_market_type(symbol: str) -> str:
    """Map QTS symbol prefix to the DNSE 'type' query parameter."""
    return "DERIVATIVE" if symbol.startswith("VNF:") else "STOCK"


def _is_warrant_lookup(symbol: str) -> bool:
    if not symbol.startswith("VNW:"):
        return False
    return not is_vn_warrant_code(strip_vn_prefix(symbol).upper())


def _looks_like_warrant(item: dict) -> bool:
    symbol = str(item.get("symbol", "")).upper()
    if not is_vn_warrant_code(symbol):
        return False
    name = f"{item.get('name', '')} {item.get('shortName', '')}".lower()
    return "chứng quyền" in name or "chung quyen" in name


def _listed_on_or_before(item: dict, as_of: date) -> bool:
    listed = item.get("listedDate")
    if not listed:
        return True
    try:
        listed_date = date.fromisoformat(str(listed))
    except ValueError:
        return True
    return listed_date <= as_of


def _epoch(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp())


def _rfc2822_now() -> str:
    return datetime.now(UTC).strftime("%a, %d %b %Y %H:%M:%S %z")


def _build_signature_header(
    api_key: str,
    api_secret: str,
    method: str,
    path: str,
    date_str: str,
    *,
    date_header: str = "x-aux-date",
    nonce: str | None = None,
) -> str:
    """Build the HTTP Signature-style X-Signature header DNSE expects."""
    header_key = date_header.lower()
    signed_headers = f"(request-target) {header_key}"
    parts = [
        f"(request-target): {method.lower()} {path}",
        f"{header_key}: {date_str}",
    ]
    if nonce:
        parts.append(f"nonce: {nonce}")
    message = "\n".join(parts)
    raw_sig = hmac.new(api_secret.encode(), message.encode(), hashlib.sha256).digest()
    signature = urllib.parse.quote(base64.b64encode(raw_sig).decode(), safe="")
    header = (
        f'Signature keyId="{api_key}",algorithm="hmac-sha256",'
        f'headers="{signed_headers}",signature="{signature}"'
    )
    if nonce:
        header += f',nonce="{nonce}"'
    return header


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


def _warn_if_truncated_futures_history(
    symbol: str,
    requested_start: date | None,
    frame: pl.DataFrame,
) -> None:
    if requested_start is None or not symbol.startswith("VNF:") or frame.height == 0:
        return
    first_date = frame["date"].min()
    if first_date is None or first_date <= requested_start:
        return
    warnings.warn(
        (
            f"DNSE futures history for {symbol} starts on {first_date}, "
            f"later than requested start {requested_start}."
        ),
        DataSourceWarning,
        stacklevel=2,
    )


class _DNSEClient:
    """Thin httpx wrapper for DNSE OpenAPI v2 (market data endpoints only)."""

    def __init__(self, api_key: str, api_secret: str) -> None:
        import httpx  # noqa: PLC0415

        self._http = httpx.Client(base_url=_BASE_URL, timeout=30)
        self._api_key = api_key
        self._api_secret = api_secret
        self._warrant_index: list[dict] | None = None

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        date_str = _rfc2822_now()
        nonce = uuid.uuid4().hex
        sig = _build_signature_header(
            self._api_key,
            self._api_secret,
            method,
            path,
            date_str,
            date_header="x-aux-date",
            nonce=nonce,
        )
        return {
            "X-API-Key": self._api_key,
            "X-Aux-Date": date_str,
            "X-Signature": sig,
            "version": os.environ.get("DNSE_API_VERSION", _DEFAULT_API_VERSION),
        }

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
            headers = self._auth_headers("GET", _OHLC_PATH)
            resp = self._http.get(_OHLC_PATH, params=params, headers=headers)
            if resp.status_code == 400:
                try:
                    payload = resp.json()
                except Exception:  # pragma: no cover - defensive
                    payload = {}
                message = str(payload.get("message", "")).lower()
                if "invalid symbol" in message:
                    raise ValueError("invalid symbol")
            resp.raise_for_status()
            page = resp.json()
            pages.append(page)
            next_time = page.get("nextTime", 0)
            if not next_time:
                break
            current_from = next_time
        return pages

    def _load_warrant_index(self) -> list[dict]:
        if self._warrant_index is not None:
            return self._warrant_index

        warrants: list[dict] = []
        page = 1
        while True:
            resp = self._http.get(
                "/instruments",
                params={"marketId": "STO", "securityGroupId": "ST", "limit": 100, "page": page},
                headers=self._auth_headers("GET", "/instruments"),
            )
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data", [])
            warrants.extend(item for item in data if _looks_like_warrant(item))
            page_size = int(payload.get("pageSize", 100) or 100)
            total = int(payload.get("total", len(data)) or len(data))
            if page * page_size >= total:
                break
            page += 1

        self._warrant_index = warrants
        return warrants

    def resolve_warrant_symbols(self, underlying_symbol: str, *, as_of: date | None = None) -> list[str]:
        as_of = as_of or datetime.now(UTC).date()
        prefix = f"C{underlying_symbol.upper()}"
        matches = [
            item["symbol"]
            for item in self._load_warrant_index()
            if str(item.get("symbol", "")).upper().startswith(prefix)
            and _listed_on_or_before(item, as_of)
        ]
        return sorted(dict.fromkeys(matches))


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

    def expand_symbols(self, data_type: DataType, symbol: str, **kwargs) -> list[str]:
        if not _is_warrant_lookup(symbol):
            return [to_vn_futures_symbol(symbol) if symbol.startswith("VNF:") else symbol]

        if self._client is None:
            underlying = strip_vn_prefix(symbol).upper()
            prefix = f"VNW:C{underlying}"
            matches = sorted(key for key in self.ohlcv_payloads if key.startswith(prefix))
            return matches or [symbol]

        end = kwargs.get("end")
        if end is None:
            return [symbol]
        underlying = strip_vn_prefix(symbol).upper()
        warrants = self._client.resolve_warrant_symbols(underlying, as_of=end)
        if not warrants:
            return [symbol]
        return [f"VNW:{warrant}" for warrant in warrants]

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
            frame = self._fixture_frame(symbol)
            _warn_if_truncated_futures_history(symbol, start, frame)
            if start is None or end is None:
                return frame
            return frame.filter(pl.col("date").is_between(start, end))

        if start is None or end is None:
            raise DataSourceError("start and end are required for live DNSE OHLCV", symbol)
        if _is_warrant_lookup(symbol):
            return self._get_warrant_basket_ohlcv(symbol, start, end, interval)

        output_symbol = to_vn_futures_symbol(symbol) if symbol.startswith("VNF:") else symbol
        try:
            pages = self._client.get_ohlc_pages(
                symbol=_to_dnse_symbol(symbol),
                market_type=_dnse_market_type(symbol),
                resolution=_to_dnse_resolution(interval),
                from_ts=_epoch(start),
                to_ts=_epoch(end),
            )
        except ValueError as exc:
            if symbol.startswith("VNW:") and str(exc) == "invalid symbol":
                return pl.DataFrame(schema=_EMPTY_SCHEMA)
            raise DataSourceError(str(exc), symbol, (start, end)) from exc
        except Exception as exc:
            raise DataSourceError(str(exc), symbol, (start, end)) from exc
        frame = _columnar_to_frame(output_symbol, pages)
        _warn_if_truncated_futures_history(output_symbol, start, frame)
        return frame

    def _fixture_frame(self, symbol: str) -> pl.DataFrame:
        candidate_symbols = [symbol]
        if symbol.startswith("VNF:"):
            candidate_symbols.append(to_vn_futures_symbol(symbol))
        if _is_warrant_lookup(symbol):
            frame = self._fixture_warrant_basket(symbol)
            if frame is None:
                raise DataSourceError("Unknown DNSE symbol", symbol)
            return frame
        for candidate in dict.fromkeys(candidate_symbols):
            frame = self.ohlcv_payloads.get(candidate)
            if frame is None:
                continue
            frame = frame.select(OHLCV_COLUMNS)
            if symbol.startswith("VNF:"):
                frame = frame.with_columns(pl.lit(to_vn_futures_symbol(symbol)).alias("symbol"))
            return frame
        raise DataSourceError("Unknown DNSE symbol", symbol)

    def _fixture_warrant_basket(self, symbol: str) -> pl.DataFrame | None:
        underlying = strip_vn_prefix(symbol).upper()
        prefix = f"VNW:C{underlying}"
        matches = [
            frame.select(OHLCV_COLUMNS)
            for key, frame in self.ohlcv_payloads.items()
            if key.startswith(prefix)
        ]
        if not matches:
            return None
        return pl.concat(matches, how="vertical").sort(["date", "symbol"])

    def _get_warrant_basket_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str,
    ) -> pl.DataFrame:
        assert self._client is not None
        underlying = strip_vn_prefix(symbol).upper()
        warrants = self._client.resolve_warrant_symbols(underlying, as_of=end)
        if not warrants:
            return pl.DataFrame(schema=_EMPTY_SCHEMA)

        frames: list[pl.DataFrame] = []
        for warrant_symbol in warrants:
            try:
                pages = self._client.get_ohlc_pages(
                    symbol=warrant_symbol,
                    market_type="STOCK",
                    resolution=_to_dnse_resolution(interval),
                    from_ts=_epoch(start),
                    to_ts=_epoch(end),
                )
            except ValueError as exc:
                if str(exc) == "invalid symbol":
                    continue
                raise DataSourceError(str(exc), symbol, (start, end)) from exc
            except Exception as exc:
                raise DataSourceError(str(exc), symbol, (start, end)) from exc
            frame = _columnar_to_frame(f"VNW:{warrant_symbol}", pages)
            if frame.height > 0:
                frames.append(frame)

        if not frames:
            return pl.DataFrame(schema=_EMPTY_SCHEMA)
        return pl.concat(frames, how="vertical").sort(["date", "symbol"])

    async def get_fundamentals(self, symbol: str) -> pl.DataFrame:
        raise NotImplementedError("DNSE does not provide fundamental data.")

    async def stream_ticks(self, symbols: list[str]):  # pragma: no cover - unsupported
        raise NotImplementedError("DNSE tick streaming requires the WebSocket datafeed.")
