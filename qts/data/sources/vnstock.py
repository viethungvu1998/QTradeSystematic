"""KBS (KB Securities) data source adapters for Vietnamese equity, warrants, and VN30 futures.

Calls the KB Securities IIS REST API directly — no third-party vnstock library required.
Base URL: https://kbbuddywts.kbsec.com.vn/iis-server/investment
Auth:     none (public endpoints, browser user-agent)

Fundamentals are stored in a tidy long-format cache:
  ~/.qts/cache/vn_fundamentals/{ticker}.parquet
  schema: symbol | report_type | period | fiscal_year | quarter | report_date | item_en | value
  report_type: KQKD (income) | CDKT (balance sheet) | LCTT (cash flow) | CSTC (ratios)
  Values from KQKD/CDKT/LCTT are in thousands of VND (unit=1000 API param).
  Values from CSTC are native ratio units (%, x, VND/share).
  Cache TTL: 24 hours. Delete the file to force a refresh.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from qts.core.errors import DataSourceError
from qts.core.registry import Registry
from qts.data._schemas import OHLCV_COLUMNS, DataType
from qts.data.base import BaseDataSource
from qts.utils.paths import cache_dir

_KBS_BASE = "https://kbbuddywts.kbsec.com.vn/iis-server/investment"
_KBS_FINANCE = f"{_KBS_BASE}/stock/finance-info"

# Maps QTS interval strings to KBS endpoint suffixes.
_INTERVAL_SUFFIX: dict[str, str] = {
    "1m": "1P",
    "5m": "5P",
    "15m": "15P",
    "30m": "30P",
    "1h": "60P",
    "1d": "day",
    "1w": "week",
    "1mo": "month",
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

_EMPTY_OHLCV_SCHEMA = {
    "date": pl.Date,
    "symbol": pl.Utf8,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}

# Tidy long-format schema for all financial statements.
_FUNDAMENTALS_SCHEMA: dict[str, type[pl.DataType]] = {
    "symbol": pl.Utf8,
    "report_type": pl.Utf8,
    "period": pl.Utf8,
    "fiscal_year": pl.Int32,
    "quarter": pl.Int32,   # NULL for annual periods
    "report_date": pl.Date,
    "item_en": pl.Utf8,
    "value": pl.Float64,
}

# KBS financial report types.
_REPORT_TYPES = ("KQKD", "CDKT", "LCTT", "CSTC")

# Cache TTL in hours before a re-fetch is triggered.
_FUNDAMENTALS_CACHE_TTL_HOURS = 24


def _to_kbs_interval(interval: str) -> str:
    return _INTERVAL_SUFFIX.get(interval.lower(), "day")


def _strip_vn_prefix(symbol: str) -> str:
    for prefix in ("VNF:", "VNW:", "VN:"):
        if symbol.startswith(prefix):
            return symbol[len(prefix):]
    return symbol


_INDEX_SYMBOLS = frozenset({
    "VNINDEX", "HNXINDEX", "UPCOMINDEX", "VN30", "VN100", "HNX30",
})

_INTRADAY_SUFFIXES = frozenset({"1P", "5P", "15P", "30P", "60P"})


def _price_scale(symbol: str) -> float:
    """Prices in thousands of VND for stocks/warrants; full points for derivatives/indices."""
    if symbol.startswith("VNF:"):
        return 1.0
    if _strip_vn_prefix(symbol) in _INDEX_SYMBOLS:
        return 1.0
    return 1000.0


_FUTURES_UNDERLYING = {"VN30": "I1", "VN100": "I2", "GB05": "B5", "GB10": "BA"}
_FUTURES_ALPHA = "ABCDEFGHJKLMNPQRSTVW"


def _to_krx_futures(raw: str) -> str:
    """Convert VN30F-style symbol to KRX format if expiry is May 2025 or later.

    KBS adopted the KRX 9-char code (e.g. 41I1G6000) from May 2025.
    Already-converted symbols starting with '41' are returned unchanged.
    """
    import re  # noqa: PLC0415

    if raw.startswith("41"):
        return raw
    m = re.match(r"^(VN30|VN100|GB05|GB10)F(\d{2})(\d{2})$", raw.upper())
    if not m:
        return raw
    underlying, yy, mm = m.group(1), int(m.group(2)), int(m.group(3))
    year = 2000 + yy
    if year < 2025 or (year == 2025 and mm < 5):
        return raw
    u_code = _FUTURES_UNDERLYING[underlying]
    idx = (year - 2010) % 30
    y_code = str(idx) if idx <= 9 else _FUTURES_ALPHA[idx - 10]
    m_code = str(mm) if mm <= 9 else chr(ord("A") + mm - 10)
    return f"41{u_code}{y_code}{m_code}000"


def _rows_to_ohlcv(symbol: str, rows: list[dict], scale: float) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=_EMPTY_OHLCV_SCHEMA)
    dates: list[date] = []
    opens, highs, lows, closes, volumes = [], [], [], [], []
    for r in rows:
        raw_t = r["t"]
        if isinstance(raw_t, str):
            dt = date.fromisoformat(raw_t[:10])
        else:
            dt = date.fromtimestamp(raw_t)
        dates.append(dt)
        opens.append(float(r["o"]) / scale)
        highs.append(float(r["h"]) / scale)
        lows.append(float(r["l"]) / scale)
        closes.append(float(r["c"]) / scale)
        volumes.append(float(r["v"]))
    return pl.DataFrame({
        "date": dates,
        "symbol": [symbol] * len(rows),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


def _parse_financial_page(
    symbol: str, report_type: str, response: dict
) -> pl.DataFrame:
    """Convert one KBS finance-info API page (wide-pivot) to tidy long-format rows.

    KBS returns Head[{ID, YearPeriod, TermName, ReportDate}] and
    Content[section][{NameEn, Value1..Value4}]. This unpivots Value1..N
    using the Head period metadata, yielding one row per (symbol, period, item).
    """
    head = response.get("Head") or []
    content = response.get("Content") or {}
    if not head or not content:
        return pl.DataFrame(schema=_FUNDAMENTALS_SCHEMA)

    # Build {head_id → period metadata} from Head array.
    period_meta: dict[int, dict] = {}
    for h in head:
        idx = int(h["ID"])
        year = int(h.get("YearPeriod") or 0)
        term_name = str(h.get("TermName") or "")
        raw_date = str(h.get("ReportDate") or "")
        quarter: int | None = None
        if "Quý" in term_name:
            try:
                quarter = int(term_name.replace("Quý", "").strip())
            except ValueError:
                pass
        period = f"{year}-Q{quarter}" if quarter is not None else str(year)
        report_date = date.fromisoformat(raw_date[:10]) if len(raw_date) >= 10 else None
        period_meta[idx] = {
            "period": period,
            "fiscal_year": year,
            "quarter": quarter,
            "report_date": report_date,
        }

    # Iterate all sections and items, unpivoting Value1..ValueN.
    record_rows: list[dict] = []
    for section_rows in content.values():
        for item in section_rows:
            item_en = (item.get("NameEn") or item.get("Name") or "").strip()
            if not item_en:
                continue
            for i, meta in period_meta.items():
                raw_val = item.get(f"Value{i}")
                if raw_val is None:
                    continue
                try:
                    value = float(raw_val)
                except (TypeError, ValueError):
                    continue
                record_rows.append({
                    "symbol": symbol,
                    "report_type": report_type,
                    "period": meta["period"],
                    "fiscal_year": meta["fiscal_year"],
                    "quarter": meta["quarter"],
                    "report_date": meta["report_date"],
                    "item_en": item_en,
                    "value": value,
                })

    if not record_rows:
        return pl.DataFrame(schema=_FUNDAMENTALS_SCHEMA)
    return pl.DataFrame(record_rows, schema=_FUNDAMENTALS_SCHEMA)


def _fundamentals_cache_path(ticker: str, termtype: int) -> Path:
    label = "annual" if termtype == 1 else "quarterly"
    return cache_dir() / "vn_fundamentals" / f"{ticker}_{label}.parquet"


class _KBSClient:
    """Thin httpx wrapper for the KB Securities IIS REST API."""

    def __init__(self) -> None:
        import httpx  # noqa: PLC0415

        self._http = httpx.Client(timeout=30, headers=_HEADERS)

    # ------------------------------------------------------------------ OHLCV

    def get_ohlcv(
        self, symbol: str, start: date, end: date, interval: str, is_index: bool
    ) -> list[dict]:
        suffix = _to_kbs_interval(interval)
        base = "index" if is_index else "stocks"
        url = f"{_KBS_BASE}/{base}/{symbol}/data_{suffix}"
        # KBS intraday endpoints exclude edate; add one day so the target day is included.
        api_end = (end + timedelta(days=1)) if suffix in _INTRADAY_SUFFIXES else end
        params = {
            "sdate": start.strftime("%d-%m-%Y"),
            "edate": api_end.strftime("%d-%m-%Y"),
        }
        resp = self._http.get(url, params=params)
        resp.raise_for_status()
        return resp.json().get(f"data_{suffix}") or []

    # -------------------------------------------------------------- Financials

    def get_financial_report(
        self,
        symbol: str,
        report_type: str,
        termtype: int,
        page: int = 1,
        page_size: int = 4,
    ) -> dict:
        """Fetch one page of a KBS financial report.

        Args:
            symbol:      Raw ticker (no VN: prefix).
            report_type: KQKD | CDKT | LCTT | CSTC.
            termtype:    1 = annual, 2 = quarterly.
            page:        Page number (4 periods per page).
            page_size:   Periods per page (default 4).

        Returns:
            Raw API dict with keys Head, Content, Audit, Unit.
        """
        params: dict = {
            "type": report_type,
            "termtype": termtype,
            "page": page,
            "pageSize": page_size,
            "unit": 1000,
        }
        if report_type == "LCTT":
            # Cash flow endpoint requires camelCase termType and explicit code param.
            params["code"] = symbol
            params["termType"] = termtype
        else:
            params["languageid"] = 1
        resp = self._http.get(f"{_KBS_FINANCE}/{symbol}", params=params)
        resp.raise_for_status()
        return resp.json()

    def get_all_financials(
        self,
        symbol: str,
        termtype: int = 1,
        pages: int = 3,
    ) -> pl.DataFrame:
        """Fetch all 4 report types, paginated, and return a unified long-format DataFrame.

        Args:
            symbol:   Raw ticker (no VN: prefix).
            termtype: 1 = annual (default), 2 = quarterly.
            pages:    Pages per report type (4 periods/page → pages*4 periods max).
                      Default 3 → up to 12 annual years or 12 quarters.
        """
        frames: list[pl.DataFrame] = []
        for report_type in _REPORT_TYPES:
            for page in range(1, pages + 1):
                try:
                    raw = self.get_financial_report(symbol, report_type, termtype, page)
                    df = _parse_financial_page(symbol, report_type, raw)
                    if df.is_empty():
                        break
                    frames.append(df)
                except Exception:
                    break
                if page < pages:
                    time.sleep(0.3)
            time.sleep(0.3)

        if not frames:
            return pl.DataFrame(schema=_FUNDAMENTALS_SCHEMA)
        return pl.concat(frames)


def _is_index(symbol: str) -> bool:
    """Return True only for known VN index codes (not equity tickers)."""
    return _strip_vn_prefix(symbol) in _INDEX_SYMBOLS


@Registry.register_data_source("vnstock")
class VnstockDataSource(BaseDataSource):
    """KBS data source for Vietnamese equity OHLCV and fundamentals.

    Calls the KB Securities (KBS) REST API directly without the vnstock library.
    Use from_env() for live data; pass ohlcv_payloads / fundamentals_payloads for tests.
    Symbols use the VN: prefix (e.g. VN:VNM). No authentication required.

    Fundamentals are cached at ~/.qts/cache/vn_fundamentals/{ticker}.parquet
    in tidy long format. The cache is refreshed when the file is older than
    _FUNDAMENTALS_CACHE_TTL_HOURS (default 24 h).
    """

    CAPABILITIES = frozenset({DataType.OHLCV, DataType.FUNDAMENTALS})

    def __init__(
        self,
        client: _KBSClient | None = None,
        ohlcv_payloads: dict[str, pl.DataFrame] | None = None,
        fundamentals_payloads: dict[str, pl.DataFrame] | None = None,
    ) -> None:
        self._client = client
        self.ohlcv_payloads = ohlcv_payloads or {}
        self.fundamentals_payloads = fundamentals_payloads or {}

    @classmethod
    def from_env(cls) -> VnstockDataSource:
        """Build live client. No credentials required for KBS public data."""
        return cls(client=_KBSClient())

    async def fetch(self, data_type: DataType, symbol: str, **kwargs) -> pl.DataFrame:
        if data_type is DataType.OHLCV:
            return await self.get_ohlcv(
                symbol,
                kwargs.get("start"),
                kwargs.get("end"),
                kwargs.get("interval", "1d"),
            )
        if data_type is DataType.FUNDAMENTALS:
            return await self.get_fundamentals(
                symbol,
                termtype=kwargs.get("termtype", 1),
                pages=kwargs.get("pages", 3),
                force_refresh=kwargs.get("force_refresh", False),
            )
        raise NotImplementedError(f"vnstock does not support {data_type.value}.")

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
                raise DataSourceError("Unknown vnstock symbol", symbol, (start, end)) from exc
            frame = frame.select(OHLCV_COLUMNS)
            if start is None or end is None:
                return frame
            return frame.filter(pl.col("date").is_between(start, end))

        if start is None or end is None:
            raise DataSourceError("start and end are required for live KBS OHLCV", symbol)
        try:
            rows = self._client.get_ohlcv(
                symbol=_strip_vn_prefix(symbol),
                start=start,
                end=end,
                interval=interval,
                is_index=_is_index(symbol),
            )
        except Exception as exc:
            raise DataSourceError(str(exc), symbol, (start, end)) from exc
        return _rows_to_ohlcv(symbol, rows, scale=_price_scale(symbol))

    async def get_fundamentals(
        self,
        symbol: str,
        *,
        termtype: int = 1,
        pages: int = 3,
        force_refresh: bool = False,
    ) -> pl.DataFrame:
        """Return financial statements for *symbol* in tidy long format.

        Args:
            symbol:        QTS symbol (e.g. VN:VNM).
            termtype:      1 = annual (default), 2 = quarterly.
            pages:         History depth; pages × 4 = max periods fetched per statement.
            force_refresh: Bypass the on-disk cache and re-fetch from the API.

        Returns:
            Polars DataFrame with schema _FUNDAMENTALS_SCHEMA.
        """
        if self._client is None:
            try:
                return self.fundamentals_payloads[symbol]
            except KeyError as exc:
                raise DataSourceError("Unknown vnstock fundamentals symbol", symbol) from exc

        ticker = _strip_vn_prefix(symbol)
        cache_path = _fundamentals_cache_path(ticker, termtype)

        if not force_refresh and cache_path.exists():
            age_h = (
                datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
            ).total_seconds() / 3600
            if age_h < _FUNDAMENTALS_CACHE_TTL_HOURS:
                return pl.read_parquet(cache_path)

        try:
            frame = self._client.get_all_financials(ticker, termtype=termtype, pages=pages)
        except Exception as exc:
            raise DataSourceError(str(exc), symbol) from exc

        if not frame.is_empty():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(cache_path)

        return frame

    async def stream_ticks(self, symbols: list[str]):  # pragma: no cover - unsupported
        raise NotImplementedError("KBS does not support streaming ticks via REST.")


@Registry.register_data_source("vnstock_futures")
class VnstockFuturesDataSource(BaseDataSource):
    """KBS data source for VN30 index futures OHLCV.

    Symbols use the VNF: prefix (e.g. VNF:VN30F2503).
    Vietnam trades VN30 index futures only — no single-stock futures or options exist.
    Note: KBS adopted new KRX symbol format for contracts expiring after May 2025.
    Old-format symbols (VNF:VN30F2606) are auto-converted to KRX (41I1G6000).
    """

    CAPABILITIES = frozenset({DataType.FUTURES_OHLCV})

    def __init__(
        self,
        client: _KBSClient | None = None,
        ohlcv_payloads: dict[str, pl.DataFrame] | None = None,
    ) -> None:
        self._client = client
        self.ohlcv_payloads = ohlcv_payloads or {}

    @classmethod
    def from_env(cls) -> VnstockFuturesDataSource:
        """Build live client. No credentials required."""
        return cls(client=_KBSClient())

    async def fetch(self, data_type: DataType, symbol: str, **kwargs) -> pl.DataFrame:
        if data_type is not DataType.FUTURES_OHLCV:
            raise NotImplementedError(f"vnstock futures does not support {data_type.value}.")
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
                raise DataSourceError(
                    "Unknown vnstock futures symbol", symbol, (start, end)
                ) from exc
            frame = frame.select(OHLCV_COLUMNS)
            if start is None or end is None:
                return frame
            return frame.filter(pl.col("date").is_between(start, end))

        if start is None or end is None:
            raise DataSourceError(
                "start and end are required for live KBS futures OHLCV", symbol
            )
        try:
            rows = self._client.get_ohlcv(
                symbol=_to_krx_futures(_strip_vn_prefix(symbol)),
                start=start,
                end=end,
                interval=interval,
                is_index=False,
            )
        except Exception as exc:
            raise DataSourceError(str(exc), symbol, (start, end)) from exc
        return _rows_to_ohlcv(symbol, rows, scale=1.0)

    async def get_fundamentals(self, symbol: str) -> pl.DataFrame:
        raise NotImplementedError("vnstock futures fundamentals are not supported.")

    async def stream_ticks(self, symbols: list[str]):  # pragma: no cover - unsupported
        raise NotImplementedError("KBS does not support streaming ticks via REST.")
