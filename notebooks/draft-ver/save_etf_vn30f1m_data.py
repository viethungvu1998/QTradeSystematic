"""Crawl Vietnam ETF and VN30F1M price data into ``data/vni-etf`` CSV files."""
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx
import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qts.data._schemas import FUTURES_INTRADAY_OHLCV_COLUMNS, OHLCV_COLUMNS
from qts.data.sources.vnstock import VnstockDataSource, VnstockFuturesDataSource
from qts.data.vn_symbols import strip_vn_prefix, to_vn_futures_symbol


ETF_LIST_URL = "https://kbbuddywts.kbsec.com.vn/iis-server/investment/index/FUND/stocks"
OUTPUT_DIR = REPO_ROOT / "data" / "vni-etf"
DEFAULT_START_DATE = date(2021, 1, 1)
DEFAULT_VN30F1M_SYMBOL = "VNF:VN30F1M"
ETF_PRICE_DIR_NAME = "etf_prices"

REQUEST_HEADERS = {
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

_EMPTY_INTRADAY_SCHEMA = {
    "bar_time": pl.Datetime,
    "date": pl.Date,
    "symbol": pl.Utf8,
    "interval": pl.Utf8,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}


@dataclass(frozen=True)
class CrawlResult:
    """Result summary for an ETF/VN30F1M crawl."""

    etf_symbols: list[str]
    etf_prices: pl.DataFrame
    vn30f1m_prices: pl.DataFrame
    failures: pl.DataFrame
    output_dir: Path


def _qts_vn_stock_symbol(symbol: str) -> str:
    raw = strip_vn_prefix(str(symbol).strip().upper())
    if not raw:
        raise ValueError("ETF symbol cannot be empty")
    return f"VN:{raw}"


def _symbol_from_payload_item(item: object) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("symbol", "SB", "sb"):
            value = item.get(key)
            if value:
                return str(value)
    return None


def _parse_etf_symbols(payload: object) -> list[str]:
    if isinstance(payload, dict):
        raw_items = payload.get("data", [])
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raise ValueError(f"Unsupported ETF listing response: {type(payload).__name__}")

    symbols = []
    for item in raw_items:
        symbol = _symbol_from_payload_item(item)
        if symbol:
            symbols.append(_qts_vn_stock_symbol(symbol))
    return sorted(dict.fromkeys(symbols))


def list_all_etfs(client: httpx.Client | None = None) -> list[str]:
    """Return all ETF symbols from KBS in QTS ``VN:`` symbol format."""
    if client is None:
        with httpx.Client(timeout=30, headers=REQUEST_HEADERS) as owned_client:
            response = owned_client.get(ETF_LIST_URL)
            response.raise_for_status()
            return _parse_etf_symbols(response.json())

    response = client.get(ETF_LIST_URL, headers=REQUEST_HEADERS)
    response.raise_for_status()
    return _parse_etf_symbols(response.json())


def _normalize_symbols(symbols: Iterable[str]) -> list[str]:
    return sorted(dict.fromkeys(_qts_vn_stock_symbol(symbol) for symbol in symbols))


def _is_intraday_interval(interval: str) -> bool:
    return interval.lower() not in {"1d", "1w", "1mo"}


def _empty_price_frame(interval: str = "1d") -> pl.DataFrame:
    schema = _EMPTY_INTRADAY_SCHEMA if _is_intraday_interval(interval) else _EMPTY_OHLCV_SCHEMA
    return pl.DataFrame(schema=schema)


def _sort_prices(frame: pl.DataFrame) -> pl.DataFrame:
    sort_columns = [
        column
        for column in ("symbol", "interval", "bar_time", "date")
        if column in frame.columns
    ]
    return frame.sort(sort_columns) if sort_columns else frame


async def _fetch_etf_prices(
    symbols: list[str],
    *,
    start: date,
    end: date,
    interval: str,
    sleep_seconds: float,
) -> tuple[pl.DataFrame, list[dict[str, str]]]:
    source = VnstockDataSource.from_env()
    frames: list[pl.DataFrame] = []
    failures: list[dict[str, str]] = []

    for symbol in symbols:
        try:
            frame = await source.get_ohlcv(symbol, start, end, interval)
            if not frame.is_empty():
                frames.append(frame.select(OHLCV_COLUMNS))
        except Exception as exc:
            failures.append({"symbol": symbol, "asset": "etf", "error": str(exc)})
        await asyncio.sleep(sleep_seconds)

    if not frames:
        return pl.DataFrame(schema=_EMPTY_OHLCV_SCHEMA), failures
    return _sort_prices(pl.concat(frames, how="vertical")), failures


async def _fetch_vn30f1m_prices(
    symbol: str,
    *,
    start: date,
    end: date,
    interval: str,
) -> tuple[pl.DataFrame, list[dict[str, str]]]:
    source = VnstockFuturesDataSource.from_env()
    try:
        frame = await source.get_ohlcv(symbol, start, end, interval)
    except Exception as exc:
        return _empty_price_frame(interval), [
            {"symbol": symbol, "asset": "vn30f1m", "error": str(exc)}
        ]

    if frame.is_empty():
        return _empty_price_frame(interval), []

    columns = FUTURES_INTRADAY_OHLCV_COLUMNS if _is_intraday_interval(interval) else OHLCV_COLUMNS
    return _sort_prices(frame.select(columns)), []


def _symbols_frame(symbols: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": symbols,
            "ticker": [strip_vn_prefix(symbol) for symbol in symbols],
            "asset": ["etf"] * len(symbols),
        }
    )


def _failures_frame(failures: list[dict[str, str]]) -> pl.DataFrame:
    if not failures:
        return pl.DataFrame(schema={"symbol": pl.Utf8, "asset": pl.Utf8, "error": pl.Utf8})
    return pl.DataFrame(failures).select(["symbol", "asset", "error"])


def _summary_frame(
    *,
    start: date,
    end: date,
    etf_interval: str,
    vn30f1m_interval: str,
    result: CrawlResult,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "start": [start],
            "end": [end],
            "etf_interval": [etf_interval],
            "vn30f1m_interval": [vn30f1m_interval],
            "etf_symbol_count": [len(result.etf_symbols)],
            "etf_price_rows": [result.etf_prices.height],
            "vn30f1m_price_rows": [result.vn30f1m_prices.height],
            "failure_count": [result.failures.height],
        }
    )


def _etf_price_path(output_dir: Path, symbol: str, interval: str) -> Path:
    return output_dir / ETF_PRICE_DIR_NAME / f"{strip_vn_prefix(symbol)}_{interval}.csv"


def _write_etf_price_files(
    output_dir: Path,
    symbols: list[str],
    prices: pl.DataFrame,
    interval: str,
) -> None:
    price_dir = output_dir / ETF_PRICE_DIR_NAME
    price_dir.mkdir(parents=True, exist_ok=True)

    old_combined_path = output_dir / f"etf_prices_{interval}.csv"
    if old_combined_path.exists():
        old_combined_path.unlink()
    for stale_path in price_dir.glob(f"*_{interval}.csv"):
        stale_path.unlink()

    for symbol in symbols:
        symbol_prices = prices.filter(pl.col("symbol") == symbol)
        symbol_prices.write_csv(_etf_price_path(output_dir, symbol, interval))


def _write_outputs(
    result: CrawlResult,
    *,
    start: date,
    end: date,
    etf_interval: str,
    vn30f1m_interval: str,
) -> None:
    result.output_dir.mkdir(parents=True, exist_ok=True)
    _symbols_frame(result.etf_symbols).write_csv(result.output_dir / "etf_symbols.csv")
    _write_etf_price_files(
        result.output_dir,
        result.etf_symbols,
        result.etf_prices,
        etf_interval,
    )
    result.vn30f1m_prices.write_csv(
        result.output_dir / f"vn30f1m_prices_{vn30f1m_interval}.csv"
    )
    result.failures.write_csv(result.output_dir / "crawl_failures.csv")
    _summary_frame(
        start=start,
        end=end,
        etf_interval=etf_interval,
        vn30f1m_interval=vn30f1m_interval,
        result=result,
    ).write_csv(result.output_dir / "crawl_summary.csv")


async def crawl_all_etf_and_vn30f1m_prices(
    *,
    start: date = DEFAULT_START_DATE,
    end: date | None = None,
    output_dir: Path = OUTPUT_DIR,
    etf_symbols: Iterable[str] | None = None,
    etf_interval: str = "1d",
    vn30f1m_symbol: str = DEFAULT_VN30F1M_SYMBOL,
    vn30f1m_interval: str = "1d",
    sleep_seconds: float = 0.3,
) -> CrawlResult:
    """Fetch ETF and VN30F1M prices through QTS sources and save them as CSV files."""
    end = end or date.today()
    normalized_etfs = (
        _normalize_symbols(etf_symbols) if etf_symbols is not None else list_all_etfs()
    )
    if not normalized_etfs:
        raise ValueError("No ETF symbols found")

    futures_symbol = to_vn_futures_symbol(vn30f1m_symbol)
    etf_prices, etf_failures = await _fetch_etf_prices(
        normalized_etfs,
        start=start,
        end=end,
        interval=etf_interval,
        sleep_seconds=sleep_seconds,
    )
    futures_prices, futures_failures = await _fetch_vn30f1m_prices(
        futures_symbol,
        start=start,
        end=end,
        interval=vn30f1m_interval,
    )

    result = CrawlResult(
        etf_symbols=normalized_etfs,
        etf_prices=etf_prices,
        vn30f1m_prices=futures_prices,
        failures=_failures_frame([*etf_failures, *futures_failures]),
        output_dir=Path(output_dir),
    )
    _write_outputs(
        result,
        start=start,
        end=end,
        etf_interval=etf_interval,
        vn30f1m_interval=vn30f1m_interval,
    )
    return result


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl Vietnam ETF and VN30F1M prices into data/vni-etf CSV files."
    )
    parser.add_argument("--start", type=_parse_date, default=DEFAULT_START_DATE)
    parser.add_argument("--end", type=_parse_date, default=date.today())
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--etf-interval", default="1d")
    parser.add_argument("--vn30f1m-interval", default="1d")
    parser.add_argument("--sleep-seconds", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = asyncio.run(
        crawl_all_etf_and_vn30f1m_prices(
            start=args.start,
            end=args.end,
            output_dir=args.output_dir,
            etf_interval=args.etf_interval,
            vn30f1m_interval=args.vn30f1m_interval,
            sleep_seconds=args.sleep_seconds,
        )
    )
    print(f"ETF symbols: {len(result.etf_symbols)}")
    print(f"ETF rows: {result.etf_prices.height:,}")
    print(f"VN30F1M rows: {result.vn30f1m_prices.height:,}")
    print(f"Failures: {result.failures.height}")
    print(f"Saved to: {result.output_dir}")


if __name__ == "__main__":
    main()
