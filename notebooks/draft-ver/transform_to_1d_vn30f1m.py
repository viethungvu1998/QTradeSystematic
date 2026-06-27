"""Transform VN30F1M intraday bars to daily OHLCV and merge into the 1d CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INTRADAY_PATH = REPO_ROOT / "data" / "VN30F1M.csv"
DEFAULT_DAILY_PATH = REPO_ROOT / "data" / "vni-etf" / "vn30f1m_prices_1d.csv"
DEFAULT_SYMBOL = "VNF:VN30F1M"

INTRADAY_COLUMNS = ["Time", "Open", "High", "Low", "Close", "Volume"]
DAILY_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]
DAILY_SCHEMA = {
    "date": pl.Date,
    "symbol": pl.Utf8,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}


def _read_intraday(path: Path) -> pl.DataFrame:
    frame = pl.read_csv(path)
    missing = sorted(set(INTRADAY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    parsed = frame.select(INTRADAY_COLUMNS).with_columns(
        pl.col("Time")
        .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
        .alias("bar_time"),
        pl.col("Open").cast(pl.Float64).alias("open"),
        pl.col("High").cast(pl.Float64).alias("high"),
        pl.col("Low").cast(pl.Float64).alias("low"),
        pl.col("Close").cast(pl.Float64).alias("close"),
        pl.col("Volume").cast(pl.Float64).alias("volume"),
    )
    if parsed["bar_time"].null_count() > 0:
        raise ValueError(f"{path} contains invalid Time values")
    return parsed.select(["bar_time", "open", "high", "low", "close", "volume"])


def transform_15m_to_1d(path: Path, *, symbol: str = DEFAULT_SYMBOL) -> pl.DataFrame:
    """Aggregate intraday VN30F1M bars to the standard daily OHLCV schema."""
    return (
        _read_intraday(path)
        .sort("bar_time")
        .with_columns(pl.col("bar_time").dt.date().alias("date"))
        .group_by("date")
        .agg(
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
        )
        .with_columns(pl.lit(symbol).alias("symbol"))
        .select(DAILY_COLUMNS)
        .sort("date")
    )


def _read_daily(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(schema=DAILY_SCHEMA)

    frame = pl.read_csv(path)
    missing = sorted(set(DAILY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    return frame.select(DAILY_COLUMNS).with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        pl.col("symbol").cast(pl.Utf8),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
    )


def merge_daily_prices(
    transformed: pl.DataFrame,
    existing: pl.DataFrame,
) -> pl.DataFrame:
    """Merge transformed rows into existing daily data, preserving existing overlaps."""
    combined = pl.concat(
        [
            transformed.with_columns(pl.lit(0).alias("_source_rank")),
            existing.with_columns(pl.lit(1).alias("_source_rank")),
        ],
        how="vertical",
    )
    return (
        combined.sort(["date", "symbol", "_source_rank"])
        .unique(subset=["date", "symbol"], keep="last", maintain_order=True)
        .sort(["date", "symbol"])
        .select(DAILY_COLUMNS)
    )


def transform_and_merge(
    *,
    intraday_path: Path = DEFAULT_INTRADAY_PATH,
    daily_path: Path = DEFAULT_DAILY_PATH,
    symbol: str = DEFAULT_SYMBOL,
) -> pl.DataFrame:
    transformed = transform_15m_to_1d(intraday_path, symbol=symbol)
    merged = merge_daily_prices(transformed, _read_daily(daily_path))
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_csv(daily_path)
    return merged


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform data/VN30F1M.csv 15-minute bars to 1d and merge them."
    )
    parser.add_argument("--intraday-path", type=Path, default=DEFAULT_INTRADAY_PATH)
    parser.add_argument("--daily-path", type=Path, default=DEFAULT_DAILY_PATH)
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    merged = transform_and_merge(
        intraday_path=args.intraday_path,
        daily_path=args.daily_path,
        symbol=args.symbol,
    )
    print(f"Saved rows: {merged.height:,}")
    print(f"Date range: {merged['date'].min()} to {merged['date'].max()}")
    print(f"Saved to: {args.daily_path}")


if __name__ == "__main__":
    main()
