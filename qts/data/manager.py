"""Data manager."""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from qts.core.errors import DataSourceError
from qts.core.instrument import AssetType
from qts.data._schemas import FUTURES_INTRADAY_OHLCV_COLUMNS, TIME_COLUMN, DataType
from qts.data.base import BaseDataSource
from qts.data.bundles.base import BaseBundleAdapter
from qts.data.bundles.zipline_ingest import ingest_duckdb_to_bundle
from qts.data.storage.base import BaseStorage
from qts.data.vn_symbols import strip_vn_prefix
from qts.utils.paths import cache_dir

_PRICE_HISTORY_DATA_TYPES: dict[AssetType, DataType] = {
    AssetType.STOCK: DataType.OHLCV,
    AssetType.VN_STOCK: DataType.OHLCV,
    AssetType.VN_WARRANT: DataType.OHLCV,
    AssetType.VN_FUTURES: DataType.FUTURES_OHLCV,
    AssetType.CRYPTO: DataType.OHLCV,
    AssetType.CRYPTO_FUTURES: DataType.FUTURES_OHLCV,
}


class DataManager:
    """Routes requests to the correct source and storage layers."""

    def __init__(
        self,
        stock_source: BaseDataSource | None,
        crypto_source: BaseDataSource | None,
        storage: BaseStorage,
        cache: BaseStorage | None = None,
        bundle_adapter: BaseBundleAdapter | None = None,
        vn_stock_source: BaseDataSource | None = None,
        vn_stock_fundamentals_source: BaseDataSource | None = None,
        vn_warrant_source: BaseDataSource | None = None,
        vn_futures_source: BaseDataSource | None = None,
        crypto_futures_source: BaseDataSource | None = None,
        stock_table: str = "stock_prices",
        vn_stock_table: str = "vn_stock_prices",
        vn_warrant_table: str = "vn_warrant_prices",
        vn_futures_table: str = "vn_futures_prices",
        vn_futures_intraday_table: str = "vn_futures_intraday_prices",
        crypto_table: str = "crypto_prices",
        futures_table: str = "futures_prices",
        crypto_futures_intraday_table: str = "crypto_futures_intraday_prices",
        bundle_name: str = "qts-stock-bundle",
    ) -> None:
        self.storage = storage
        self.cache = cache
        self.bundle_adapter = bundle_adapter
        self.stock_table = stock_table
        self.vn_stock_table = vn_stock_table
        self.vn_warrant_table = vn_warrant_table
        self.vn_futures_table = vn_futures_table
        self.vn_futures_intraday_table = vn_futures_intraday_table
        self.crypto_table = crypto_table
        self.futures_table = futures_table
        self.crypto_futures_intraday_table = crypto_futures_intraday_table
        self.bundle_name = bundle_name

        source_map: dict[AssetType, BaseDataSource] = {}
        if stock_source is not None:
            source_map[AssetType.STOCK] = stock_source
        if vn_stock_source is not None:
            source_map[AssetType.VN_STOCK] = vn_stock_source
        if vn_warrant_source is not None:
            source_map[AssetType.VN_WARRANT] = vn_warrant_source
        if vn_futures_source is not None:
            source_map[AssetType.VN_FUTURES] = vn_futures_source
        if crypto_source is not None:
            source_map[AssetType.CRYPTO] = crypto_source
        if crypto_futures_source is not None:
            source_map[AssetType.CRYPTO_FUTURES] = crypto_futures_source

        self._capability_map: dict[tuple[AssetType, DataType], BaseDataSource] = {
            (asset_type, data_type): source
            for asset_type, source in source_map.items()
            for data_type in source.CAPABILITIES
        }
        if (
            vn_stock_fundamentals_source is not None
            and DataType.FUNDAMENTALS in vn_stock_fundamentals_source.CAPABILITIES
        ):
            self._capability_map[(AssetType.VN_STOCK, DataType.FUNDAMENTALS)] = (
                vn_stock_fundamentals_source
            )
        self._table_map: dict[tuple[AssetType, DataType], str] = {
            (AssetType.STOCK, DataType.OHLCV): self.stock_table,
            (AssetType.VN_STOCK, DataType.OHLCV): self.vn_stock_table,
            (AssetType.VN_WARRANT, DataType.OHLCV): self.vn_warrant_table,
            (AssetType.VN_FUTURES, DataType.FUTURES_OHLCV): self.vn_futures_table,
            (AssetType.CRYPTO, DataType.OHLCV): self.crypto_table,
            (AssetType.CRYPTO_FUTURES, DataType.FUTURES_OHLCV): self.futures_table,
        }

    async def get(self, data_type: DataType, symbols: list[str], **kwargs) -> pl.DataFrame:
        resolved_symbols = self._expand_symbols(data_type, symbols, **kwargs)
        frames = await asyncio.gather(
            *(self._fetch_with_cache(data_type, symbol, **kwargs) for symbol in resolved_symbols)
        )
        nonempty = [frame for frame in frames if frame.height > 0]
        if not nonempty:
            return pl.DataFrame()
        combined = pl.concat(nonempty, how="vertical")
        return self._sort_frame(combined, data_type)

    async def _fetch_with_cache(self, data_type: DataType, symbol: str, **kwargs) -> pl.DataFrame:
        asset_type = AssetType.from_symbol(symbol)
        source = self._capability_map.get((asset_type, data_type))
        if source is None:
            return pl.DataFrame()

        table = self._table_name(asset_type, data_type, **kwargs)
        cache_key = self._cache_key(data_type, symbol, **kwargs)
        if self.cache is not None and self.cache.exists(cache_key):
            cached = self._sort_frame(self.cache.read(cache_key), data_type)
            if cached.height > 0:
                cached = self._prepare_for_storage(table, cached, symbol, data_type, kwargs)
                self.storage.append(table, cached)
                return cached

        if self.storage.exists(table):
            existing = self._sort_frame(
                self._db_lookup(table, symbol, data_type, **kwargs),
                data_type,
            )
            if existing.height > 0:
                if TIME_COLUMN[data_type] is None:
                    return existing
                missing_ranges = self._missing_boundary_ranges(existing, data_type, kwargs)
                if not missing_ranges:
                    return existing
                return await self._fetch_store_and_reload(
                    source,
                    table,
                    data_type,
                    symbol,
                    missing_ranges,
                    kwargs,
                )

        return await self._fetch_store_and_reload(
            source,
            table,
            data_type,
            symbol,
            [dict(kwargs)],
            kwargs,
            cache_key=cache_key,
        )

    def _requested_time_bounds(
        self,
        data_type: DataType,
        kwargs: dict,
    ) -> tuple[str | None, object | None, object | None]:
        time_column = TIME_COLUMN[data_type]
        if time_column is None:
            return None, None, None
        return time_column, kwargs.get("start"), kwargs.get("end")

    def _missing_boundary_ranges(
        self,
        existing: pl.DataFrame,
        data_type: DataType,
        kwargs: dict,
    ) -> list[dict]:
        time_column, start, end = self._requested_time_bounds(data_type, kwargs)
        if time_column is None or time_column not in existing.columns:
            return []

        ranges: list[dict] = []
        min_existing = existing[time_column].min()
        max_existing = existing[time_column].max()
        if start is not None and min_existing is not None and min_existing > start:
            leading = dict(kwargs)
            leading["start"] = start
            leading["end"] = min_existing - timedelta(days=1)
            ranges.append(leading)
        if end is not None and max_existing is not None and max_existing < end:
            trailing = dict(kwargs)
            trailing["start"] = max_existing + timedelta(days=1)
            trailing["end"] = end
            ranges.append(trailing)
        return ranges

    async def _fetch_store_and_reload(
        self,
        source: BaseDataSource,
        table: str,
        data_type: DataType,
        symbol: str,
        ranges: list[dict],
        requested_kwargs: dict,
        *,
        cache_key: str | None = None,
    ) -> pl.DataFrame:
        fetched_frames = [
            self._prepare_for_storage(
                table,
                self._sort_frame(
                    await source.fetch(data_type, symbol, **range_kwargs),
                    data_type,
                ),
                symbol,
                data_type,
                range_kwargs,
            )
            for range_kwargs in ranges
        ]
        nonempty = [frame for frame in fetched_frames if frame.height > 0]
        if nonempty:
            fetched = self._sort_frame(pl.concat(nonempty, how="vertical"), data_type)
            self.storage.append(table, fetched)
            if cache_key is not None and self.cache is not None:
                self.cache.write(cache_key, fetched)

        if self.storage.exists(table):
            stored = self._sort_frame(
                self._db_lookup(table, symbol, data_type, **requested_kwargs),
                data_type,
            )
            if stored.height > 0:
                return stored
        if not nonempty:
            return pl.DataFrame()
        return fetched

    def _db_lookup(self, table: str, symbol: str, data_type: DataType, **kwargs) -> pl.DataFrame:
        conditions = [f"symbol = '{symbol}'"]
        intraday_tables = {self.vn_futures_intraday_table, self.crypto_futures_intraday_table}
        if table in intraday_tables and kwargs.get("interval") is not None:
            conditions.append(f"interval = '{kwargs['interval']}'")
        time_column = TIME_COLUMN[data_type]
        if time_column is not None:
            start = kwargs.get("start")
            end = kwargs.get("end")
            if start is not None:
                conditions.append(f"{time_column} >= '{start}'")
            if end is not None:
                conditions.append(f"{time_column} <= '{end}'")
        query = f"SELECT * FROM {table} WHERE {' AND '.join(conditions)}"
        return self.storage.query(query)

    def _expand_symbols(self, data_type: DataType, symbols: list[str], **kwargs) -> list[str]:
        expanded: list[str] = []
        for symbol in symbols:
            asset_type = AssetType.from_symbol(symbol)
            source = self._capability_map.get((asset_type, data_type))
            if source is None:
                expanded.append(symbol)
                continue
            resolved = source.expand_symbols(data_type, symbol, **kwargs)
            if not resolved:
                continue
            expanded.extend(resolved)
        return list(dict.fromkeys(expanded))

    def _table_name(self, asset_type: AssetType, data_type: DataType, **kwargs) -> str:
        if data_type is DataType.FUTURES_OHLCV and self._is_intraday_interval(kwargs.get("interval", "1d")):
            if asset_type is AssetType.VN_FUTURES:
                return self.vn_futures_intraday_table
            if asset_type is AssetType.CRYPTO_FUTURES:
                return self.crypto_futures_intraday_table
        return self._table_map.get((asset_type, data_type), data_type.value)

    def price_history_table(self, symbol: str) -> str | None:
        asset_type = AssetType.from_symbol(symbol)
        data_type = _PRICE_HISTORY_DATA_TYPES.get(asset_type)
        if data_type is None:
            return None
        return self._table_name(asset_type, data_type)

    def upsert_bars(
        self,
        table: str,
        frame: pl.DataFrame,
        sort_by: list[str],
        identity: list[str],
    ) -> pl.DataFrame:
        if not sort_by:
            raise ValueError("sort_by must contain at least one column")
        if not identity:
            raise ValueError("identity must contain at least one column")

        required_columns = list(dict.fromkeys([*identity, *sort_by]))
        missing_columns = [column for column in required_columns if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"upsert frame is missing columns: {missing_columns}")

        if frame.height > 0:
            incoming = frame.unique(subset=identity, keep="last", maintain_order=True).sort(
                sort_by
            )
            self.storage.append(table, incoming)
        elif not self.storage.exists(table):
            return frame

        stored = self.storage.read(table)
        missing_stored_columns = [
            column for column in required_columns if column not in stored.columns
        ]
        if missing_stored_columns:
            raise ValueError(f"stored table is missing columns: {missing_stored_columns}")

        canonical = stored.unique(subset=identity, keep="last", maintain_order=True).sort(sort_by)
        self.storage.write(table, canonical)
        return canonical

    def _cache_key(self, data_type: DataType, symbol: str, **kwargs) -> str:
        normalized_kwargs = "_".join(
            f"{key}-{kwargs[key]}" for key in sorted(kwargs) if kwargs[key] is not None
        )
        suffix = f"_{normalized_kwargs}" if normalized_kwargs else ""
        return f"{data_type.value}_{symbol.replace('/', '_')}{suffix}"

    def _sort_frame(self, frame: pl.DataFrame, data_type: DataType) -> pl.DataFrame:
        time_column = TIME_COLUMN[data_type]
        sort_columns = [
            column
            for column in ["bar_time", time_column, "symbol", "interval"]
            if column and column in frame.columns
        ]
        if sort_columns:
            return frame.sort(sort_columns)
        if "symbol" in frame.columns:
            return frame.sort("symbol")
        return frame

    def _prepare_for_storage(
        self,
        table: str,
        frame: pl.DataFrame,
        symbol: str,
        data_type: DataType,
        kwargs: dict,
    ) -> pl.DataFrame:
        intraday_tables = {self.vn_futures_intraday_table, self.crypto_futures_intraday_table}
        if table not in intraday_tables or frame.height == 0:
            return frame
        if "bar_time" not in frame.columns:
            start = kwargs.get("start")
            end = kwargs.get("end")
            raise DataSourceError(
                "Intraday VN futures bars require bar_time before storage",
                symbol,
                (start, end) if start is not None and end is not None else None,
            )
        interval = kwargs.get("interval", "1d")
        result = frame
        if "interval" not in result.columns:
            result = result.with_columns(pl.lit(interval).alias("interval"))
        result = result.with_columns(
            pl.col("bar_time").cast(pl.Datetime).alias("bar_time"),
            pl.col("bar_time").cast(pl.Date).alias("date"),
        )
        return self._sort_frame(result.select(FUTURES_INTRADAY_OHLCV_COLUMNS), data_type)

    @staticmethod
    def _is_intraday_interval(interval: str | None) -> bool:
        return interval is not None and interval.lower() not in {"1d", "1w", "1mo"}

    async def get_ohlcv(
        self,
        symbols: list[str],
        start: date | None,
        end: date | None,
        interval: str = "1d",
    ) -> pl.DataFrame:
        result = await self.get(DataType.OHLCV, symbols, start=start, end=end, interval=interval)
        should_ingest_bundle = (
            self.bundle_adapter is not None
            and any(AssetType.from_symbol(symbol) is AssetType.STOCK for symbol in symbols)
        )
        if should_ingest_bundle:
            ingest_duckdb_to_bundle(
                storage=self.storage,
                adapter=self.bundle_adapter,
                table=self.stock_table,
                bundle_name=self.bundle_name,
                start=start,
                end=end,
            )
        return result

    async def get_futures_ohlcv(
        self,
        symbols: list[str],
        start: date | None,
        end: date | None,
        interval: str = "1d",
    ) -> pl.DataFrame:
        return await self.get(
            DataType.FUTURES_OHLCV, symbols, start=start, end=end, interval=interval
        )

    async def get_vn_futures_ohlcv(
        self,
        symbols: list[str],
        start: date | None,
        end: date | None,
        interval: str = "1d",
    ) -> pl.DataFrame:
        return await self.get(
            DataType.FUTURES_OHLCV, symbols, start=start, end=end, interval=interval
        )

    async def get_fundamentals(self, symbols: list[str]) -> pl.DataFrame:
        return await self.get(DataType.FUNDAMENTALS, symbols)

    async def fetch_vn_fundamentals(
        self,
        symbols: list[str],
        start: date,
        end: date,
        show_progress: bool = False,
    ) -> pl.DataFrame:
        if start > end:
            raise ValueError("start must be on or before end")

        vn_symbols = self._normalize_vn_fundamental_symbols(symbols)
        source = self._capability_map.get((AssetType.VN_STOCK, DataType.FUNDAMENTALS))
        if source is None:
            raise ValueError("VN fundamentals source is not configured")

        annual_years = self._requested_annual_years(start, end)
        quarterly_keys = self._requested_quarterly_keys(start, end)

        annual_missing = [
            symbol
            for symbol in vn_symbols
            if self._missing_annual_years(symbol, annual_years)
        ]
        quarterly_missing = [
            symbol
            for symbol in vn_symbols
            if self._missing_quarterly_keys(symbol, quarterly_keys)
        ]

        if annual_missing:
            annual_before = {
                symbol: self._load_vn_fundamentals_cache(symbol, termtype=1)
                for symbol in annual_missing
            }
            annual_pages = max(
                self._infer_annual_pages(symbol, annual_years) for symbol in annual_missing
            )
            await self.bulk_fetch_vn_fundamentals(
                annual_missing,
                termtype=1,
                pages=annual_pages,
                force_refresh=True,
                progress_label="Annual fundamentals" if show_progress else None,
            )
            self._merge_vn_fundamentals_caches(annual_before, termtype=1)
            still_missing = [
                symbol
                for symbol in annual_missing
                if self._missing_annual_years(symbol, annual_years)
            ]
            if still_missing:
                annual_pages = max(
                    self._infer_annual_pages(symbol, annual_years) for symbol in still_missing
                )
                await self.bulk_fetch_vn_fundamentals(
                    still_missing,
                    termtype=1,
                    pages=annual_pages,
                    force_refresh=True,
                    progress_label="Annual fundamentals retry" if show_progress else None,
                )
                self._merge_vn_fundamentals_caches(
                    {
                        symbol: self._merge_frames(
                            annual_before[symbol],
                            self._load_vn_fundamentals_cache(symbol, termtype=1),
                        )
                        for symbol in still_missing
                    },
                    termtype=1,
                )

        if quarterly_missing:
            quarterly_before = {
                symbol: self._load_vn_fundamentals_cache(symbol, termtype=2)
                for symbol in quarterly_missing
            }
            quarterly_pages = max(
                self._infer_quarterly_pages(symbol, quarterly_keys) for symbol in quarterly_missing
            )
            await self.bulk_fetch_vn_fundamentals(
                quarterly_missing,
                termtype=2,
                pages=quarterly_pages,
                force_refresh=True,
                progress_label="Quarterly fundamentals" if show_progress else None,
            )
            self._merge_vn_fundamentals_caches(quarterly_before, termtype=2)
            still_missing = [
                symbol
                for symbol in quarterly_missing
                if self._missing_quarterly_keys(symbol, quarterly_keys)
            ]
            if still_missing:
                quarterly_pages = max(
                    self._infer_quarterly_pages(symbol, quarterly_keys) for symbol in still_missing
                )
                await self.bulk_fetch_vn_fundamentals(
                    still_missing,
                    termtype=2,
                    pages=quarterly_pages,
                    force_refresh=True,
                    progress_label="Quarterly fundamentals retry" if show_progress else None,
                )
                self._merge_vn_fundamentals_caches(
                    {
                        symbol: self._merge_frames(
                            quarterly_before[symbol],
                            self._load_vn_fundamentals_cache(symbol, termtype=2),
                        )
                        for symbol in still_missing
                    },
                    termtype=2,
                )

        annual_frames = [
            self._filter_annual_frame(
                self._load_vn_fundamentals_cache(symbol, termtype=1),
                annual_years,
            ).with_columns(pl.lit("annual").alias("frequency"))
            for symbol in vn_symbols
        ]
        quarterly_frames = [
            self._filter_quarterly_frame(
                self._load_vn_fundamentals_cache(symbol, termtype=2),
                quarterly_keys,
            ).with_columns(pl.lit("quarterly").alias("frequency"))
            for symbol in vn_symbols
        ]
        nonempty = [frame for frame in [*annual_frames, *quarterly_frames] if frame.height > 0]
        if not nonempty:
            return pl.DataFrame()

        combined = pl.concat(nonempty, how="vertical_relaxed")
        sort_columns = [
            column
            for column in [
                "symbol",
                "frequency",
                "fiscal_year",
                "quarter",
                "report_date",
                "report_type",
                "item_en",
            ]
            if column in combined.columns
        ]
        return combined.sort(sort_columns) if sort_columns else combined

    async def bulk_fetch_vn_fundamentals(
        self,
        symbols: list[str],
        termtype: int = 1,
        pages: int = 3,
        force_refresh: bool = False,
        progress_label: str | None = None,
    ) -> None:
        """Crawl VN fundamentals for *symbols* with per-request rate limiting.

        Calls get_fundamentals() on the registered VN stock source, which writes
        results to ~/.qts/cache/vn_fundamentals/{ticker}_{annual|quarterly}.parquet.
        Sleeps 0.3 s between requests to stay within KBS rate limits.
        """
        source = self._capability_map.get((AssetType.VN_STOCK, DataType.FUNDAMENTALS))
        if source is None:
            return
        get_fn = getattr(source, "get_fundamentals", None)
        if get_fn is None:
            return
        progress = self._build_progress(symbols, progress_label)
        iterator = progress if progress is not None else symbols
        for symbol in iterator:
            try:
                await get_fn(
                    symbol,
                    termtype=termtype,
                    pages=pages,
                    force_refresh=force_refresh,
                )
            except Exception:
                pass
            await asyncio.sleep(0.3)
        if progress is not None:
            progress.close()

    def _normalize_vn_fundamental_symbols(self, symbols: list[str]) -> list[str]:
        unique_symbols = list(dict.fromkeys(symbols))
        invalid = [symbol for symbol in unique_symbols if AssetType.from_symbol(symbol) is not AssetType.VN_STOCK]
        if invalid:
            raise ValueError(f"fetch_vn_fundamentals supports VN stock symbols only: {invalid}")
        return unique_symbols

    @staticmethod
    def _build_progress(
        symbols: list[str],
        label: str | None,
    ) -> object | None:
        if not label or not symbols:
            return None
        try:
            from tqdm.auto import tqdm  # noqa: PLC0415
        except ImportError:
            return None
        return tqdm(symbols, desc=label, unit="symbol")

    @staticmethod
    def _requested_annual_years(start: date, end: date) -> list[int]:
        return list(range(start.year, end.year + 1))

    @staticmethod
    def _requested_quarterly_keys(start: date, end: date) -> list[tuple[int, int]]:
        keys: list[tuple[int, int]] = []
        year = start.year
        quarter = ((start.month - 1) // 3) + 1
        end_key = (end.year, ((end.month - 1) // 3) + 1)
        while (year, quarter) <= end_key:
            keys.append((year, quarter))
            if quarter == 4:
                year += 1
                quarter = 1
            else:
                quarter += 1
        return keys

    @staticmethod
    def _vn_fundamentals_cache_path(symbol: str, termtype: int) -> Path:
        label = "annual" if termtype == 1 else "quarterly"
        return cache_dir() / "vn_fundamentals" / f"{strip_vn_prefix(symbol)}_{label}.parquet"

    def _load_vn_fundamentals_cache(self, symbol: str, *, termtype: int) -> pl.DataFrame:
        path = self._vn_fundamentals_cache_path(symbol, termtype)
        if not path.exists():
            return pl.DataFrame()
        return pl.read_parquet(path)

    def _load_vn_fundamentals_cache_columns(
        self,
        symbol: str,
        *,
        termtype: int,
        columns: list[str],
    ) -> pl.DataFrame:
        path = self._vn_fundamentals_cache_path(symbol, termtype)
        if not path.exists():
            return pl.DataFrame()
        return pl.read_parquet(path, columns=columns)

    def _missing_annual_years(self, symbol: str, years: list[int]) -> list[int]:
        frame = self._load_vn_fundamentals_cache_columns(symbol, termtype=1, columns=["fiscal_year"])
        if frame.is_empty() or "fiscal_year" not in frame.columns:
            return years
        covered = {
            int(value)
            for value in frame["fiscal_year"].drop_nulls().cast(pl.Int64).to_list()
        }
        return [year for year in years if year not in covered]

    def _missing_quarterly_keys(
        self,
        symbol: str,
        keys: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        frame = self._load_vn_fundamentals_cache_columns(
            symbol,
            termtype=2,
            columns=["fiscal_year", "quarter"],
        )
        if frame.is_empty() or "fiscal_year" not in frame.columns or "quarter" not in frame.columns:
            return keys
        covered = {
            (int(row[0]), int(row[1]))
            for row in frame.select(
                pl.col("fiscal_year").cast(pl.Int64),
                pl.col("quarter").cast(pl.Int64, strict=False),
            ).drop_nulls().iter_rows()
        }
        return [key for key in keys if key not in covered]

    def _infer_annual_pages(self, symbol: str, years: list[int]) -> int:
        missing = self._missing_annual_years(symbol, years)
        if not missing:
            return 1
        frame = self._load_vn_fundamentals_cache_columns(symbol, termtype=1, columns=["fiscal_year"])
        latest_year = years[-1]
        if not frame.is_empty() and "fiscal_year" in frame.columns:
            cached_max = frame.select(pl.col("fiscal_year").cast(pl.Int64).max()).item()
            if cached_max is not None:
                latest_year = max(latest_year, int(cached_max))
        span = max(1, latest_year - min(missing) + 1)
        return max(1, (span + 3) // 4)

    def _infer_quarterly_pages(self, symbol: str, keys: list[tuple[int, int]]) -> int:
        missing = self._missing_quarterly_keys(symbol, keys)
        if not missing:
            return 1
        latest_year, latest_quarter = keys[-1]
        frame = self._load_vn_fundamentals_cache_columns(
            symbol,
            termtype=2,
            columns=["fiscal_year", "quarter"],
        )
        if not frame.is_empty() and {"fiscal_year", "quarter"}.issubset(frame.columns):
            latest_cached = frame.select(
                (
                    pl.col("fiscal_year").cast(pl.Int64) * 4
                    + pl.col("quarter").cast(pl.Int64, strict=False)
                ).max()
            ).item()
            if latest_cached is not None:
                latest_cached = int(latest_cached)
                latest_year = max(latest_year, (latest_cached - 1) // 4)
                latest_quarter = max(latest_quarter, ((latest_cached - 1) % 4) + 1)
        latest_index = latest_year * 4 + latest_quarter - 1
        oldest_year, oldest_quarter = min(missing)
        oldest_index = oldest_year * 4 + oldest_quarter - 1
        span = max(1, latest_index - oldest_index + 1)
        return max(1, (span + 3) // 4)

    def _merge_vn_fundamentals_caches(
        self,
        previous_frames: dict[str, pl.DataFrame],
        *,
        termtype: int,
    ) -> None:
        for symbol, previous in previous_frames.items():
            path = self._vn_fundamentals_cache_path(symbol, termtype)
            current = self._load_vn_fundamentals_cache(symbol, termtype=termtype)
            merged = self._merge_frames(previous, current)
            if merged.height == 0:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            merged.write_parquet(path)

    @staticmethod
    def _merge_frames(previous: pl.DataFrame, current: pl.DataFrame) -> pl.DataFrame:
        nonempty = [frame for frame in (previous, current) if frame.height > 0]
        if not nonempty:
            return pl.DataFrame()
        merged = pl.concat(nonempty, how="vertical_relaxed")
        identity = [
            column
            for column in [
                "symbol",
                "report_type",
                "period",
                "fiscal_year",
                "quarter",
                "report_date",
                "item_en",
            ]
            if column in merged.columns
        ]
        if not identity:
            return merged
        return merged.unique(subset=identity, keep="last", maintain_order=True)

    @staticmethod
    def _filter_annual_frame(frame: pl.DataFrame, years: list[int]) -> pl.DataFrame:
        if frame.is_empty() or "fiscal_year" not in frame.columns:
            return frame
        return frame.filter(pl.col("fiscal_year").cast(pl.Int64).is_in(years))

    @staticmethod
    def _filter_quarterly_frame(
        frame: pl.DataFrame,
        keys: list[tuple[int, int]],
    ) -> pl.DataFrame:
        if frame.is_empty() or not {"fiscal_year", "quarter"}.issubset(frame.columns):
            return frame
        key_frame = pl.DataFrame(
            keys,
            schema=["fiscal_year", "quarter"],
            orient="row",
        )
        return frame.join(
            key_frame.with_columns(
                pl.col("fiscal_year").cast(frame.schema["fiscal_year"]),
                pl.col("quarter").cast(frame.schema["quarter"]),
            ),
            on=["fiscal_year", "quarter"],
            how="inner",
        )
