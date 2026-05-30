"""Data manager."""

from __future__ import annotations

import asyncio
from datetime import date, timedelta

import polars as pl

from qts.core.errors import DataSourceError
from qts.core.instrument import AssetType
from qts.data._schemas import FUTURES_INTRADAY_OHLCV_COLUMNS, TIME_COLUMN, DataType
from qts.data.base import BaseDataSource
from qts.data.bundles.base import BaseBundleAdapter
from qts.data.bundles.zipline_ingest import ingest_duckdb_to_bundle
from qts.data.storage.base import BaseStorage

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
        if table == self.vn_futures_intraday_table and kwargs.get("interval") is not None:
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
        if (
            asset_type is AssetType.VN_FUTURES
            and data_type is DataType.FUTURES_OHLCV
            and self._is_intraday_interval(kwargs.get("interval", "1d"))
        ):
            return self.vn_futures_intraday_table
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
        if table != self.vn_futures_intraday_table or frame.height == 0:
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

    async def bulk_fetch_vn_fundamentals(
        self,
        symbols: list[str],
        termtype: int = 1,
        pages: int = 3,
        force_refresh: bool = False,
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
        for symbol in symbols:
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
