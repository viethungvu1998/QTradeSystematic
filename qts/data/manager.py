"""Data manager."""

from __future__ import annotations

import asyncio
from datetime import date

import polars as pl

from qts.core.instrument import AssetType
from qts.data._schemas import TIME_COLUMN, DataType
from qts.data.base import BaseDataSource
from qts.data.bundles.base import BaseBundleAdapter
from qts.data.bundles.zipline_ingest import ingest_duckdb_to_bundle
from qts.data.storage.base import BaseStorage


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
        frames = await asyncio.gather(
            *(self._fetch_with_cache(data_type, symbol, **kwargs) for symbol in symbols)
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

        table = self._table_name(asset_type, data_type)
        if self.storage.exists(table):
            existing = self._db_lookup(table, symbol, data_type, **kwargs)
            if existing.height > 0:
                return self._sort_frame(existing, data_type)

        cache_key = self._cache_key(data_type, symbol, **kwargs)
        if self.cache is not None and self.cache.exists(cache_key):
            cached = self._sort_frame(self.cache.read(cache_key), data_type)
            if cached.height > 0:
                self.storage.append(table, cached)
                return cached

        frame = self._sort_frame(await source.fetch(data_type, symbol, **kwargs), data_type)
        if frame.height > 0:
            if self.cache is not None:
                self.cache.write(cache_key, frame)
            self.storage.append(table, frame)
        return frame

    def _db_lookup(self, table: str, symbol: str, data_type: DataType, **kwargs) -> pl.DataFrame:
        conditions = [f"symbol = '{symbol}'"]
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

    def _table_name(self, asset_type: AssetType, data_type: DataType) -> str:
        return self._table_map.get((asset_type, data_type), data_type.value)

    def _cache_key(self, data_type: DataType, symbol: str, **kwargs) -> str:
        normalized_kwargs = "_".join(
            f"{key}-{kwargs[key]}" for key in sorted(kwargs) if kwargs[key] is not None
        )
        suffix = f"_{normalized_kwargs}" if normalized_kwargs else ""
        return f"{data_type.value}_{symbol.replace('/', '_')}{suffix}"

    def _sort_frame(self, frame: pl.DataFrame, data_type: DataType) -> pl.DataFrame:
        time_column = TIME_COLUMN[data_type]
        sort_columns = [
            column for column in [time_column, "symbol"] if column and column in frame.columns
        ]
        if sort_columns:
            return frame.sort(sort_columns)
        if "symbol" in frame.columns:
            return frame.sort("symbol")
        return frame

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
