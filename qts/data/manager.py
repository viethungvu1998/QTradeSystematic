"""Data manager."""

from __future__ import annotations

import asyncio
from datetime import date

import polars as pl

from qts.core.instrument import AssetType
from qts.data.bundles.base import BaseBundleAdapter
from qts.data.bundles.zipline_ingest import ingest_duckdb_to_bundle
from qts.data.storage.base import BaseStorage


class DataManager:
    """Routes requests to the correct source and storage layers."""

    def __init__(
        self,
        stock_source,
        crypto_source,
        storage: BaseStorage,
        cache: BaseStorage | None = None,
        bundle_adapter: BaseBundleAdapter | None = None,
        stock_table: str = "stock_prices",
        crypto_table: str = "crypto_prices",
        bundle_name: str = "qts-stock-bundle",
    ) -> None:
        self.source_map = {
            AssetType.STOCK: stock_source,
            AssetType.CRYPTO: crypto_source,
        }
        self.storage = storage
        self.cache = cache
        self.bundle_adapter = bundle_adapter
        self.stock_table = stock_table
        self.crypto_table = crypto_table
        self.bundle_name = bundle_name

    async def get_ohlcv(
        self,
        symbols: list[str],
        start: date | None,
        end: date | None,
        interval: str = "1d",
    ) -> pl.DataFrame:
        frames = await asyncio.gather(
            *(self._get_symbol_ohlcv(symbol, start, end, interval) for symbol in symbols)
        )
        combined = pl.concat(frames, how="vertical").sort(["date", "symbol"])
        if self.bundle_adapter is not None and any("/" not in symbol for symbol in symbols):
            ingest_duckdb_to_bundle(
                storage=self.storage,
                adapter=self.bundle_adapter,
                table=self.stock_table,
                bundle_name=self.bundle_name,
                start=start,
                end=end,
            )
        return combined

    async def _get_symbol_ohlcv(
        self,
        symbol: str,
        start: date | None,
        end: date | None,
        interval: str,
    ) -> pl.DataFrame:
        cache_key = f"{symbol.replace('/', '_')}_{start}_{end}_{interval}"
        asset_type = AssetType.from_symbol(symbol)
        table_name = self.stock_table if asset_type is AssetType.STOCK else self.crypto_table
        if self.cache is not None and self.cache.exists(cache_key):
            frame = self.cache.read(cache_key)
            self.storage.append(table_name, frame)
            return frame

        source = self.source_map[asset_type]
        frame = await source.get_ohlcv(symbol, start, end, interval)
        if self.cache is not None:
            self.cache.write(cache_key, frame)
        self.storage.append(table_name, frame)
        return frame

    async def get_fundamentals(self, symbols: list[str]) -> pl.DataFrame:
        frames = []
        for symbol in symbols:
            asset_type = AssetType.from_symbol(symbol)
            source = self.source_map[asset_type]
            try:
                frames.append(await source.get_fundamentals(symbol))
            except NotImplementedError:
                continue
        return pl.concat(frames, how="vertical") if frames else pl.DataFrame()
