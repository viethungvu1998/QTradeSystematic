"""Runtime assembly helpers for orchestration flows."""

from __future__ import annotations

from qts.config.builder import _collect_components
from qts.core.instrument import AssetType
from qts.data.manager import DataManager
from qts.data.sources.vnstock import VnstockDataSource


def resolved_brokers(resolved: object) -> dict[AssetType, object]:
    return _collect_components(resolved, suffix="broker")


def build_data_manager(
    resolved: object,
    *,
    include_bundle: bool | None = None,
) -> DataManager:
    raw = getattr(resolved, "raw", None)
    if include_bundle is None:
        include_bundle = AssetType.STOCK.value in getattr(raw, "asset_types", [])
    sources = _collect_components(resolved, suffix="source")
    vn_stock_source = sources.get(AssetType.VN_STOCK)
    vn_stock_fundamentals_source = _default_vn_stock_fundamentals_source(vn_stock_source)
    return DataManager(
        stock_source=sources.get(AssetType.STOCK),
        vn_stock_source=vn_stock_source,
        vn_stock_fundamentals_source=vn_stock_fundamentals_source,
        vn_warrant_source=sources.get(AssetType.VN_WARRANT),
        vn_futures_source=sources.get(AssetType.VN_FUTURES),
        crypto_source=sources.get(AssetType.CRYPTO),
        crypto_futures_source=sources.get(AssetType.CRYPTO_FUTURES),
        storage=resolved.storage,
        cache=resolved.cache,
        bundle_adapter=getattr(resolved, "bundle_adapter", None) if include_bundle else None,
    )


def _default_vn_stock_fundamentals_source(vn_stock_source: object | None) -> object | None:
    if not isinstance(vn_stock_source, VnstockDataSource):
        return None
    return VnstockDataSource(fundamentals_source="vci")
