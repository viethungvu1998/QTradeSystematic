"""Config resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qts.config.loader import load_config
from qts.core.registry import Registry
from qts.data.bundles.local import LocalBundleAdapter
from qts.research.features.pipeline import FeaturePipeline
from qts.utils.paths import bundle_dir, cache_dir, database_path


@dataclass(slots=True)
class ResolvedConfig:
    """Fully resolved runtime dependencies."""

    raw: object
    stock_source: object | None
    vn_stock_source: object | None
    vn_warrant_source: object | None
    vn_futures_source: object | None
    crypto_source: object | None
    crypto_futures_source: object | None
    storage: object
    cache: object
    bundle_adapter: object
    feature_pipeline: FeaturePipeline
    strategy: object
    engine: object
    fill_model: object | None
    slippage_model: object | None
    commission_model: object | None
    calendar: object | None
    stock_broker: object | None
    vn_stock_broker: object | None
    crypto_broker: object | None


class Config:
    """Config builder."""

    @staticmethod
    def build(path: str | Path) -> ResolvedConfig:
        raw = load_config(path)
        stock_source = (
            Registry.get_data_source(raw.data_sources.stock)() if raw.data_sources.stock else None
        )
        vn_stock_source = (
            Registry.get_data_source(raw.data_sources.vn_stock)() if raw.data_sources.vn_stock else None
        )
        vn_warrant_source = (
            Registry.get_data_source(raw.data_sources.vn_warrant)()
            if raw.data_sources.vn_warrant
            else None
        )
        vn_futures_source = (
            Registry.get_data_source(raw.data_sources.vn_futures)()
            if raw.data_sources.vn_futures
            else None
        )
        crypto_source = (
            Registry.get_data_source(raw.data_sources.crypto)() if raw.data_sources.crypto else None
        )
        crypto_futures_source = (
            Registry.get_data_source(raw.data_sources.crypto_futures)()
            if raw.data_sources.crypto_futures
            else None
        )
        storage = _build_storage(raw.storage, role="primary")
        cache = _build_storage("parquet", role="cache")
        bundle_adapter = LocalBundleAdapter(root=bundle_dir())
        feature_pipeline = FeaturePipeline(_resolve_features(raw))
        strategy = Registry.get_strategy(raw.strategy.type)(**raw.strategy.params)
        engine = Registry.get_engine(raw.backtest_engine)()
        fill_model = Registry.get_fill_model(raw.fill_model)() if raw.fill_model else None
        slippage_model = (
            Registry.get_slippage_model(raw.slippage_model)() if raw.slippage_model else None
        )
        commission_model = (
            Registry.get_commission_model(raw.commission.model)(rate=raw.commission.rate)
            if raw.commission
            else None
        )
        calendar = Registry.get_calendar(raw.calendar)() if raw.calendar else None
        stock_broker = Registry.get_broker(raw.brokers.stock)() if raw.brokers and raw.brokers.stock else None
        vn_stock_broker = (
            Registry.get_broker(raw.brokers.vn_stock)()
            if raw.brokers and raw.brokers.vn_stock
            else None
        )
        crypto_broker = (
            Registry.get_broker(raw.brokers.crypto)() if raw.brokers and raw.brokers.crypto else None
        )
        return ResolvedConfig(
            raw=raw,
            stock_source=stock_source,
            vn_stock_source=vn_stock_source,
            vn_warrant_source=vn_warrant_source,
            vn_futures_source=vn_futures_source,
            crypto_source=crypto_source,
            crypto_futures_source=crypto_futures_source,
            storage=storage,
            cache=cache,
            bundle_adapter=bundle_adapter,
            feature_pipeline=feature_pipeline,
            strategy=strategy,
            engine=engine,
            fill_model=fill_model,
            slippage_model=slippage_model,
            commission_model=commission_model,
            calendar=calendar,
            stock_broker=stock_broker,
            vn_stock_broker=vn_stock_broker,
            crypto_broker=crypto_broker,
        )


def _build_storage(name: str, *, role: str) -> object:
    storage_cls = Registry.get_storage(name)
    if name == "duckdb":
        return storage_cls(database=str(database_path()))
    if name == "parquet":
        root = cache_dir() if role == "cache" else database_path().parent
        return storage_cls(root=root)
    return storage_cls()


def _resolve_features(raw) -> list[object]:
    features: list[object] = []
    if raw.features.indicators:
        for indicator in raw.features.indicators:
            features.append(Registry.get_feature(indicator.name)(**indicator.params))
    elif raw.features.technical:
        features.append(Registry.get_feature("technical")())
    if raw.features.fundamental:
        features.append(Registry.get_feature("fundamental")())
    if raw.features.onchain:
        features.append(Registry.get_feature("onchain")())
    if raw.features.forward_returns.periods:
        features.append(Registry.get_feature("forward_returns")(periods=raw.features.forward_returns.periods))
    return features
