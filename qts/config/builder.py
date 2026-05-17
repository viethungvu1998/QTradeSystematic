"""Config resolution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from qts.config.loader import load_config
from qts.core.instrument import AssetType
from qts.core.registry import Registry
from qts.data.bundles.local import LocalBundleAdapter
from qts.research.backtest.base import BacktestConfig
from qts.research.features.pipeline import FeaturePipeline
from qts.utils.paths import bundle_dir, cache_dir, database_path

ASSET_COMPONENTS: tuple[tuple[AssetType, str], ...] = (
    (AssetType.STOCK, "stock"),
    (AssetType.VN_STOCK, "vn_stock"),
    (AssetType.VN_WARRANT, "vn_warrant"),
    (AssetType.VN_FUTURES, "vn_futures"),
    (AssetType.CRYPTO, "crypto"),
    (AssetType.CRYPTO_FUTURES, "crypto_futures"),
)


@dataclass(slots=True)
class ResolvedConfig:
    """Fully resolved runtime dependencies."""

    raw: BacktestConfig
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
    vn_warrant_broker: object | None
    vn_futures_broker: object | None
    crypto_broker: object | None
    crypto_futures_broker: object | None

    def data_sources(self) -> dict[AssetType, object]:
        return _collect_components(self, suffix="source")

    def brokers(self) -> dict[AssetType, object]:
        return _collect_components(self, suffix="broker")

    def uses_asset_type(self, asset_type: AssetType) -> bool:
        return asset_type.value in self.raw.asset_types

    def with_fundamentals(self, fundamentals: pl.DataFrame) -> FeaturePipeline:
        return self.feature_pipeline.with_fundamentals(fundamentals)


class Config:
    """Config builder."""

    @staticmethod
    def build(path: str | Path) -> ResolvedConfig:
        raw = load_config(path)
        source_components = _build_component_fields(
            raw.data_sources,
            suffix="source",
            builder=_build_data_source,
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
        broker_components = _build_component_fields(
            raw.brokers,
            suffix="broker",
            builder=_build_broker,
        )
        return ResolvedConfig(
            raw=raw,
            **source_components,
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
            **broker_components,
        )


def _build_component_fields(
    section: object | None,
    *,
    suffix: str,
    builder: Callable[[str | None], object | None],
) -> dict[str, object | None]:
    return {
        f"{name}_{suffix}": builder(getattr(section, name, None) if section is not None else None)
        for _, name in ASSET_COMPONENTS
    }


def _build_storage(name: str, *, role: str) -> object:
    storage_cls = Registry.get_storage(name)
    if name == "duckdb":
        return storage_cls(database=str(database_path()))
    if name == "parquet":
        root = cache_dir() if role == "cache" else database_path().parent
        return storage_cls(root=root)
    return storage_cls()


def _build_data_source(name: str | None) -> object | None:
    if not name:
        return None
    source_cls = Registry.get_data_source(name)
    if name in {"dnse", "vnstock", "vnstock_futures"} and hasattr(source_cls, "from_env"):
        try:
            return source_cls.from_env()
        except KeyError:
            # Preserve fixture-friendly construction when live credentials are absent.
            pass
    return source_cls()


def _build_broker(name: str | None) -> object | None:
    return Registry.get_broker(name)() if name else None


def _collect_components(resolved: object, *, suffix: str) -> dict[AssetType, object]:
    return {
        asset_type: component
        for asset_type, name in ASSET_COMPONENTS
        if (component := getattr(resolved, f"{name}_{suffix}", None)) is not None
    }


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
