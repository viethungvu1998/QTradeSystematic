"""Config resolution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import polars as pl

from qts.config.loader import load_config, load_config_from_mapping
from qts.core.instrument import AssetType
from qts.core.registry import Registry
from qts.data.bundles.local import LocalBundleAdapter
from qts.research.backtest.base import BacktestConfig, PortfolioConstructionConfig
from qts.research.features.base import BaseFeature
from qts.research.features.pipeline import FeaturePipeline
from qts.utils.paths import bundle_dir, cache_dir, database_path


class _FunctionTransformAdapter(BaseFeature):
    """Wraps a registry-resolved transform function as a BaseFeature step."""

    def __init__(self, fn: Callable[..., pl.DataFrame], params: dict[str, Any]) -> None:
        self._fn = fn
        self._params = params

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return self._fn(df, **self._params)


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
        return Config._build(load_config(path))

    @staticmethod
    def build_from_mapping(raw: Mapping[str, Any]) -> ResolvedConfig:
        return Config._build(load_config_from_mapping(raw))

    @staticmethod
    def _build(raw: BacktestConfig) -> ResolvedConfig:
        source_components = _build_component_fields(
            raw.data_sources,
            suffix="source",
            builder=_build_data_source,
        )
        storage = _build_storage(raw.storage, role="primary")
        cache = _build_storage("parquet", role="cache")
        bundle_adapter = LocalBundleAdapter(root=bundle_dir())
        portfolio_func = _build_portfolio_constructor(raw.portfolio_construction)
        feature_pipeline = FeaturePipeline(
            features=_resolve_features(raw),
            transforms=_resolve_transforms(raw),
        )
        strategy = _build_strategy(raw, portfolio_func=portfolio_func)
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


def _build_strategy(
    raw: BacktestConfig,
    *,
    portfolio_func: Callable | None = None,
) -> object:
    strategy_cls = Registry.get_strategy(raw.strategy.type)
    if hasattr(strategy_cls, "from_config_params"):
        return strategy_cls.from_config_params(
            raw.strategy.params,
            portfolio_func=portfolio_func,
        )
    try:
        return strategy_cls(**raw.strategy.params, portfolio_func=portfolio_func)
    except TypeError:
        import warnings

        warnings.warn(
            f"{strategy_cls.__name__} does not accept portfolio_func; "
            "portfolio_construction config will be ignored for this strategy.",
            stacklevel=2,
        )
        return strategy_cls(**raw.strategy.params)


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


def _build_portfolio_constructor(
    cfg: PortfolioConstructionConfig | None,
) -> Callable | None:
    if cfg is None:
        return None
    fn = Registry.get_portfolio_constructor(cfg.name)
    return partial(fn, **cfg.params)


def _collect_components(resolved: object, *, suffix: str) -> dict[AssetType, object]:
    return {
        asset_type: component
        for asset_type, name in ASSET_COMPONENTS
        if (component := getattr(resolved, f"{name}_{suffix}", None)) is not None
    }


def _resolve_features(raw: BacktestConfig) -> list[BaseFeature]:
    features: list[BaseFeature] = []
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


def _resolve_transforms(raw: BacktestConfig) -> list[BaseFeature]:
    return [
        _FunctionTransformAdapter(Registry.get_transform(step.name), step.params)
        for step in raw.features.transforms
    ]
