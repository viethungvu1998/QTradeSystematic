"""Config resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp

from qts.core.registry import Registry
from qts.data.bundles.local import LocalBundleAdapter
from qts.data.manager import DataManager
from qts.data.storage.parquet import ParquetStorage
from qts.research.features.forward_returns import ForwardReturns
from qts.research.features.fundamentals import FundamentalFeatures
from qts.research.features.onchain import OnchainFeatures
from qts.research.features.pipeline import FeaturePipeline
from qts.research.features.technical import TechnicalFeatures
from qts.config.loader import load_config


@dataclass(slots=True)
class ResolvedConfig:
    """Fully resolved runtime dependencies."""

    raw: object
    stock_source: object | None
    crypto_source: object | None
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
    crypto_broker: object | None


class Config:
    """Config builder."""

    @staticmethod
    def build(path: str | Path) -> ResolvedConfig:
        raw = load_config(path)
        stock_source = (
            Registry.get_data_source(raw.data_sources.stock)() if raw.data_sources.stock else None
        )
        crypto_source = (
            Registry.get_data_source(raw.data_sources.crypto)() if raw.data_sources.crypto else None
        )
        storage = Registry.get_storage(raw.storage)()
        cache = ParquetStorage(Path(mkdtemp()) / "cache")
        bundle_adapter = LocalBundleAdapter(Path(mkdtemp()) / "bundle")
        features = []
        if raw.features.technical:
            features.append(TechnicalFeatures())
        if raw.features.fundamental:
            features.append(FundamentalFeatures())
        if raw.features.onchain:
            features.append(OnchainFeatures())
        if raw.features.forward_returns.periods:
            features.append(ForwardReturns(raw.features.forward_returns.periods))
        feature_pipeline = FeaturePipeline(features)
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
        crypto_broker = (
            Registry.get_broker(raw.brokers.crypto)() if raw.brokers and raw.brokers.crypto else None
        )
        return ResolvedConfig(
            raw=raw,
            stock_source=stock_source,
            crypto_source=crypto_source,
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
            crypto_broker=crypto_broker,
        )
