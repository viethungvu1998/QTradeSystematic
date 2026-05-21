"""Plugin registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from qts.core.errors import RegistryError

RegistryDict = dict[str, Any]


class Registry:
    """Central registry for pluggable implementations."""

    _data_sources: ClassVar[RegistryDict] = {}
    _storages: ClassVar[RegistryDict] = {}
    _features: ClassVar[RegistryDict] = {}
    _strategies: ClassVar[RegistryDict] = {}
    _engines: ClassVar[RegistryDict] = {}
    _brokers: ClassVar[RegistryDict] = {}
    _fill_models: ClassVar[RegistryDict] = {}
    _slippage_models: ClassVar[RegistryDict] = {}
    _commission_models: ClassVar[RegistryDict] = {}
    _calendars: ClassVar[RegistryDict] = {}
    _factor_trainers: ClassVar[RegistryDict] = {}
    _portfolio_constructors: ClassVar[RegistryDict] = {}
    _transforms: ClassVar[RegistryDict] = {}
    _signal_algorithms: ClassVar[RegistryDict] = {}
    _spread_models: ClassVar[RegistryDict] = {}
    _signal_rules: ClassVar[RegistryDict] = {}
    _models: ClassVar[RegistryDict] = {}

    @classmethod
    def _register(cls, registry: RegistryDict, name: str) -> Callable[[Any], Any]:
        def decorator(target: Any) -> Any:
            registry[name] = target
            return target

        return decorator

    @classmethod
    def _get(cls, registry: RegistryDict, name: str, kind: str) -> Any:
        try:
            return registry[name]
        except KeyError as exc:
            raise RegistryError(f"Unknown {kind}: {name}") from exc

    @classmethod
    def register_data_source(cls, name: str) -> Callable[[type[Any]], type[Any]]:
        return cls._register(cls._data_sources, name)

    @classmethod
    def get_data_source(cls, name: str) -> type[Any]:
        return cls._get(cls._data_sources, name, "data source")

    @classmethod
    def register_storage(cls, name: str) -> Callable[[type[Any]], type[Any]]:
        return cls._register(cls._storages, name)

    @classmethod
    def get_storage(cls, name: str) -> type[Any]:
        return cls._get(cls._storages, name, "storage")

    @classmethod
    def register_feature(cls, name: str) -> Callable[[type[Any]], type[Any]]:
        return cls._register(cls._features, name)

    @classmethod
    def get_feature(cls, name: str) -> type[Any]:
        return cls._get(cls._features, name, "feature")

    @classmethod
    def register_strategy(cls, name: str) -> Callable[[type[Any]], type[Any]]:
        return cls._register(cls._strategies, name)

    @classmethod
    def get_strategy(cls, name: str) -> type[Any]:
        return cls._get(cls._strategies, name, "strategy")

    @classmethod
    def register_engine(cls, name: str) -> Callable[[type[Any]], type[Any]]:
        return cls._register(cls._engines, name)

    @classmethod
    def get_engine(cls, name: str) -> type[Any]:
        return cls._get(cls._engines, name, "engine")

    @classmethod
    def register_broker(cls, name: str) -> Callable[[type[Any]], type[Any]]:
        return cls._register(cls._brokers, name)

    @classmethod
    def get_broker(cls, name: str) -> type[Any]:
        return cls._get(cls._brokers, name, "broker")

    @classmethod
    def register_fill_model(cls, name: str) -> Callable[[type[Any]], type[Any]]:
        return cls._register(cls._fill_models, name)

    @classmethod
    def get_fill_model(cls, name: str) -> type[Any]:
        return cls._get(cls._fill_models, name, "fill model")

    @classmethod
    def register_slippage_model(cls, name: str) -> Callable[[type[Any]], type[Any]]:
        return cls._register(cls._slippage_models, name)

    @classmethod
    def get_slippage_model(cls, name: str) -> type[Any]:
        return cls._get(cls._slippage_models, name, "slippage model")

    @classmethod
    def register_commission_model(cls, name: str) -> Callable[[type[Any]], type[Any]]:
        return cls._register(cls._commission_models, name)

    @classmethod
    def get_commission_model(cls, name: str) -> type[Any]:
        return cls._get(cls._commission_models, name, "commission model")

    @classmethod
    def register_calendar(cls, name: str) -> Callable[[type[Any]], type[Any]]:
        return cls._register(cls._calendars, name)

    @classmethod
    def get_calendar(cls, name: str) -> type[Any]:
        return cls._get(cls._calendars, name, "calendar")

    @classmethod
    def register_factor_trainer(cls, name: str) -> Callable[[Any], Any]:
        return cls._register(cls._factor_trainers, name)

    @classmethod
    def get_factor_trainer(cls, name: str) -> Callable[..., Any]:
        return cls._get(cls._factor_trainers, name, "factor trainer")

    @classmethod
    def register_portfolio_constructor(cls, name: str) -> Callable[[Any], Any]:
        return cls._register(cls._portfolio_constructors, name)

    @classmethod
    def get_portfolio_constructor(cls, name: str) -> Callable[..., Any]:
        return cls._get(cls._portfolio_constructors, name, "portfolio constructor")

    @classmethod
    def register_transform(cls, name: str) -> Callable[[Any], Any]:
        return cls._register(cls._transforms, name)

    @classmethod
    def get_transform(cls, name: str) -> Callable[..., Any]:
        return cls._get(cls._transforms, name, "transform")

    @classmethod
    def register_signal_algorithm(cls, name: str) -> Callable[[Any], Any]:
        return cls._register(cls._signal_algorithms, name)

    @classmethod
    def get_signal_algorithm(cls, name: str) -> Callable[..., Any]:
        return cls._get(cls._signal_algorithms, name, "signal algorithm")

    @classmethod
    def register_spread_model(cls, name: str) -> Callable[[Any], Any]:
        return cls._register(cls._spread_models, name)

    @classmethod
    def get_spread_model(cls, name: str) -> Callable[..., Any]:
        return cls._get(cls._spread_models, name, "spread model")

    @classmethod
    def register_signal_rule(cls, name: str) -> Callable[[Any], Any]:
        return cls._register(cls._signal_rules, name)

    @classmethod
    def get_signal_rule(cls, name: str) -> Callable[..., Any]:
        return cls._get(cls._signal_rules, name, "signal rule")

    @classmethod
    def register_model(cls, name: str) -> Callable[[Any], Any]:
        return cls._register(cls._models, name)

    @classmethod
    def get_model(cls, name: str) -> Callable[..., Any]:
        return cls._get(cls._models, name, "model")
