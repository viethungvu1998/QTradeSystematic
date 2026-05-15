"""Plugin registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from qts.core.errors import RegistryError


RegistryDict = dict[str, type[Any]]


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

    @classmethod
    def _register(cls, registry: RegistryDict, name: str) -> Callable[[type[Any]], type[Any]]:
        def decorator(target: type[Any]) -> type[Any]:
            registry[name] = target
            return target

        return decorator

    @classmethod
    def _get(cls, registry: RegistryDict, name: str, kind: str) -> type[Any]:
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
