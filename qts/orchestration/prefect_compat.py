"""Prefect compatibility helpers for environments without Prefect installed."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

try:  # pragma: no cover - exercised when Prefect is installed
    from prefect import flow, task
    try:
        from prefect.runner import serve
    except ImportError:
        from prefect.flows import serve
except ModuleNotFoundError:  # pragma: no cover - simple compatibility path
    class _Deployment:
        def __init__(self, name: str, **kwargs) -> None:
            self.name = name
            self.kwargs = kwargs

    def task(*dargs, **dkwargs):
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            fn.fn = fn
            return fn

        if dargs and callable(dargs[0]) and len(dargs) == 1 and not dkwargs:
            return decorator(dargs[0])
        return decorator

    def flow(*dargs, **dkwargs):
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            def to_deployment(name: str, **kwargs) -> _Deployment:
                return _Deployment(name, **kwargs)

            fn.to_deployment = to_deployment
            fn.fn = fn
            return fn

        if dargs and callable(dargs[0]) and len(dargs) == 1 and not dkwargs:
            return decorator(dargs[0])
        return decorator

    def serve(*deployments):
        return deployments
