"""Stat-arb strategy."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial

from qts.core.registry import Registry

from .base import BaseStatArbStrategy


def _resolve_named_section(raw: object, section_name: str) -> dict:
    """Normalise a YAML sub-section into {'name': str, 'params': dict}."""
    if isinstance(raw, str):
        return {"name": raw, "params": {}}
    if isinstance(raw, dict):
        return {"name": str(raw["name"]), "params": dict(raw.get("params", {}))}
    raise ValueError(f"Cannot resolve {section_name!r} section from {raw!r}")


@Registry.register_strategy("stat_arb")
class StatArbStrategy(BaseStatArbStrategy):
    """Universe-level mean-reversion spread strategy."""

    @classmethod
    def from_config_params(
        cls,
        params: Mapping[str, object],
        *,
        portfolio_func: Callable | None = None,
    ) -> StatArbStrategy:
        payload = dict(params)

        spread_raw = payload.pop("spread_model", {"name": "ols"})
        spread_cfg = _resolve_named_section(spread_raw, "spread_model")
        spread_fn = partial(
            Registry.get_spread_model(spread_cfg["name"]),
            **spread_cfg["params"],
        )

        rule_raw = payload.pop("signal_rule", {"name": "zscore_threshold"})
        rule_cfg = _resolve_named_section(rule_raw, "signal_rule")
        signal_fn = partial(
            Registry.get_signal_rule(rule_cfg["name"]),
            **rule_cfg["params"],
        )

        if portfolio_func is None and "portfolio" in payload:
            portfolio_raw = _resolve_named_section(payload.pop("portfolio"), "portfolio")
            port_fn = Registry.get_portfolio_constructor(portfolio_raw["name"])
            portfolio_func = partial(port_fn, **portfolio_raw["params"])
        else:
            payload.pop("portfolio", None)

        return cls(
            spread_fn=spread_fn,
            signal_fn=signal_fn,
            portfolio_func=portfolio_func,
            **payload,
        )
