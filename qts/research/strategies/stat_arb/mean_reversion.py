"""Universe-level statistical arbitrage strategy."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial

from qts.core.registry import Registry
from qts.research.strategies._config import resolve_named_section

from .base import BaseStatArbStrategy


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
        spread_cfg = resolve_named_section(spread_raw, "spread_model")
        spread_fn = partial(
            Registry.get_spread_model(str(spread_cfg["name"])),
            **dict(spread_cfg["params"]),
        )

        rule_raw = payload.pop("signal_rule", {"name": "zscore_threshold"})
        rule_cfg = resolve_named_section(rule_raw, "signal_rule")
        signal_fn = partial(
            Registry.get_signal_rule(str(rule_cfg["name"])),
            **dict(rule_cfg["params"]),
        )

        if portfolio_func is None and "portfolio" in payload:
            portfolio_raw = resolve_named_section(payload.pop("portfolio"), "portfolio")
            port_fn = Registry.get_portfolio_constructor(str(portfolio_raw["name"]))
            portfolio_func = partial(port_fn, **dict(portfolio_raw["params"]))
        else:
            payload.pop("portfolio", None)

        return cls(
            spread_fn=spread_fn,
            signal_fn=signal_fn,
            portfolio_func=portfolio_func,
            **payload,
        )


__all__ = ["StatArbStrategy"]
