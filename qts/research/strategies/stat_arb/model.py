"""Stat-arb strategy."""

from __future__ import annotations

from qts.core.registry import Registry

from .base import BaseStatArbStrategy


@Registry.register_strategy("stat_arb")
class StatArbStrategy(BaseStatArbStrategy):
    """Universe-level mean-reversion spread strategy."""
