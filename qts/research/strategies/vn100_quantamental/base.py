"""Family base for VN100 quantamental strategies."""

from __future__ import annotations

from qts.research.strategies.factor.base import BaseFactorStrategy


class BaseVN100QuantamentalStrategy(BaseFactorStrategy):
    """Shared interface for VN100 quantamental strategies."""


__all__ = ["BaseVN100QuantamentalStrategy"]
