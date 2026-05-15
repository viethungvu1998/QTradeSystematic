"""Instrument models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AssetType(StrEnum):
    """Supported asset classes."""

    STOCK = "stock"
    CRYPTO = "crypto"

    @classmethod
    def from_symbol(cls, symbol: str) -> "AssetType":
        return cls.CRYPTO if "/" in symbol else cls.STOCK


@dataclass(frozen=True, slots=True)
class Instrument:
    """Tradable instrument identifier."""

    symbol: str
    asset_type: AssetType
    exchange: str
    currency: str
