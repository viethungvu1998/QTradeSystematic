"""Instrument models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AssetType(StrEnum):
    """Supported asset classes."""

    STOCK = "stock"
    VN_STOCK = "vn_stock"
    VN_WARRANT = "vn_warrant"
    VN_FUTURES = "vn_futures"
    COMMODITY = "commodity"
    CRYPTO = "crypto"
    CRYPTO_FUTURES = "crypto_futures"

    @classmethod
    def from_symbol(cls, symbol: str) -> AssetType:
        if symbol.startswith("PERP:"):
            return cls.CRYPTO_FUTURES
        if symbol.startswith("VNF:"):
            return cls.VN_FUTURES
        if symbol.startswith("VNW:"):
            return cls.VN_WARRANT
        if "/" in symbol:
            return cls.CRYPTO
        if symbol.startswith("VN:"):
            return cls.VN_STOCK
        if symbol.startswith("CMX:"):
            return cls.COMMODITY
        return cls.STOCK


@dataclass(frozen=True, slots=True)
class Instrument:
    """Tradable instrument identifier."""

    symbol: str
    asset_type: AssetType
    exchange: str
    currency: str
