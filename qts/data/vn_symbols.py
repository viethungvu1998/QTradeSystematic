"""Helpers for normalizing Vietnam-specific symbol conventions."""

from __future__ import annotations

import re


def strip_vn_prefix(symbol: str) -> str:
    for prefix in ("VNF:", "VNW:", "VN:"):
        if symbol.startswith(prefix):
            return symbol[len(prefix):]
    return symbol


def is_vn_warrant_code(symbol: str) -> bool:
    return re.fullmatch(r"C[A-Z0-9]{2,}\d{4}", symbol.upper()) is not None


def to_vn_warrant_request(symbol: str) -> str:
    raw = strip_vn_prefix(symbol).upper()
    if is_vn_warrant_code(raw):
        return f"VNW:{raw}"
    return f"VNW:{raw}"


def to_dnse_futures_alias(symbol: str) -> str:
    raw = strip_vn_prefix(symbol).upper()
    if re.fullmatch(r"VN30F[1-4][MQ]", raw):
        return raw
    if re.fullmatch(r"VN30F\d{4}", raw) or raw.startswith("41"):
        return "VN30F1M"
    return raw


def to_vn_futures_symbol(symbol: str) -> str:
    return f"VNF:{to_dnse_futures_alias(symbol)}"
