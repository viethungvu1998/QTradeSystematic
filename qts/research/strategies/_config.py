"""Shared strategy config parsing helpers."""

from __future__ import annotations


def resolve_named_section(raw: object, section_name: str) -> dict[str, object]:
    if isinstance(raw, str):
        return {"name": raw, "params": {}}
    if isinstance(raw, dict):
        return {"name": str(raw["name"]), "params": dict(raw.get("params", {}))}
    raise ValueError(f"Cannot resolve {section_name!r} section from {raw!r}")
