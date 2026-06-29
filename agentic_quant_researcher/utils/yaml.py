"""YAML loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agentic_quant_researcher.schemas.autoresearch import AutoresearchError


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML file and require a mapping at the document root."""
    if not path.exists():
        raise AutoresearchError(f"YAML file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise AutoresearchError(f"{path} must contain a YAML mapping")
    return data
