"""Centralised on-disk paths."""

from __future__ import annotations

import os
from pathlib import Path


def qts_root() -> Path:
    return Path(os.environ.get("QTS_ROOT", Path.home() / ".qts"))


def database_path() -> Path:
    return qts_root() / "database" / "qts.duckdb"


def cache_dir() -> Path:
    return qts_root() / "cache"


def bundle_dir() -> Path:
    return qts_root() / "bundles"


def backtest_exports_dir() -> Path:
    path = qts_root() / "exports" / "backtest"
    path.mkdir(parents=True, exist_ok=True)
    return path


def live_portfolio_dir() -> Path:
    path = qts_root() / "exports" / "live"
    path.mkdir(parents=True, exist_ok=True)
    return path
