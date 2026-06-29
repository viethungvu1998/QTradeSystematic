"""Path helpers for autoresearch runs."""

from __future__ import annotations

from pathlib import Path

from agentic_quant_researcher.schemas.autoresearch import AutoresearchError


def find_repo_root(start: Path) -> Path:
    """Find the QTradeSystematic repository root from ``start``."""
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "qts").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    raise AutoresearchError(f"could not find QTradeSystematic root from {start}")


def resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    """Resolve ``value`` relative to ``repo_root`` unless it is already absolute."""
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def rel_path(path: Path, repo_root: Path) -> str:
    """Return a stable POSIX relative path where possible."""
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def is_same_or_child(path: str, root: str) -> bool:
    """Return whether ``path`` is ``root`` or nested below it."""
    clean_path = path.strip("/").replace("\\", "/")
    clean_root = root.strip("/").replace("\\", "/")
    return clean_path == clean_root or clean_path.startswith(f"{clean_root}/")
