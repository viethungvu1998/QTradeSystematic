"""Small non-interactive git helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path


def git_output(repo_root: Path, args: list[str]) -> str:
    """Run git and return stdout, or an empty string if git cannot answer."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def git_changed_paths(repo_root: Path) -> set[str]:
    """Return dirty, staged, and untracked paths relative to ``repo_root``."""
    paths: set[str] = set()
    for args in (
        ["diff", "--name-only"],
        ["diff", "--cached", "--name-only"],
        ["ls-files", "--others", "--exclude-standard"],
    ):
        output = git_output(repo_root, args)
        if output:
            paths.update(line.strip() for line in output.splitlines() if line.strip())
    return paths


def changed_files_for_commit(repo_root: Path, commit: str) -> set[str]:
    """Return files changed by a commit-ish."""
    output = git_output(repo_root, ["show", "--pretty=format:", "--name-only", commit])
    return {line.strip() for line in output.splitlines() if line.strip()}
