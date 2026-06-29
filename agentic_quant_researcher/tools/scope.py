"""Worktree scope checks for manifest-scoped autoresearch."""

from __future__ import annotations

import json

from agentic_quant_researcher.schemas.autoresearch import (
    AutoresearchContext,
    AutoresearchError,
)
from agentic_quant_researcher.tools.ledger import append_event, utc_now
from agentic_quant_researcher.utils.git import git_changed_paths, git_output
from agentic_quant_researcher.utils.paths import is_same_or_child, rel_path


def preflight_status(context: AutoresearchContext, *, strict: bool = False) -> dict[str, object]:
    changed = git_changed_paths(context.repo_root)
    snapshot = read_worktree_snapshot(context)
    if strict or snapshot:
        checked = changed if strict else new_dirty_paths_since_snapshot(context, changed)
    else:
        checked = set()
    out_of_scope = out_of_scope_paths(context, changed, include_artifacts=True)
    new_out_of_scope = out_of_scope_paths(context, checked, include_artifacts=True)
    ready = not new_out_of_scope
    if strict and out_of_scope:
        ready = False
    blocked_by: list[str] = []
    if not ready:
        blocked_by.append("dirty files outside manifest scope")
    if not ready:
        next_action = "clean, stash, or move the new out-of-scope changes before recording"
    elif snapshot:
        next_action = "run baseline or the next trial"
    elif out_of_scope:
        next_action = "run init or baseline to snapshot current dirty paths before recording"
    else:
        next_action = "run baseline"
    return {
        "ready": ready,
        "blocked_by": blocked_by,
        "next_action": next_action,
        "artifact_root": rel_path(context.artifact_root, context.repo_root),
        "run_root": rel_path(context.run_root, context.repo_root),
        "allowed_edit_roots": sorted(context.allowed_edit_roots),
        "changed_file_count": len(changed),
        "out_of_scope_dirty_count": len(out_of_scope),
        "new_out_of_scope_dirty_count": len(new_out_of_scope),
        "new_out_of_scope_dirty_files": sorted(new_out_of_scope),
        "snapshot_exists": bool(snapshot),
        "snapshot_path": rel_path(context.snapshot_path, context.repo_root),
    }


def check_edit_scope(
    context: AutoresearchContext,
    *,
    edit_roots_only: bool = False,
    strict: bool = False,
) -> dict[str, object]:
    changed = git_changed_paths(context.repo_root)
    checked = changed if strict else new_dirty_paths_since_snapshot(context, changed)
    violations = out_of_scope_paths(
        context,
        checked,
        include_artifacts=not edit_roots_only,
    )
    if violations:
        raise AutoresearchError(
            "changes escape manifest scope: " + ", ".join(sorted(violations))
        )
    return {
        "changed_files": sorted(changed),
        "ignored_snapshot_files": sorted(changed - checked),
        "scope": "ok",
    }


def write_worktree_snapshot(context: AutoresearchContext) -> dict[str, object]:
    changed = git_changed_paths(context.repo_root)
    out_of_scope = out_of_scope_paths(context, changed, include_artifacts=True)
    snapshot = {
        "created_at": utc_now(),
        "base_commit": git_output(context.repo_root, ["rev-parse", "HEAD"]),
        "dirty_files": sorted(changed),
        "out_of_scope_dirty_files": sorted(out_of_scope),
    }
    context.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    context.snapshot_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    append_event(
        context,
        {
            "event": "worktree_snapshot",
            "timestamp": utc_now(),
            "snapshot": snapshot,
        },
    )
    return snapshot


def read_worktree_snapshot(context: AutoresearchContext) -> dict[str, object] | None:
    if not context.snapshot_path.exists():
        return None
    try:
        data = json.loads(context.snapshot_path.read_text())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def new_dirty_paths_since_snapshot(
    context: AutoresearchContext,
    changed: set[str] | None = None,
) -> set[str]:
    changed = set(changed if changed is not None else git_changed_paths(context.repo_root))
    snapshot = read_worktree_snapshot(context)
    if not snapshot:
        return changed
    return changed - set(snapshot.get("dirty_files", []))


def edit_root_paths(context: AutoresearchContext, paths: set[str]) -> set[str]:
    return {
        path
        for path in paths
        if any(is_same_or_child(path, root) for root in context.allowed_edit_roots)
    }


def out_of_scope_paths(
    context: AutoresearchContext,
    paths: set[str],
    *,
    include_artifacts: bool,
) -> set[str]:
    allowed = set(context.allowed_edit_roots)
    if include_artifacts:
        allowed.add(context.artifact_root_rel)
        allowed.add(context.run_root_rel)
    return {path for path in paths if not any(is_same_or_child(path, root) for root in allowed)}
