"""High-level autoresearch workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from agentic_quant_researcher.schemas.autoresearch import (
    AutoresearchContext,
    AutoresearchError,
    canonical_research_action,
    load_context,
)
from agentic_quant_researcher.tools.criteria import (
    best_kept_value,
    evaluate_criteria,
    is_improvement,
    loop_state,
    select_best_value,
)
from agentic_quant_researcher.tools.ledger import (
    append_event,
    append_result_row,
    init_ledgers,
    read_ledger,
    read_run_metadata,
    result_row_exists,
    utc_now,
)
from agentic_quant_researcher.tools.metrics import extract_metrics, metric_float
from agentic_quant_researcher.tools.runner import resolve_run_dir, run_manifest_command
from agentic_quant_researcher.tools.scope import (
    check_edit_scope,
    edit_root_paths,
    new_dirty_paths_since_snapshot,
    out_of_scope_paths,
    preflight_status,
    read_worktree_snapshot,
    write_worktree_snapshot,
)
from agentic_quant_researcher.tools.summary import (
    build_next_step_analysis,
    error_tail,
    write_next_steps,
    write_summary,
    write_trial_summary,
)
from agentic_quant_researcher.utils.git import (
    changed_files_for_commit,
    git_changed_paths,
    git_output,
)
from agentic_quant_researcher.utils.paths import rel_path


def load(args: argparse.Namespace) -> AutoresearchContext:
    return load_context(Path(args.manifest), getattr(args, "repo_root", None))


def preflight(args: argparse.Namespace) -> dict[str, object]:
    return preflight_status(load(args), strict=getattr(args, "strict", False))


def check_edits(args: argparse.Namespace) -> dict[str, object]:
    return check_edit_scope(
        load(args),
        edit_roots_only=getattr(args, "edit_roots_only", False),
        strict=getattr(args, "strict", False),
    )


def run_trial(args: argparse.Namespace) -> tuple[dict[str, object], int]:
    context = load(args)
    metadata, returncode = run_manifest_command(
        context,
        trial=args.trial,
        run_id=args.run_id,
        run_dir=getattr(args, "run_dir", None),
        timeout=getattr(args, "timeout", None),
    )
    run_dir = resolve_run_dir(context, args.run_id, getattr(args, "run_dir", None))
    write_trial_summary(
        run_dir,
        {
            "status": "crash" if returncode != 0 else "ran",
            "stop_reason": "timeout" if metadata.get("timed_out") else "record_pending",
            "criteria_hit": False,
            "hit_metric": "",
            "objective_metric": context.manifest.objective.metric,
            "objective_value": (metadata.get("metrics") or {}).get(
                context.manifest.objective.metric
            ),
            "best_value": None,
            "description": "Run completed; record step pending.",
        },
        metadata,
        error_tail(run_dir / "run.log"),
    )
    return metadata, returncode


def record(args: argparse.Namespace) -> dict[str, object]:
    context = load(args)
    init_ledgers(context)
    idea_family = canonical_research_action(args.idea_family, context.manifest)
    run_dir = resolve_run_dir(context, args.run_id, getattr(args, "run_dir", None))
    metadata = read_run_metadata(run_dir)
    metrics = dict(metadata.get("metrics") or {})
    if not metrics:
        metrics, source = extract_metrics(run_dir, context.manifest)
        metadata["metrics"] = metrics
        metadata["metric_source"] = source
    commit = (
        getattr(args, "commit", None)
        or git_output(context.repo_root, ["rev-parse", "--short", "HEAD"])
        or "unknown"
    )
    changed_files = changed_files_for_record(context, commit)
    dirty_files = git_changed_paths(context.repo_root)
    dirty_to_check = new_dirty_paths_since_snapshot(context, dirty_files)
    violations = out_of_scope_paths(context, changed_files | dirty_to_check, include_artifacts=True)
    if violations:
        raise AutoresearchError(
            "changes escape manifest scope: " + ", ".join(sorted(violations))
        )
    record_changed_files = changed_files | edit_root_paths(context, dirty_to_check)

    metric_name = context.manifest.objective.metric
    value = metric_float(metrics.get(metric_name))
    rows = read_ledger(context)
    previous_best = best_kept_value(rows, context.manifest)
    criteria_hit, hit_metrics = evaluate_criteria(metrics, context.manifest)
    crashed = bool(metadata.get("returncode", 0)) or value is None
    improved = value is not None and is_improvement(value, previous_best, context.manifest)
    requested_status = getattr(args, "status", "auto")
    if requested_status == "auto":
        status = "crash" if crashed else "keep" if criteria_hit or improved else "discard"
    else:
        status = requested_status
    best_value = previous_best
    if status == "keep" and value is not None:
        best_value = value if previous_best is None else select_best_value(
            previous_best,
            value,
            context.manifest,
        )
    recorded_count_after = len(rows) + 1
    stop_reason = "continue"
    if criteria_hit:
        stop_reason = "criteria_met"
    elif recorded_count_after >= context.manifest.quota.max_trials:
        stop_reason = "quota_exhausted"
    row = {
        "trial": args.trial,
        "run_id": args.run_id,
        "commit": commit,
        "status": status,
        "stop_reason": stop_reason,
        "objective_metric": metric_name,
        "objective_value": value,
        "best_value": best_value,
        "criteria_hit": str(criteria_hit).lower(),
        "hit_metric": ",".join(hit_metrics),
        "sharpe": metrics.get("sharpe"),
        "sortino": metrics.get("sortino"),
        "cagr": metrics.get("cagr"),
        "max_drawdown": metrics.get("max_drawdown"),
        "win_rate": metrics.get("win_rate"),
        "signal_rows": metrics.get("signal_rows"),
        "featured_rows": metrics.get("featured_rows"),
        "predictor_count": metrics.get("predictor_count"),
        "wall_seconds": metadata.get("wall_seconds"),
        "idea_family": idea_family,
        "changed_files": ";".join(sorted(record_changed_files)),
        "artifact_dir": rel_path(run_dir, context.repo_root),
        "description": args.description,
    }
    append_result_row(context, row)
    append_event(
        context,
        {
            "event": "trial_recorded",
            "timestamp": utc_now(),
            "row": row,
            "metrics": metrics,
            "metadata": metadata,
        },
    )
    write_trial_summary(run_dir, row, metadata, error_tail(run_dir / "run.log"))
    write_summary(context)
    return {
        "run_id": args.run_id,
        "status": status,
        "stop_reason": stop_reason,
        "criteria_hit": criteria_hit,
        "objective_value": value,
        "best_value": best_value,
    }


def baseline(args: argparse.Namespace) -> dict[str, object]:
    context = load(args)
    status = preflight_status(context)
    if not status["ready"]:
        raise AutoresearchError(
            "preflight failed: "
            + "; ".join(status["blocked_by"])
            + f". Next action: {status['next_action']}"
        )
    init_ledgers(context)
    snapshot = write_worktree_snapshot(context)
    write_summary(context)
    run_dir = resolve_run_dir(context, args.run_id, None)
    metadata_path = run_dir / "metrics.json"
    reused_existing_run = metadata_path.exists() and not getattr(args, "rerun", False)
    if reused_existing_run:
        metadata = read_run_metadata(run_dir)
        run_result = {
            "run_id": args.run_id,
            "returncode": metadata.get("returncode"),
            "metrics": metadata.get("metrics", {}),
            "reused": True,
        }
    else:
        metadata, returncode = run_manifest_command(
            context,
            trial=args.trial,
            run_id=args.run_id,
            timeout=getattr(args, "timeout", None),
        )
        if returncode != 0:
            return {"baseline": {"run_id": args.run_id, "returncode": returncode}}
        run_result = {
            "run_id": args.run_id,
            "returncode": metadata.get("returncode"),
            "metrics": metadata.get("metrics", {}),
            "reused": False,
        }
    if result_row_exists(context, args.run_id):
        record_result = {"run_id": args.run_id, "recorded": "already_present"}
    else:
        record_args = argparse.Namespace(
            manifest=args.manifest,
            repo_root=getattr(args, "repo_root", None),
            trial=args.trial,
            run_id=args.run_id,
            run_dir=None,
            idea_family=args.idea_family,
            description=args.description,
            status="auto",
            commit=None,
        )
        record_result = record(record_args)
    write_summary(context)
    next_step = build_next_step_analysis(context, run_id=args.run_id)
    write_next_steps(context, next_step)
    return {
        "artifact_root": str(context.artifact_root),
        "baseline": run_result,
        "next_steps": str(context.next_steps_path),
        "record": record_result,
        "snapshot": {
            "path": str(context.snapshot_path),
            "out_of_scope_dirty_count": len(snapshot["out_of_scope_dirty_files"]),
        },
        "loop": loop_state(read_ledger(context), context.manifest),
    }


def next_step(args: argparse.Namespace) -> dict[str, Any]:
    context = load(args)
    analysis = build_next_step_analysis(context, run_id=getattr(args, "run_id", None))
    write_next_steps(context, analysis)
    return analysis


def status(args: argparse.Namespace) -> dict[str, Any]:
    context = load(args)
    return loop_state(read_ledger(context), context.manifest)


def summarize(args: argparse.Namespace) -> dict[str, str]:
    context = load(args)
    write_summary(context)
    return {"summary": str(context.summary_path)}


def changed_files_for_record(context: AutoresearchContext, commit: str) -> set[str]:
    if not commit or commit == "unknown":
        return set()
    commit_full = git_output(context.repo_root, ["rev-parse", "--verify", commit])
    if not commit_full:
        return set()
    snapshot = read_worktree_snapshot(context)
    base_commit = str((snapshot or {}).get("base_commit") or "")
    if base_commit:
        base_full = git_output(context.repo_root, ["rev-parse", "--verify", base_commit])
        if base_full == commit_full:
            return set()
        if base_full:
            output = git_output(
                context.repo_root,
                ["diff", "--name-only", f"{base_full}..{commit_full}"],
            )
            return {line.strip() for line in output.splitlines() if line.strip()}
    return changed_files_for_commit(context.repo_root, commit_full)
