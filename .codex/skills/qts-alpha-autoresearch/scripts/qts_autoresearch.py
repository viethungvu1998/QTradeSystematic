#!/usr/bin/env python3
"""Deterministic helpers for QTS alpha autoresearch runs."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import operator
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

RESULT_COLUMNS = [
    "trial",
    "run_id",
    "commit",
    "status",
    "stop_reason",
    "objective_metric",
    "objective_value",
    "best_value",
    "criteria_hit",
    "hit_metric",
    "sharpe",
    "sortino",
    "cagr",
    "max_drawdown",
    "win_rate",
    "signal_rows",
    "featured_rows",
    "predictor_count",
    "wall_seconds",
    "idea_family",
    "changed_files",
    "artifact_dir",
    "description",
]

NUMERIC_LEDGER_COLUMNS = {
    "objective_value",
    "best_value",
    "sharpe",
    "sortino",
    "cagr",
    "max_drawdown",
    "win_rate",
    "signal_rows",
    "featured_rows",
    "predictor_count",
    "wall_seconds",
}

OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
}

LOG_METRIC_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)
TARGET_PREFIXES = ("forward_return", "future_", "target_")
WORKTREE_SNAPSHOT_FILE = "worktree_snapshot.json"
NEXT_STEPS_FILE = "next_steps.md"
PROGRESS_PNG_FILE = "progress.png"
PROGRESS_JSON_FILE = "progress.json"
TERMINAL_STOP_REASONS = {"criteria_met", "quota_exhausted"}
SKILL_ROOT = Path(__file__).resolve().parents[1]
SKILL_FILE = "SKILL.md"
DEFAULT_ACTION_ALIASES = {
    "parameter_tuning": {"parameter_tuning", "parameters", "tweak_parameters", "model_params"},
    "hyperparameter_optimization": {
        "hyperparameter_optimization",
        "hyperparams",
        "hyperparam_opt",
        "hyperopt",
    },
    "feature_engineering": {
        "feature_engineering",
        "features",
        "non_leaking_features",
    },
    "strategy_model_update": {
        "strategy_model_update",
        "strategy_model",
        "model_update",
        "trainer",
    },
}


class AutoresearchError(RuntimeError):
    """Raised for user-correctable autoresearch setup errors."""


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except AutoresearchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QTS alpha autoresearch helper")
    parser.add_argument("--repo-root", type=Path, help="QTradeSystematic project root override")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser(
        "preflight",
        help="Report whether the manifest can run and record cleanly",
    )
    preflight_parser.add_argument("manifest", type=Path)
    preflight_parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat all out-of-scope dirty files as blockers, even before init snapshots them.",
    )
    preflight_parser.set_defaults(func=cmd_preflight)

    init_parser = subparsers.add_parser("init", help="Initialize ledgers and optionally branch")
    init_parser.add_argument("manifest", type=Path)
    init_parser.add_argument("--no-branch", action="store_true")
    init_parser.set_defaults(func=cmd_init)

    check_parser = subparsers.add_parser("check-edits", help="Validate dirty/staged edit scope")
    check_parser.add_argument("manifest", type=Path)
    check_parser.add_argument(
        "--edit-roots-only",
        action="store_true",
        help="Reject artifact-root changes too; useful for pre-commit hygiene checks.",
    )
    check_parser.add_argument(
        "--strict",
        action="store_true",
        help="Do not ignore paths captured by the init worktree snapshot.",
    )
    check_parser.set_defaults(func=cmd_check_edits)

    run_parser = subparsers.add_parser("run", help="Run a manifest command into run.log")
    run_parser.add_argument("manifest", type=Path)
    run_parser.add_argument("--trial", type=int, required=True)
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--run-dir", type=Path)
    run_parser.add_argument("--timeout", type=float)
    run_parser.set_defaults(func=cmd_run)

    record_parser = subparsers.add_parser("record", help="Record a trial in TSV and JSONL")
    record_parser.add_argument("manifest", type=Path)
    record_parser.add_argument("--trial", type=int, required=True)
    record_parser.add_argument("--run-id", required=True)
    record_parser.add_argument("--run-dir", type=Path)
    record_parser.add_argument("--idea-family", default="unspecified")
    record_parser.add_argument("--description", default="")
    record_parser.add_argument(
        "--status",
        choices=["auto", "keep", "discard", "crash"],
        default="auto",
    )
    record_parser.add_argument("--commit")
    record_parser.add_argument("--apply-git-reset", action="store_true")
    record_parser.set_defaults(func=cmd_record)

    baseline_parser = subparsers.add_parser(
        "baseline",
        help="Run preflight, init, baseline run, record, summarize, and next-step analysis",
    )
    baseline_parser.add_argument("manifest", type=Path)
    baseline_parser.add_argument("--trial", type=int, default=0)
    baseline_parser.add_argument("--run-id", default="000_baseline")
    baseline_parser.add_argument("--idea-family", default="parameter_tuning")
    baseline_parser.add_argument("--description", default="baseline run before strategy edits")
    baseline_parser.add_argument("--timeout", type=float)
    baseline_parser.add_argument("--no-branch", action="store_true")
    baseline_parser.add_argument("--rerun", action="store_true")
    baseline_parser.set_defaults(func=cmd_baseline)

    next_parser = subparsers.add_parser(
        "next",
        help="Analyze latest metrics and write the next actionable experiment",
    )
    next_parser.add_argument("manifest", type=Path)
    next_parser.add_argument("--run-id")
    next_parser.set_defaults(func=cmd_next)

    status_parser = subparsers.add_parser(
        "status",
        help="Report whether the autoresearch loop may stop or must continue",
    )
    status_parser.add_argument("manifest", type=Path)
    status_parser.set_defaults(func=cmd_status)

    progress_parser = subparsers.add_parser(
        "progress",
        help="Render Karpathy-style progress.png and progress.json under artifact_root",
    )
    progress_parser.add_argument("manifest", type=Path)
    progress_parser.set_defaults(func=cmd_progress)

    loop_parser = subparsers.add_parser(
        "loop",
        help="Run autonomous experiments until criteria/quota or --max-steps is reached",
    )
    loop_parser.add_argument("manifest", type=Path)
    loop_parser.add_argument("--max-steps", type=int, required=True)
    loop_parser.add_argument("--timeout", type=float)
    loop_parser.add_argument("--no-branch", action="store_true")
    loop_parser.add_argument("--rerun-baseline", action="store_true")
    loop_parser.set_defaults(func=cmd_loop)

    summarize_parser = subparsers.add_parser("summarize", help="Rebuild summary.md")
    summarize_parser.add_argument("manifest", type=Path)
    summarize_parser.set_defaults(func=cmd_summarize)
    return parser


def cmd_preflight(args: argparse.Namespace) -> int:
    context = load_context(args.manifest, args.repo_root)
    status = preflight_status(context, strict=args.strict)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if status["ready"] else 2


def cmd_init(args: argparse.Namespace) -> int:
    context = load_context(args.manifest, args.repo_root)
    init_ledgers(context)
    snapshot = write_worktree_snapshot(context)
    if not args.no_branch:
        create_branch_if_needed(context)
    write_summary(context)
    write_autonomous_prompt(context)
    print(
        json.dumps(
            {
                "artifact_root": str(context["artifact_root"]),
                "worktree_snapshot": str(context["snapshot_path"]),
                "out_of_scope_dirty_files": snapshot["out_of_scope_dirty_files"],
            },
            sort_keys=True,
        )
    )
    return 0


def cmd_check_edits(args: argparse.Namespace) -> int:
    context = load_context(args.manifest, args.repo_root)
    changed = git_changed_paths(context["repo_root"])
    checked = changed if args.strict else new_dirty_paths_since_snapshot(context, changed)
    violations = out_of_scope_paths(
        context,
        checked,
        include_artifacts=not args.edit_roots_only,
    )
    if violations:
        raise AutoresearchError(
            "changes escape manifest scope: " + ", ".join(sorted(violations))
        )
    print(
        json.dumps(
            {
                "changed_files": sorted(changed),
                "ignored_snapshot_files": sorted(changed - checked),
                "scope": "ok",
            },
            sort_keys=True,
        )
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    context = load_context(args.manifest, args.repo_root)
    run_dir = resolve_run_dir(context, args.run_id, args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    argv = render_command(context, args.run_id, run_dir)
    log_path = run_dir / "run.log"
    command_path = run_dir / "command.json"
    started_at = utc_now()
    start = time.monotonic()
    returncode = 0
    timed_out = False

    command_path.write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "trial": args.trial,
                "argv": argv,
                "started_at": started_at,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with log_path.open("w") as log_file:
        try:
            completed = subprocess.run(
                argv,
                cwd=context["repo_root"],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.timeout,
                check=False,
            )
            returncode = int(completed.returncode)
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = 124
            log_file.write(f"\nTIMEOUT: exceeded {args.timeout} seconds\n")

    wall_seconds = time.monotonic() - start
    metrics, source = extract_metrics(run_dir, context["manifest"])
    metadata = {
        "run_id": args.run_id,
        "trial": args.trial,
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_seconds": wall_seconds,
        "metrics": metrics,
        "metric_source": source,
        "artifact_dir": str(run_dir),
        "finished_at": utc_now(),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    write_trial_summary(
        run_dir,
        {
            "status": "crash" if returncode != 0 else "ran",
            "stop_reason": "timeout" if timed_out else "record_pending",
            "criteria_hit": False,
            "hit_metric": "",
            "objective_metric": objective_metric(context["manifest"]),
            "objective_value": metrics.get(objective_metric(context["manifest"])),
            "best_value": None,
            "description": "Run completed; record step pending.",
        },
        metadata,
        error_tail(log_path),
    )
    print(json.dumps({"run_id": args.run_id, "returncode": returncode, "metrics": metrics}))
    return 0 if returncode == 0 else 1


def cmd_record(args: argparse.Namespace) -> int:
    context = load_context(args.manifest, args.repo_root)
    init_ledgers(context)
    idea_family = canonical_research_action(args.idea_family, context["manifest"])
    run_dir = resolve_run_dir(context, args.run_id, args.run_dir)
    metadata = read_run_metadata(run_dir)
    metrics = dict(metadata.get("metrics") or {})
    if not metrics:
        metrics, source = extract_metrics(run_dir, context["manifest"])
        metadata["metrics"] = metrics
        metadata["metric_source"] = source
    commit = (
        args.commit
        or git_output(context["repo_root"], ["rev-parse", "--short", "HEAD"])
        or "unknown"
    )
    changed_files = changed_files_for_record(context, commit)
    dirty_files = git_changed_paths(context["repo_root"])
    dirty_to_check = new_dirty_paths_since_snapshot(context, dirty_files)
    violations = out_of_scope_paths(context, changed_files | dirty_to_check, include_artifacts=True)
    if violations:
        raise AutoresearchError(
            "changes escape manifest scope: " + ", ".join(sorted(violations))
        )
    record_changed_files = changed_files | edit_root_paths(context, dirty_to_check)

    metric_name = objective_metric(context["manifest"])
    value = metric_float(metrics.get(metric_name))
    previous_best = best_kept_value(context)
    previous_best_commit = best_kept_commit(context)
    criteria_hit, hit_metrics = evaluate_criteria(metrics, context["manifest"].get("criteria", {}))
    crashed = bool(metadata.get("returncode", 0)) or value is None
    improved = value is not None and is_improvement(
        value,
        previous_best,
        context["manifest"].get("objective", {}),
    )
    if args.status == "auto":
        status = "crash" if crashed else "keep" if criteria_hit or improved else "discard"
    else:
        status = args.status

    best_value = previous_best
    if status == "keep" and value is not None:
        best_value = value if previous_best is None else select_best_value(
            previous_best,
            value,
            context["manifest"].get("objective", {}),
        )

    recorded_count_after = len(read_ledger(context)) + 1
    stop_reason = "continue"
    if criteria_hit:
        stop_reason = "criteria_met"
    elif recorded_count_after >= max_trials(context["manifest"]):
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
        "artifact_dir": rel_path(run_dir, context["repo_root"]),
        "description": clean_cell(args.description),
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
    write_progress_artifacts(context)

    if args.apply_git_reset and status in {"discard", "crash"} and previous_best_commit:
        subprocess.run(
            ["git", "reset", "--hard", previous_best_commit],
            cwd=context["repo_root"],
            check=True,
        )

    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "status": status,
                "stop_reason": stop_reason,
                "criteria_hit": criteria_hit,
                "objective_value": value,
                "best_value": best_value,
            },
            sort_keys=True,
        )
    )
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    context = load_context(args.manifest, args.repo_root)
    status = preflight_status(context)
    if not status["ready"]:
        raise AutoresearchError(
            "preflight failed: "
            + "; ".join(status["blocked_by"])
            + f". Next action: {status['next_action']}"
        )

    init_ledgers(context)
    snapshot = write_worktree_snapshot(context)
    if not args.no_branch:
        create_branch_if_needed(context)
    write_summary(context)
    write_autonomous_prompt(context)

    run_dir = resolve_run_dir(context, args.run_id, None)
    metadata_path = run_dir / "metrics.json"
    reused_existing_run = metadata_path.exists() and not args.rerun
    if reused_existing_run:
        metadata = read_run_metadata(run_dir)
        run_result = {
            "run_id": args.run_id,
            "returncode": metadata.get("returncode"),
            "metrics": metadata.get("metrics", {}),
            "reused": True,
        }
    else:
        run_result_code = cmd_run(
            argparse.Namespace(
                manifest=args.manifest,
                repo_root=args.repo_root,
                trial=args.trial,
                run_id=args.run_id,
                run_dir=None,
                timeout=args.timeout,
            )
        )
        if run_result_code != 0:
            return run_result_code
        metadata = read_run_metadata(run_dir)
        run_result = {
            "run_id": args.run_id,
            "returncode": metadata.get("returncode"),
            "metrics": metadata.get("metrics", {}),
            "reused": False,
        }

    already_recorded = result_row_exists(context, args.run_id)
    if already_recorded:
        record_result = {"run_id": args.run_id, "recorded": "already_present"}
    else:
        record_result_code = cmd_record(
            argparse.Namespace(
                manifest=args.manifest,
                repo_root=args.repo_root,
                trial=args.trial,
                run_id=args.run_id,
                run_dir=None,
                idea_family=args.idea_family,
                description=args.description,
                status="auto",
                commit=None,
                apply_git_reset=False,
            )
        )
        if record_result_code != 0:
            return record_result_code
        record_result = {"run_id": args.run_id, "recorded": "yes"}

    write_summary(context)
    next_step = build_next_step_analysis(context, run_id=args.run_id)
    state = loop_state(context)
    write_next_steps(context, next_step)
    progress = write_progress_artifacts(context)
    print(
        json.dumps(
            {
                "artifact_root": str(context["artifact_root"]),
                "baseline": run_result,
                "next_steps": str(context["next_steps_path"]),
                "prompt": str(context["prompt_path"]),
                "progress": progress,
                "record": record_result,
                "snapshot": {
                    "path": str(context["snapshot_path"]),
                    "out_of_scope_dirty_count": len(snapshot["out_of_scope_dirty_files"]),
                },
                "loop": state,
            },
            sort_keys=True,
        )
    )
    return 0


def cmd_next(args: argparse.Namespace) -> int:
    context = load_context(args.manifest, args.repo_root)
    analysis = build_next_step_analysis(context, run_id=args.run_id)
    write_next_steps(context, analysis)
    write_progress_artifacts(context)
    print(json.dumps(analysis, indent=2, sort_keys=True, default=str))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    context = load_context(args.manifest, args.repo_root)
    state = loop_state(context)
    print(json.dumps(state, indent=2, sort_keys=True, default=str))
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    context = load_context(args.manifest, args.repo_root)
    write_summary(context)
    write_progress_artifacts(context)
    print(json.dumps({"summary": str(context["summary_path"])}))
    return 0


def cmd_progress(args: argparse.Namespace) -> int:
    context = load_context(args.manifest, args.repo_root)
    progress = write_progress_artifacts(context)
    print(json.dumps(progress, indent=2, sort_keys=True))
    return 0


def cmd_loop(args: argparse.Namespace) -> int:
    if args.max_steps < 1:
        raise AutoresearchError("--max-steps must be >= 1")
    context = load_context(args.manifest, args.repo_root)
    status = preflight_status(context)
    if not status["ready"]:
        raise AutoresearchError(
            "preflight failed: "
            + "; ".join(status["blocked_by"])
            + f". Next action: {status['next_action']}"
        )

    init_ledgers(context)
    if not read_worktree_snapshot(context):
        write_worktree_snapshot(context)
    write_autonomous_prompt(context)

    if not read_ledger(context):
        baseline_code = cmd_baseline(
            argparse.Namespace(
                manifest=args.manifest,
                repo_root=args.repo_root,
                trial=0,
                run_id="000_baseline",
                idea_family="parameter_tuning",
                description="baseline run before strategy edits",
                timeout=args.timeout,
                no_branch=args.no_branch,
                rerun=args.rerun_baseline,
            )
        )
        if baseline_code != 0:
            return baseline_code

    executed_steps = 0
    step_results: list[dict[str, Any]] = []
    stop_reason = "max_steps_reached"
    while executed_steps < args.max_steps:
        state = loop_state(context)
        if state["terminal"]:
            stop_reason = str(state["stop_reason"])
            break

        analysis = build_next_step_analysis(context)
        experiment = analysis["next_experiment"]
        edit_path = experiment_edit_path(context, experiment)
        original_edit_text = edit_path.read_text()
        applied = apply_manifest_experiment(context, experiment)
        run_id = str(experiment["run_id"])
        trial = trial_from_run_id(run_id)

        run_code = cmd_run(
            argparse.Namespace(
                manifest=args.manifest,
                repo_root=args.repo_root,
                trial=trial,
                run_id=run_id,
                run_dir=None,
                timeout=args.timeout,
            )
        )
        record_code = cmd_record(
            argparse.Namespace(
                manifest=args.manifest,
                repo_root=args.repo_root,
                trial=trial,
                run_id=run_id,
                run_dir=None,
                idea_family=experiment["idea_family"],
                description=experiment["change"],
                status="auto",
                commit=None,
                apply_git_reset=False,
            )
        )
        if record_code != 0:
            return record_code

        executed_steps += 1
        state = loop_state(context)
        latest = latest_result_row(read_ledger(context))
        rollback = maybe_rollback_discarded_experiment(
            context,
            edit_path=edit_path,
            original_text=original_edit_text,
            latest=latest,
        )
        analysis = build_next_step_analysis(context)
        write_next_steps(context, analysis)
        progress = write_progress_artifacts(context)
        step_results.append(
            {
                "run_id": run_id,
                "trial": trial,
                "run_returncode": run_code,
                "applied": applied,
                "rollback": rollback,
                "loop": state,
                "progress": progress,
            }
        )
        if state["terminal"]:
            stop_reason = str(state["stop_reason"])
            break

    final_state = loop_state(context)
    print(
        json.dumps(
            {
                "executed_steps": executed_steps,
                "max_steps": args.max_steps,
                "stop_reason": stop_reason if not final_state["terminal"] else final_state["stop_reason"],
                "terminal": final_state["terminal"],
                "loop": final_state,
                "steps": step_results,
            },
            sort_keys=True,
            default=str,
        )
    )
    return 0


def load_context(manifest_path: Path, repo_root_arg: Path | None = None) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    repo_root = (repo_root_arg.resolve() if repo_root_arg else find_repo_root(manifest_path))
    manifest = load_manifest(manifest_path)
    artifact_root = resolve_repo_path(repo_root, manifest["paths"]["artifact_root"])
    run_root = resolve_repo_path(
        repo_root,
        manifest["paths"].get("run_root", artifact_root / "runs"),
    )
    allowed_roots = [
        rel_path(resolve_repo_path(repo_root, item), repo_root)
        for item in manifest["paths"]["allowed_edit_roots"]
    ]
    return {
        "manifest_path": manifest_path,
        "manifest_dir": manifest_path.parent,
        "manifest": manifest,
        "repo_root": repo_root,
        "artifact_root": artifact_root,
        "artifact_root_rel": rel_path(artifact_root, repo_root),
        "run_root": run_root,
        "run_root_rel": rel_path(run_root, repo_root),
        "allowed_edit_roots": allowed_roots,
        "results_path": artifact_root / "results.tsv",
        "events_path": artifact_root / "events.jsonl",
        "summary_path": artifact_root / "summary.md",
        "snapshot_path": artifact_root / WORKTREE_SNAPSHOT_FILE,
        "next_steps_path": artifact_root / NEXT_STEPS_FILE,
        "progress_path": artifact_root / PROGRESS_PNG_FILE,
        "progress_json_path": artifact_root / PROGRESS_JSON_FILE,
        "prompt_path": SKILL_ROOT / SKILL_FILE,
    }


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise AutoresearchError(f"manifest not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise AutoresearchError("manifest must be a mapping")
    validate_manifest(data)
    return data


def validate_manifest(data: dict[str, Any]) -> None:
    required = [
        "version",
        "run_tag",
        "strategy",
        "algorithm",
        "quota",
        "objective",
        "criteria",
        "paths",
        "command",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise AutoresearchError(f"manifest missing keys: {', '.join(missing)}")
    if int(data["version"]) != 1:
        raise AutoresearchError("manifest version must be 1")
    if max_trials(data) < 1:
        raise AutoresearchError("quota.max_trials must be >= 1")
    objective = data.get("objective", {})
    if objective.get("direction", "maximize") not in {"maximize", "minimize"}:
        raise AutoresearchError("objective.direction must be maximize or minimize")
    if not objective.get("metric"):
        raise AutoresearchError("objective.metric is required")
    criteria = data.get("criteria", {})
    if criteria.get("mode", "any") not in {"any", "all"}:
        raise AutoresearchError("criteria.mode must be any or all")
    for target in criteria.get("targets", []):
        if target.get("op") not in OPS:
            raise AutoresearchError(f"unsupported criteria op: {target.get('op')}")
    paths = data.get("paths", {})
    if not paths.get("artifact_root"):
        raise AutoresearchError("paths.artifact_root is required")
    if paths.get("run_root") is not None and not str(paths.get("run_root")).strip():
        raise AutoresearchError("paths.run_root must not be empty when provided")
    if not paths.get("allowed_edit_roots"):
        raise AutoresearchError("paths.allowed_edit_roots must not be empty")
    argv = data.get("command", {}).get("argv")
    if not isinstance(argv, list) or not argv:
        raise AutoresearchError("command.argv must be a non-empty list")
    validate_research_limits(data)
    validate_predictor_names(data)


def validate_research_limits(data: dict[str, Any]) -> None:
    actions = data.get("research_limits", {}).get("allowed_actions", [])
    if not actions:
        return
    seen: set[str] = set()
    for action in actions:
        action_id, _ = research_action_entry(action)
        if not action_id:
            raise AutoresearchError("research_limits.allowed_actions entries need an id")
        if action_id in seen:
            raise AutoresearchError(f"duplicate research action id: {action_id}")
        seen.add(action_id)


def validate_predictor_names(data: dict[str, Any]) -> None:
    raw_predictors = data.get("predictor_cols", [])
    if not raw_predictors:
        return
    forbidden = [
        str(column)
        for column in raw_predictors
        if str(column).startswith(TARGET_PREFIXES)
    ]
    if forbidden:
        raise AutoresearchError(
            "predictor_cols contains future-looking columns: " + ", ".join(forbidden)
        )


def init_ledgers(context: dict[str, Any]) -> None:
    context["artifact_root"].mkdir(parents=True, exist_ok=True)
    context["run_root"].mkdir(parents=True, exist_ok=True)
    if not context["results_path"].exists():
        context["results_path"].write_text("\t".join(RESULT_COLUMNS) + "\n")
    if not context["events_path"].exists():
        context["events_path"].write_text("")
    if not context["summary_path"].exists():
        context["summary_path"].write_text("# Autoresearch Summary\n\nNo trials recorded yet.\n")


def preflight_status(context: dict[str, Any], *, strict: bool = False) -> dict[str, Any]:
    changed = git_changed_paths(context["repo_root"])
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
        "artifact_root": rel_path(context["artifact_root"], context["repo_root"]),
        "run_root": rel_path(context["run_root"], context["repo_root"]),
        "allowed_edit_roots": sorted(context["allowed_edit_roots"]),
        "changed_file_count": len(changed),
        "out_of_scope_dirty_count": len(out_of_scope),
        "new_out_of_scope_dirty_count": len(new_out_of_scope),
        "new_out_of_scope_dirty_files": sorted(new_out_of_scope),
        "snapshot_exists": bool(snapshot),
        "snapshot_path": rel_path(context["snapshot_path"], context["repo_root"]),
    }


def write_worktree_snapshot(context: dict[str, Any]) -> dict[str, Any]:
    changed = git_changed_paths(context["repo_root"])
    out_of_scope = out_of_scope_paths(context, changed, include_artifacts=True)
    snapshot = {
        "created_at": utc_now(),
        "base_commit": git_output(context["repo_root"], ["rev-parse", "HEAD"]),
        "dirty_files": sorted(changed),
        "out_of_scope_dirty_files": sorted(out_of_scope),
    }
    context["snapshot_path"].write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    append_event(
        context,
        {
            "event": "worktree_snapshot",
            "timestamp": utc_now(),
            "snapshot": snapshot,
        },
    )
    return snapshot


def read_worktree_snapshot(context: dict[str, Any]) -> dict[str, Any] | None:
    path = context["snapshot_path"]
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def new_dirty_paths_since_snapshot(
    context: dict[str, Any],
    changed: set[str] | None = None,
) -> set[str]:
    changed = set(changed if changed is not None else git_changed_paths(context["repo_root"]))
    snapshot = read_worktree_snapshot(context)
    if not snapshot:
        return changed
    return changed - set(snapshot.get("dirty_files", []))


def edit_root_paths(context: dict[str, Any], paths: set[str]) -> set[str]:
    return {
        path
        for path in paths
        if any(is_same_or_child(path, root) for root in context["allowed_edit_roots"])
    }


def create_branch_if_needed(context: dict[str, Any]) -> None:
    run_tag = str(context["manifest"]["run_tag"])
    branch = f"autoresearch/{run_tag}"
    current = git_output(context["repo_root"], ["branch", "--show-current"])
    if current == branch:
        return
    existing = git_output(context["repo_root"], ["branch", "--list", branch])
    if existing:
        raise AutoresearchError(f"branch already exists: {branch}")
    subprocess.run(["git", "checkout", "-b", branch], cwd=context["repo_root"], check=True)


def resolve_run_dir(context: dict[str, Any], run_id: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return resolve_repo_path(context["repo_root"], explicit)
    return context["run_root"] / str(context["manifest"]["run_tag"]) / run_id


def render_command(context: dict[str, Any], run_id: str, run_dir: Path) -> list[str]:
    mapping = {
        "run_dir": str(run_dir),
        "run_id": run_id,
        "artifact_root": str(context["artifact_root"]),
        "run_root": str(context["run_root"]),
        "repo_root": str(context["repo_root"]),
        "manifest_dir": str(context["manifest_dir"]),
    }
    return [str(item).format(**mapping) for item in context["manifest"]["command"]["argv"]]


def extract_metrics(run_dir: Path, manifest: dict[str, Any]) -> tuple[dict[str, float], str]:
    summary_path = run_dir / "results_summary.csv"
    if summary_path.exists():
        metrics = parse_results_summary(summary_path, manifest.get("objective", {}))
        if metrics:
            return metrics, "results_summary.csv"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        metrics = parse_metrics_json(metrics_path)
        if metrics:
            return metrics, "metrics.json"
    log_path = run_dir / "run.log"
    if log_path.exists():
        metrics = parse_log_metrics(log_path)
        if metrics:
            return metrics, "run.log"
    return {}, "none"


def parse_results_summary(path: Path, objective: dict[str, Any]) -> dict[str, float]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    candidates = [
        row for row in rows if str(row.get("status", "ok")).lower() in {"ok", "keep", "success", ""}
    ]
    candidates = candidates or rows
    metric = str(objective.get("metric", "sharpe"))
    direction = str(objective.get("direction", "maximize"))
    numeric_rows = [(row, metric_float(row.get(metric))) for row in candidates]
    numeric_rows = [(row, value) for row, value in numeric_rows if value is not None]
    if numeric_rows:
        reverse = direction != "minimize"
        selected = sorted(numeric_rows, key=lambda item: item[1], reverse=reverse)[0][0]
    else:
        selected = candidates[-1]
    return numeric_values(selected)


def parse_metrics_json(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and isinstance(data.get("metrics"), dict):
        return numeric_values(data["metrics"])
    if isinstance(data, dict):
        return numeric_values(data)
    return {}


def parse_log_metrics(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "Baseline metrics:" in line:
            _, _, payload = line.partition("Baseline metrics:")
            try:
                value = ast.literal_eval(payload.strip())
            except (SyntaxError, ValueError):
                value = None
            if isinstance(value, dict):
                metrics.update(numeric_values(value))
        match = LOG_METRIC_RE.match(line)
        if match:
            metrics[match.group(1)] = float(match.group(2))
    return metrics


def evaluate_criteria(metrics: dict[str, Any], criteria: dict[str, Any]) -> tuple[bool, list[str]]:
    targets = list(criteria.get("targets", []))
    if not targets:
        return False, []
    hits = []
    evaluated = []
    for target in targets:
        metric = str(target.get("metric"))
        actual = metric_float(metrics.get(metric))
        expected = metric_float(target.get("value"))
        op = target.get("op")
        ok = actual is not None and expected is not None and OPS[str(op)](actual, expected)
        evaluated.append(ok)
        if ok:
            hits.append(metric)
    mode = str(criteria.get("mode", "any"))
    return (all(evaluated) if mode == "all" else any(evaluated)), hits


def is_improvement(value: float, current: float | None, objective: dict[str, Any]) -> bool:
    if current is None:
        return True
    min_delta = float(objective.get("min_delta", 0.0) or 0.0)
    if objective.get("direction", "maximize") == "minimize":
        return value < current - min_delta
    return value > current + min_delta


def select_best_value(current: float, value: float, objective: dict[str, Any]) -> float:
    if objective.get("direction", "maximize") == "minimize":
        return min(current, value)
    return max(current, value)


def read_ledger(context: dict[str, Any]) -> list[dict[str, Any]]:
    path = context["results_path"]
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        for key in NUMERIC_LEDGER_COLUMNS:
            if key in row:
                row[key] = metric_float(row[key])
    return rows


def append_result_row(context: dict[str, Any], row: dict[str, Any]) -> None:
    if not context["results_path"].exists():
        init_ledgers(context)
    with context["results_path"].open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=RESULT_COLUMNS,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writerow({column: clean_cell(row.get(column, "")) for column in RESULT_COLUMNS})


def append_event(context: dict[str, Any], event: dict[str, Any]) -> None:
    with context["events_path"].open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")


def result_row_exists(context: dict[str, Any], run_id: str) -> bool:
    return any(str(row.get("run_id")) == str(run_id) for row in read_ledger(context))


def best_kept_value(context: dict[str, Any]) -> float | None:
    objective = context["manifest"].get("objective", {})
    metric = objective_metric(context["manifest"])
    values = [
        row_objective_value(row, metric)
        for row in read_ledger(context)
        if row.get("status") == "keep"
    ]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return min(values) if objective.get("direction") == "minimize" else max(values)


def best_kept_commit(context: dict[str, Any]) -> str | None:
    best_value = best_kept_value(context)
    metric = objective_metric(context["manifest"])
    for row in read_ledger(context):
        value = row_objective_value(row, metric)
        if row.get("status") == "keep" and value == best_value:
            commit = str(row.get("commit") or "")
            return commit or None
    return None


def loop_state(context: dict[str, Any]) -> dict[str, Any]:
    rows = read_ledger(context)
    quota = max_trials(context["manifest"])
    if not rows:
        return {
            "terminal": False,
            "stop_reason": "no_trials",
            "trials_recorded": 0,
            "max_trials": quota,
            "remaining_trials": quota,
            "next_action": "run baseline",
            "can_stop": False,
        }

    latest = latest_result_row(rows)
    stop_reason = str(latest.get("stop_reason") or "continue")
    criteria_hit = str(latest.get("criteria_hit") or "").lower() == "true"
    quota_exhausted = len(rows) >= quota
    if criteria_hit:
        stop_reason = "criteria_met"
    elif quota_exhausted:
        stop_reason = "quota_exhausted"

    terminal = stop_reason in TERMINAL_STOP_REASONS
    next_action = (
        "stop: criteria met"
        if stop_reason == "criteria_met"
        else "stop: quota exhausted"
        if stop_reason == "quota_exhausted"
        else "continue with next experiment"
    )
    return {
        "terminal": terminal,
        "can_stop": terminal,
        "stop_reason": stop_reason,
        "trials_recorded": len(rows),
        "max_trials": quota,
        "remaining_trials": max(0, quota - len(rows)),
        "latest_run_id": latest.get("run_id", ""),
        "latest_trial": latest.get("trial", ""),
        "latest_status": latest.get("status", ""),
        "criteria_hit": criteria_hit,
        "next_action": next_action,
    }


def latest_result_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def sort_key(row: dict[str, Any]) -> tuple[int, str]:
        trial = metric_float(row.get("trial"))
        return (int(trial) if trial is not None else -1, str(row.get("run_id") or ""))

    return sorted(rows, key=sort_key)[-1]


def build_next_step_analysis(context: dict[str, Any], run_id: str | None = None) -> dict[str, Any]:
    state = loop_state(context)
    metadata = select_run_metadata(context, run_id)
    metrics = dict(metadata.get("metrics") or {})
    if not metrics:
        raise AutoresearchError("no metrics found; run and record a baseline first")

    config_path = strategy_config_path(context)
    config = load_yaml_if_exists(config_path)
    strategy_params = nested_mapping(config, ["strategy", "params"])
    portfolio_params = nested_mapping(strategy_params, ["portfolio", "params"])
    predictors = list(strategy_params.get("predictor_cols") or [])

    diagnostics = diagnose_next_step(context, metrics, predictors)
    experiment = choose_next_experiment(context, diagnostics)
    current = {
        "run_id": metadata.get("run_id", run_id or ""),
        "sharpe": metrics.get("sharpe"),
        "sortino": metrics.get("sortino"),
        "cagr": metrics.get("cagr"),
        "max_drawdown": metrics.get("max_drawdown"),
        "win_rate": metrics.get("win_rate"),
        "signal_rows": metrics.get("signal_rows"),
        "featured_rows": metrics.get("featured_rows"),
        "predictor_count": metrics.get("predictor_count", len(predictors)),
    }
    return {
        "loop": state,
        "current_best": current,
        "diagnosis": diagnostics,
        "next_experiment": experiment,
        "strategy_config": rel_path(config_path, context["repo_root"]) if config_path else "",
        "strategy_state": {
            "predictor_cols": predictors,
            "num_long_positions": portfolio_params.get("num_long_positions"),
            "num_short_positions": portfolio_params.get("num_short_positions"),
            "portfolio": nested_mapping(strategy_params, ["portfolio"]).get("name"),
            "rebalance_period": strategy_params.get("rebalance_period"),
        },
    }


def select_run_metadata(context: dict[str, Any], run_id: str | None = None) -> dict[str, Any]:
    if run_id:
        run_dir = resolve_run_dir(context, run_id, None)
        return read_run_metadata(run_dir)

    rows = read_ledger(context)
    metric = objective_metric(context["manifest"])
    direction = context["manifest"].get("objective", {}).get("direction", "maximize")
    rows_with_values = [
        (row, row_objective_value(row, metric))
        for row in rows
        if row_objective_value(row, metric) is not None
    ]
    if rows_with_values:
        reverse = direction != "minimize"
        row = sorted(rows_with_values, key=lambda item: item[1] or 0.0, reverse=reverse)[0][0]
        run_id = str(row.get("run_id") or "")
        run_dir_raw = str(row.get("artifact_dir") or "")
        run_dir = resolve_repo_path(context["repo_root"], run_dir_raw) if run_dir_raw else resolve_run_dir(context, run_id, None)
        metadata = read_run_metadata(run_dir)
        if not metadata.get("metrics") and run_id:
            metadata = read_run_metadata(resolve_run_dir(context, run_id, None))
        if metadata.get("metrics"):
            return metadata
        return {"run_id": run_id, "metrics": {key: row.get(key) for key in NUMERIC_LEDGER_COLUMNS}}

    metadata_files = sorted(
        (context["run_root"] / str(context["manifest"]["run_tag"])).glob("*/metrics.json")
    )
    if not metadata_files:
        return {"metrics": {}}
    return json.loads(metadata_files[-1].read_text())


def diagnose_next_step(
    context: dict[str, Any],
    metrics: dict[str, Any],
    predictors: list[Any],
) -> list[str]:
    policy = context["manifest"].get("next_step_policy", {})
    thresholds = policy.get("diagnostics", {})
    low_predictor_count = float(thresholds.get("low_predictor_count_below", 5))
    bad_win_rate = float(thresholds.get("bad_win_rate_below", 0.5))
    bad_drawdown = float(thresholds.get("bad_max_drawdown_above", 0.4))
    sparse_signals = float(thresholds.get("sparse_signal_rows_below", 100))

    diagnosis: list[str] = []
    predictor_count = metric_float(metrics.get("predictor_count"))
    if predictor_count is None:
        predictor_count = float(len(predictors))
    if predictor_count < low_predictor_count:
        diagnosis.append(
            f"predictor_count is only {format_metric(predictor_count)} "
            f"(< {format_metric(low_predictor_count)})"
        )

    win_rate = metric_float(metrics.get("win_rate"))
    if win_rate is not None and win_rate < bad_win_rate:
        diagnosis.append(
            f"win_rate {format_metric(win_rate)} is below {format_metric(bad_win_rate)}"
        )

    max_drawdown = metric_float(metrics.get("max_drawdown"))
    if max_drawdown is not None and max_drawdown > bad_drawdown:
        diagnosis.append(
            f"max_drawdown {format_metric(max_drawdown)} is above {format_metric(bad_drawdown)}"
        )

    signal_rows = metric_float(metrics.get("signal_rows"))
    if signal_rows is not None and signal_rows < sparse_signals:
        diagnosis.append(
            f"signal_rows {format_metric(signal_rows)} is sparse "
            f"(< {format_metric(sparse_signals)})"
        )

    return diagnosis or ["no threshold breach; run a bounded hyperparameter sweep next"]


def choose_next_experiment(context: dict[str, Any], diagnostics: list[str]) -> dict[str, Any]:
    rows = read_ledger(context)
    completed_text = "\n".join(
        f"{row.get('run_id', '')} {row.get('description', '')}" for row in rows
    ).lower()
    experiments = context["manifest"].get("next_step_policy", {}).get("first_experiments", [])
    for experiment in experiments:
        if not isinstance(experiment, dict):
            continue
        experiment_id = str(experiment.get("id", "")).strip()
        run_id = str(experiment.get("run_id", "")).strip()
        if experiment_id and experiment_id.lower() in completed_text:
            continue
        if run_id and run_id.lower() in completed_text:
            continue
        return normalize_experiment(context, experiment, diagnostics)

    fallback = {
        "id": "bounded_model_size_sweep",
        "run_id": next_trial_run_id(context, "bounded_model_size_sweep"),
        "idea_family": "hyperparameter_optimization",
        "edit": "configs/strategies/ml_factor/sweeps/model_size.yaml",
        "change": "run the existing bounded model-size sweep",
        "reason": diagnostics[0] if diagnostics else "no queued experiments remain",
    }
    return normalize_experiment(context, fallback, diagnostics)


def normalize_experiment(
    context: dict[str, Any],
    experiment: dict[str, Any],
    diagnostics: list[str],
) -> dict[str, Any]:
    experiment_id = normalize_action(str(experiment.get("id") or "next_trial"))
    idea_family = canonical_research_action(
        str(experiment.get("idea_family") or "parameter_tuning"),
        context["manifest"],
    )
    run_id = str(experiment.get("run_id") or next_trial_run_id(context, experiment_id))
    return {
        "id": experiment_id,
        "run_id": run_id,
        "idea_family": idea_family,
        "edit": str(experiment.get("edit") or strategy_config_path(context) or ""),
        "change": str(experiment.get("change") or "make one manifest-allowed improvement"),
        "reason": str(experiment.get("reason") or (diagnostics[0] if diagnostics else "")),
    }


def next_trial_run_id(context: dict[str, Any], slug: str) -> str:
    trials = [int(row["trial"]) for row in read_ledger(context) if metric_float(row.get("trial")) is not None]
    run_root = context["run_root"] / str(context["manifest"]["run_tag"])
    for path in run_root.glob("*/metrics.json") if run_root.exists() else []:
        metadata = json.loads(path.read_text())
        trial = metric_float(metadata.get("trial"))
        if trial is not None:
            trials.append(int(trial))
    return f"{(max(trials) + 1) if trials else 1:03d}_{normalize_action(slug)}"


def write_next_steps(context: dict[str, Any], analysis: dict[str, Any]) -> None:
    current = analysis["current_best"]
    experiment = analysis["next_experiment"]
    state = analysis.get("loop", loop_state(context))
    lines = [
        "# Next Autoresearch Step",
        "",
        "## Stop Gate",
        "",
        f"- terminal: {str(state.get('terminal')).lower()}",
        f"- stop_reason: {state.get('stop_reason')}",
        f"- trials_recorded: {state.get('trials_recorded')} / {state.get('max_trials')}",
        f"- next_action: {state.get('next_action')}",
        "",
        "## Current Best",
        "",
    ]
    for key in [
        "run_id",
        "sharpe",
        "win_rate",
        "max_drawdown",
        "signal_rows",
        "predictor_count",
    ]:
        lines.append(f"- {key}: {current.get(key, '')}")
    lines.extend(["", "## Diagnosis", ""])
    for item in analysis["diagnosis"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Next Experiment",
            "",
            f"- run_id: {experiment['run_id']}",
            f"- idea_family: {experiment['idea_family']}",
            f"- edit: {experiment['edit']}",
            f"- change: {experiment['change']}",
            f"- reason: {experiment['reason']}",
            "",
            "## Commands",
            "",
        ]
    )
    if state.get("terminal"):
        lines.extend(["No next run is required because the stop gate is terminal.", ""])
        context["next_steps_path"].write_text("\n".join(lines))
        return

    lines.extend(
        [
            "```bash",
            "python .codex/skills/qts-alpha-autoresearch/scripts/qts_autoresearch.py run \\",
            f"  {rel_path(context['manifest_path'], context['repo_root'])} \\",
            f"  --trial {trial_from_run_id(experiment['run_id'])} \\",
            f"  --run-id {experiment['run_id']}",
            "",
            "python .codex/skills/qts-alpha-autoresearch/scripts/qts_autoresearch.py record \\",
            f"  {rel_path(context['manifest_path'], context['repo_root'])} \\",
            f"  --trial {trial_from_run_id(experiment['run_id'])} \\",
            f"  --run-id {experiment['run_id']} \\",
            f"  --idea-family {experiment['idea_family']} \\",
            f"  --description \"{experiment['change']}\"",
            "```",
            "",
        ]
    )
    context["next_steps_path"].write_text("\n".join(lines))


def write_progress_artifacts(context: dict[str, Any]) -> dict[str, Any]:
    rows = progress_rows(context)
    metric = objective_metric(context["manifest"])
    direction = str(context["manifest"].get("objective", {}).get("direction", "maximize"))
    payload = {
        "generated_at": utc_now(),
        "metric": metric,
        "direction": direction,
        "experiments": len(rows),
        "kept_improvements": sum(1 for row in rows if row["status"] == "keep"),
        "rows": rows,
        "png": rel_path(context["progress_path"], context["repo_root"]),
    }
    context["progress_json_path"].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if rows:
        render_progress_png(context, rows, metric, direction)
    return {
        "progress_png": str(context["progress_path"]),
        "progress_json": str(context["progress_json_path"]),
        "experiments": payload["experiments"],
        "kept_improvements": payload["kept_improvements"],
    }


def write_autonomous_prompt(context: dict[str, Any]) -> None:
    prompt_path = context["prompt_path"]
    if not prompt_path.exists():
        raise AutoresearchError(f"skill prompt not found: {prompt_path}")


def apply_manifest_experiment(context: dict[str, Any], experiment: dict[str, Any]) -> dict[str, Any]:
    experiment_id = str(experiment.get("id") or "")
    edit_path = experiment_edit_path(context, experiment)
    edit_rel = rel_path(edit_path, context["repo_root"])

    config = load_yaml_if_exists(edit_path)
    if experiment_id == "add_momentum_volatility_features":
        apply_momentum_volatility_features(config)
    elif experiment_id == "increase_diversification":
        apply_increase_diversification(config)
    elif experiment_id == "switch_to_inverse_volatility":
        apply_inverse_volatility_portfolio(config)
    else:
        raise AutoresearchError(f"no automatic apply recipe for experiment: {experiment_id}")

    predictors = nested_mapping(config, ["strategy", "params"]).get("predictor_cols", [])
    validate_predictor_list([str(item) for item in predictors])
    edit_path.write_text(yaml.safe_dump(config, sort_keys=False))
    append_event(
        context,
        {
            "event": "experiment_applied",
            "timestamp": utc_now(),
            "experiment": experiment,
            "edit": edit_rel,
        },
    )
    return {"experiment_id": experiment_id, "edit": edit_rel}


def experiment_edit_path(context: dict[str, Any], experiment: dict[str, Any]) -> Path:
    edit_path = resolve_repo_path(context["repo_root"], str(experiment.get("edit") or ""))
    edit_rel = rel_path(edit_path, context["repo_root"])
    if not any(is_same_or_child(edit_rel, root) for root in context["allowed_edit_roots"]):
        raise AutoresearchError(f"experiment edit path escapes allowed roots: {edit_rel}")
    if not edit_path.exists():
        raise AutoresearchError(f"experiment edit path does not exist: {edit_rel}")
    return edit_path


def maybe_rollback_discarded_experiment(
    context: dict[str, Any],
    *,
    edit_path: Path,
    original_text: str,
    latest: dict[str, Any],
) -> dict[str, Any]:
    status = str(latest.get("status") or "")
    if status == "keep":
        return {"rolled_back": False, "reason": "kept"}
    edit_path.write_text(original_text)
    event = {
        "event": "experiment_rolled_back",
        "timestamp": utc_now(),
        "run_id": latest.get("run_id", ""),
        "status": status,
        "edit": rel_path(edit_path, context["repo_root"]),
    }
    append_event(context, event)
    return {"rolled_back": True, "reason": status or "not_kept", "edit": event["edit"]}


def apply_momentum_volatility_features(config: dict[str, Any]) -> None:
    features = config.setdefault("features", {})
    indicators = features.setdefault("indicators", [])
    ensure_indicator(indicators, "rsi", {"periods": [14, 63]})
    ensure_indicator(indicators, "roc", {"periods": [21, 63, 126, 252]})
    ensure_indicator(indicators, "macd", {"fast": 50, "slow": 200, "signal": 30})
    ensure_indicator(indicators, "atr", {"periods": [14]})
    ensure_indicator(indicators, "hist_vol", {"periods": [21, 63, 126]})
    params = nested_mapping(config, ["strategy", "params"])
    params["predictor_cols"] = [
        "close_qsmom_21_252_126",
        "roc_21",
        "roc_63",
        "roc_126",
        "roc_252",
        "rsi_14",
        "rsi_63",
        "macd_hist",
        "atr_14",
        "hist_vol_21",
        "hist_vol_63",
        "hist_vol_126",
        "close_to_sma_50",
        "close_to_sma_200",
        "volume_ratio_20",
        "dollar_volume_zscore_63",
    ]


def apply_increase_diversification(config: dict[str, Any]) -> None:
    portfolio_params = nested_mapping(config, ["strategy", "params", "portfolio", "params"])
    portfolio_params["num_long_positions"] = 10


def apply_inverse_volatility_portfolio(config: dict[str, Any]) -> None:
    portfolio = nested_mapping(config, ["strategy", "params", "portfolio"])
    portfolio["name"] = "inverse_volatility"
    params = portfolio.setdefault("params", {})
    params.setdefault("num_long_positions", 10)
    params.setdefault("num_short_positions", 0)
    params.setdefault("lookback_days", 63)


def ensure_indicator(indicators: list[Any], name: str, params: dict[str, Any]) -> None:
    for item in indicators:
        if isinstance(item, dict) and item.get("name") == name:
            item["params"] = merge_indicator_params(dict(item.get("params") or {}), params)
            return
    indicators.append({"name": name, "params": params})


def merge_indicator_params(current: dict[str, Any], desired: dict[str, Any]) -> dict[str, Any]:
    merged = dict(current)
    for key, value in desired.items():
        if isinstance(value, list):
            existing = merged.get(key, [])
            if not isinstance(existing, list):
                existing = []
            merged[key] = sorted({*existing, *value})
        else:
            merged[key] = value
    return merged


def validate_predictor_list(predictors: list[str]) -> None:
    forbidden = [item for item in predictors if item.startswith(TARGET_PREFIXES)]
    if forbidden:
        raise AutoresearchError(
            "experiment would add future-looking predictors: " + ", ".join(forbidden)
        )


def progress_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_ledger(context)
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        metric = objective_metric(context["manifest"])
        value = row_objective_value(row, metric)
        if value is None:
            continue
        trial = metric_float(row.get("trial"))
        x_value = int(trial) if trial is not None else index
        result.append(
            {
                "x": x_value,
                "trial": row.get("trial", ""),
                "run_id": str(row.get("run_id") or ""),
                "status": str(row.get("status") or ""),
                "stop_reason": str(row.get("stop_reason") or ""),
                "idea_family": str(row.get("idea_family") or ""),
                "description": str(row.get("description") or ""),
                "value": value,
            }
        )
    return sorted(result, key=lambda item: (item["x"], item["run_id"]))


def render_progress_png(
    context: dict[str, Any],
    rows: list[dict[str, Any]],
    metric: str,
    direction: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise AutoresearchError("matplotlib is required to render progress.png") from exc

    higher_is_better = direction != "minimize"
    xs = [row["x"] for row in rows]
    values = [float(row["value"]) for row in rows]
    kept = [row for row in rows if row["status"] == "keep"]
    discarded = [row for row in rows if row["status"] != "keep"]
    best_line = running_best_points(rows, higher_is_better=higher_is_better)

    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    if discarded:
        ax.scatter(
            [row["x"] for row in discarded],
            [row["value"] for row in discarded],
            s=18,
            color="#cfcfcf",
            alpha=0.65,
            edgecolors="none",
            label="Discarded",
            zorder=1,
        )
    if kept:
        ax.scatter(
            [row["x"] for row in kept],
            [row["value"] for row in kept],
            s=34,
            color="#18a558",
            edgecolors="#0b6b37",
            linewidths=0.7,
            label="Kept",
            zorder=3,
        )
    if best_line:
        ax.step(
            [point[0] for point in best_line],
            [point[1] for point in best_line],
            where="post",
            color="#20b15a",
            linewidth=1.7,
            label="Running best",
            zorder=2,
        )

    for row in kept:
        label = progress_label(row)
        if not label:
            continue
        ax.annotate(
            label,
            (row["x"], row["value"]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=7,
            rotation=28,
            color="#18864a",
            ha="left",
            va="bottom",
        )

    y_margin = max((max(values) - min(values)) * 0.08, abs(max(values) or 1.0) * 0.01, 1e-6)
    ax.set_ylim(min(values) - y_margin, max(values) + y_margin)
    if xs:
        x_margin = max(1, int((max(xs) - min(xs)) * 0.04))
        ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)

    direction_label = "higher is better" if higher_is_better else "lower is better"
    ax.set_xlabel("Experiment #")
    ax.set_ylabel(f"{metric} ({direction_label})")
    ax.set_title(
        f"Autoresearch Progress: {len(rows)} Experiments, {len(kept)} Kept Improvements"
    )
    ax.grid(True, color="#eeeeee", linewidth=0.8)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(context["progress_path"], dpi=150)
    plt.close(fig)


def running_best_points(
    rows: list[dict[str, Any]],
    *,
    higher_is_better: bool,
) -> list[tuple[int, float]]:
    best: float | None = None
    points: list[tuple[int, float]] = []
    for row in rows:
        value = float(row["value"])
        if row["status"] != "keep":
            if best is not None:
                points.append((int(row["x"]), best))
            continue
        if best is None:
            best = value
        elif higher_is_better:
            best = max(best, value)
        else:
            best = min(best, value)
        points.append((int(row["x"]), best))
    return points


def progress_label(row: dict[str, Any]) -> str:
    text = row.get("description") or row.get("run_id") or ""
    text = str(text).strip()
    if not text:
        return ""
    return text if len(text) <= 42 else text[:39].rstrip() + "..."


def strategy_config_path(context: dict[str, Any]) -> Path | None:
    argv = [str(item) for item in context["manifest"].get("command", {}).get("argv", [])]
    for index, item in enumerate(argv):
        if item == "--strategy-config" and index + 1 < len(argv):
            return resolve_repo_path(context["repo_root"], argv[index + 1])
    return None


def load_yaml_if_exists(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    return data if isinstance(data, dict) else {}


def nested_mapping(data: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key, {})
    return current if isinstance(current, dict) else {}


def trial_from_run_id(run_id: str) -> int:
    match = re.match(r"^(\d+)_", run_id)
    return int(match.group(1)) if match else 1


def format_metric(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4g}"


def write_trial_summary(
    run_dir: Path,
    row: dict[str, Any],
    metadata: dict[str, Any],
    tail: str,
) -> None:
    lines = [
        f"# Trial {row.get('run_id') or metadata.get('run_id', '')}",
        "",
        f"- Status: {row.get('status')}",
        f"- Stop reason: {row.get('stop_reason')}",
        f"- Objective: {row.get('objective_metric')} = {row.get('objective_value')}",
        f"- Best value: {row.get('best_value')}",
        f"- Criteria hit: {row.get('criteria_hit')} {row.get('hit_metric') or ''}".rstrip(),
        f"- Description: {row.get('description', '')}",
        "",
    ]
    metrics = metadata.get("metrics") or {}
    if metrics:
        lines.extend(["## Metrics", ""])
        for key in sorted(metrics):
            lines.append(f"- {key}: {metrics[key]}")
        lines.append("")
    if tail:
        lines.extend(["## Log Tail", "", "```text", tail.rstrip(), "```", ""])
    (run_dir / "trial_summary.md").write_text("\n".join(lines))


def write_summary(context: dict[str, Any]) -> None:
    rows = read_ledger(context)
    metric = objective_metric(context["manifest"])
    direction = context["manifest"].get("objective", {}).get("direction", "maximize")
    lines = [
        "# Autoresearch Summary",
        "",
        f"- Run tag: {context['manifest']['run_tag']}",
        f"- Strategy: {context['manifest']['strategy']}",
        f"- Algorithm: {context['manifest']['algorithm']}",
        f"- Objective: {metric} ({direction})",
        f"- Trials recorded: {len(rows)}",
        f"- Progress chart: {rel_path(context['progress_path'], context['repo_root'])}",
        f"- Run root: {rel_path(context['run_root'], context['repo_root'])}",
        "",
    ]
    actions = sorted(set(allowed_research_action_map(context["manifest"]).values()))
    if actions:
        lines.extend(["## Allowed Research Actions", ""])
        for action in actions:
            lines.append(f"- {action}")
        lines.append("")
    if not rows:
        lines.append("No trials recorded yet.")
        context["summary_path"].write_text("\n".join(lines) + "\n")
        return

    counts = Counter(str(row.get("status", "")) for row in rows)
    lines.extend(["## Status Counts", ""])
    for status, count in sorted(counts.items()):
        lines.append(f"- {status}: {count}")
    lines.append("")

    kept = [
        row
        for row in rows
        if row.get("status") == "keep" and metric_float(row.get("objective_value")) is not None
    ]
    if kept:
        reverse = direction != "minimize"
        leaderboard = sorted(
            kept,
            key=lambda row: metric_float(row.get("objective_value")) or 0.0,
            reverse=reverse,
        )
        lines.extend(
            [
                "## Leaderboard",
                "",
                "| rank | run_id | value | commit | idea_family |",
                "|---:|---|---:|---|---|",
            ]
        )
        for rank, row in enumerate(leaderboard[:10], start=1):
            lines.append(
                f"| {rank} | {row.get('run_id', '')} | {row.get('objective_value', '')} | "
                f"{row.get('commit', '')} | {row.get('idea_family', '')} |"
            )
        lines.append("")

    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = metric_float(row.get("objective_value"))
        if value is not None:
            families[str(row.get("idea_family") or "unspecified")].append(row)
    if families:
        lines.extend(
            [
                "## Idea Families",
                "",
                "| family | trials | best | detailed_idea | intuition |",
                "|---|---:|---:|---|---|",
            ]
        )
        for family, family_rows in sorted(families.items()):
            values = [
                value
                for value in (
                    metric_float(row.get("objective_value")) for row in family_rows
                )
                if value is not None
            ]
            best = min(values) if direction == "minimize" else max(values)
            lines.append(
                f"| {markdown_table_cell(family)} | {len(values)} | {best} | "
                f"{markdown_table_cell(idea_family_detail(family_rows, metric))} | "
                f"{markdown_table_cell(idea_family_intuition(family))} |"
            )
        lines.append("")
    context["summary_path"].write_text("\n".join(lines) + "\n")


def idea_family_detail(rows: list[dict[str, Any]], metric: str) -> str:
    details: list[str] = []
    for row in sorted(rows, key=idea_family_row_key):
        run_id = str(row.get("run_id") or "trial")
        status = str(row.get("status") or "unknown")
        value = row_objective_value(row, metric)
        value_text = format_metric(value) if value is not None else "n/a"
        description = clean_cell(row.get("description"))
        prefix = f"{run_id} ({status}, {metric} {value_text})"
        details.append(f"{prefix}: {description}" if description else prefix)
    return "; ".join(details) if details else "No recorded idea details."


def idea_family_row_key(row: dict[str, Any]) -> tuple[int, str]:
    trial = metric_float(row.get("trial"))
    trial_index = int(trial) if trial is not None else sys.maxsize
    return (trial_index, str(row.get("run_id") or ""))


def idea_family_intuition(family: str) -> str:
    explanations = {
        "feature_engineering": (
            "Change the input signals so the model can see different market or "
            "fundamental information."
        ),
        "hyperparameter_optimization": (
            "Tune training settings to improve how the model learns from the same inputs."
        ),
        "parameter_tuning": (
            "Adjust strategy windows, thresholds, or portfolio settings without changing "
            "the core model."
        ),
        "strategy_model_update": (
            "Change the strategy or model logic to test a different decision rule."
        ),
    }
    return explanations.get(
        family,
        "Use the recorded trial details to understand the practical idea being tested.",
    )


def markdown_table_cell(value: Any) -> str:
    return clean_cell(value).replace("|", r"\|")


def read_run_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"metrics": {}, "returncode": 1, "wall_seconds": None, "artifact_dir": str(run_dir)}


def error_tail(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""
    content = path.read_text(errors="replace").splitlines()
    return "\n".join(content[-lines:])


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "qts").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    raise AutoresearchError(f"could not find QTradeSystematic root from {start}")


def resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def rel_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def git_output(repo_root: Path, args: list[str]) -> str:
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
    output = git_output(repo_root, ["show", "--pretty=format:", "--name-only", commit])
    return {line.strip() for line in output.splitlines() if line.strip()}


def changed_files_for_record(context: dict[str, Any], commit: str) -> set[str]:
    if not commit or commit == "unknown":
        return set()
    repo_root = context["repo_root"]
    commit_full = git_output(repo_root, ["rev-parse", "--verify", commit])
    if not commit_full:
        return set()
    snapshot = read_worktree_snapshot(context)
    base_commit = str((snapshot or {}).get("base_commit") or "")
    if base_commit:
        base_full = git_output(repo_root, ["rev-parse", "--verify", base_commit])
        if base_full == commit_full:
            return set()
        if base_full:
            output = git_output(repo_root, ["diff", "--name-only", f"{base_full}..{commit_full}"])
            return {line.strip() for line in output.splitlines() if line.strip()}
    return changed_files_for_commit(repo_root, commit_full)


def out_of_scope_paths(
    context: dict[str, Any],
    paths: set[str],
    *,
    include_artifacts: bool,
) -> set[str]:
    allowed = set(context["allowed_edit_roots"])
    if include_artifacts:
        allowed.add(context["artifact_root_rel"])
        allowed.add(context["run_root_rel"])
    return {path for path in paths if not any(is_same_or_child(path, root) for root in allowed)}


def is_same_or_child(path: str, root: str) -> bool:
    clean_path = path.strip("/").replace("\\", "/")
    clean_root = root.strip("/").replace("\\", "/")
    return clean_path == clean_root or clean_path.startswith(f"{clean_root}/")


def numeric_values(row: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, raw in row.items():
        value = metric_float(raw)
        if value is not None:
            values[str(key)] = value
    return values


def metric_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def canonical_research_action(value: str, manifest: dict[str, Any]) -> str:
    normalized = normalize_action(value)
    action_map = allowed_research_action_map(manifest)
    if not action_map:
        return normalized
    if normalized in action_map:
        return action_map[normalized]
    allowed = ", ".join(sorted(set(action_map.values())))
    raise AutoresearchError(
        f"research action is not allowed: {value}. Allowed actions: {allowed}"
    )


def allowed_research_action_map(manifest: dict[str, Any]) -> dict[str, str]:
    actions = manifest.get("research_limits", {}).get("allowed_actions", [])
    action_map: dict[str, str] = {}
    for action in actions:
        action_id, aliases = research_action_entry(action)
        if not action_id:
            continue
        alias_values = set(aliases)
        alias_values.add(action_id)
        alias_values.update(DEFAULT_ACTION_ALIASES.get(action_id, set()))
        for alias in alias_values:
            action_map[normalize_action(alias)] = action_id
    return action_map


def research_action_entry(action: object) -> tuple[str, list[str]]:
    if isinstance(action, str):
        return normalize_action(action), []
    if not isinstance(action, dict):
        return "", []
    action_id = normalize_action(str(action.get("id", "")))
    aliases = [normalize_action(str(alias)) for alias in action.get("aliases", [])]
    return action_id, aliases


def normalize_action(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def row_objective_value(row: dict[str, Any], metric: str) -> float | None:
    if row.get("objective_metric") == metric:
        return metric_float(row.get("objective_value"))
    return metric_float(row.get(metric))


def objective_metric(manifest: dict[str, Any]) -> str:
    return str(manifest.get("objective", {}).get("metric", "sharpe"))


def max_trials(manifest: dict[str, Any]) -> int:
    return int(manifest.get("quota", {}).get("max_trials", 1))


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
