"""Markdown summaries and next-step analysis."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agentic_quant_researcher.schemas.autoresearch import (
    AutoresearchContext,
    AutoresearchError,
    allowed_research_action_map,
    canonical_research_action,
    normalize_action,
)
from agentic_quant_researcher.tools.criteria import (
    clean_cell,
    format_metric,
    idea_family_detail,
    loop_state,
    row_objective_value,
)
from agentic_quant_researcher.tools.ledger import (
    NUMERIC_LEDGER_COLUMNS,
    read_ledger,
    read_run_metadata,
)
from agentic_quant_researcher.tools.metrics import metric_float
from agentic_quant_researcher.tools.runner import resolve_run_dir
from agentic_quant_researcher.utils.paths import rel_path, resolve_repo_path


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


def write_summary(context: AutoresearchContext) -> None:
    rows = read_ledger(context)
    metric = context.manifest.objective.metric
    direction = context.manifest.objective.direction
    lines = [
        "# Autoresearch Summary",
        "",
        f"- Run tag: {context.manifest.run_tag}",
        f"- Strategy: {context.manifest.strategy}",
        f"- Algorithm: {context.manifest.algorithm}",
        f"- Objective: {metric} ({direction})",
        f"- Trials recorded: {len(rows)}",
        f"- Run root: {rel_path(context.run_root, context.repo_root)}",
        "",
    ]
    actions = sorted(set(allowed_research_action_map(context.manifest).values()))
    if actions:
        lines.extend(["## Allowed Research Actions", ""])
        for action in actions:
            lines.append(f"- {action}")
        lines.append("")
    if not rows:
        lines.append("No trials recorded yet.")
        context.summary_path.write_text("\n".join(lines) + "\n")
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
                "| family | trials | best | detailed_idea |",
                "|---|---:|---:|---|",
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
                f"{markdown_table_cell(idea_family_detail(family_rows, context.manifest))} |"
            )
        lines.append("")
    context.summary_path.write_text("\n".join(lines) + "\n")


def build_next_step_analysis(
    context: AutoresearchContext,
    run_id: str | None = None,
) -> dict[str, Any]:
    state = loop_state(read_ledger(context), context.manifest)
    metadata = select_run_metadata(context, run_id)
    metrics = dict(metadata.get("metrics") or {})
    if not metrics:
        raise AutoresearchError("no metrics found; run and record a baseline first")
    diagnostics = diagnose_next_step(context, metrics)
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
        "predictor_count": metrics.get("predictor_count"),
    }
    return {
        "loop": state,
        "current_best": current,
        "diagnosis": diagnostics,
        "next_experiment": experiment,
    }


def select_run_metadata(context: AutoresearchContext, run_id: str | None = None) -> dict[str, Any]:
    if run_id:
        return read_run_metadata(resolve_run_dir(context, run_id, None))
    rows = read_ledger(context)
    direction = context.manifest.objective.direction
    rows_with_values = [
        (row, row_objective_value(row, context.manifest))
        for row in rows
        if row_objective_value(row, context.manifest) is not None
    ]
    if rows_with_values:
        reverse = direction != "minimize"
        row = sorted(rows_with_values, key=lambda item: item[1] or 0.0, reverse=reverse)[0][0]
        run_id = str(row.get("run_id") or "")
        run_dir_raw = str(row.get("artifact_dir") or "")
        run_dir = (
            resolve_repo_path(context.repo_root, run_dir_raw)
            if run_dir_raw
            else resolve_run_dir(context, run_id, None)
        )
        metadata = read_run_metadata(run_dir)
        if metadata.get("metrics"):
            return metadata
        return {"run_id": run_id, "metrics": {key: row.get(key) for key in NUMERIC_LEDGER_COLUMNS}}
    metadata_files = sorted((context.run_root / context.manifest.run_tag).glob("*/metrics.json"))
    if not metadata_files:
        return {"metrics": {}}
    return json.loads(metadata_files[-1].read_text())


def diagnose_next_step(context: AutoresearchContext, metrics: dict[str, Any]) -> list[str]:
    thresholds = context.manifest.next_step_policy.get("diagnostics", {})
    low_predictor_count = float(thresholds.get("low_predictor_count_below", 5))
    bad_win_rate = float(thresholds.get("bad_win_rate_below", 0.5))
    bad_drawdown = float(thresholds.get("bad_max_drawdown_above", 0.4))
    sparse_signals = float(thresholds.get("sparse_signal_rows_below", 100))

    diagnosis: list[str] = []
    predictor_count = metric_float(metrics.get("predictor_count"))
    if predictor_count is not None and predictor_count < low_predictor_count:
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
    return diagnosis or ["no threshold breach; use the next manifest-defined experiment"]


def choose_next_experiment(
    context: AutoresearchContext,
    diagnostics: list[str],
) -> dict[str, str]:
    rows = read_ledger(context)
    completed_text = "\n".join(
        f"{row.get('run_id', '')} {row.get('description', '')}" for row in rows
    ).lower()
    experiments = context.manifest.next_step_policy.get("first_experiments", [])
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
        "id": "manifest_command_trial",
        "run_id": next_trial_run_id(context, "manifest_command_trial"),
        "idea_family": "parameter_tuning",
        "change": "run the manifest command without automatic code mutation",
        "reason": diagnostics[0] if diagnostics else "no queued experiments remain",
    }
    return normalize_experiment(context, fallback, diagnostics)


def normalize_experiment(
    context: AutoresearchContext,
    experiment: dict[str, Any],
    diagnostics: list[str],
) -> dict[str, str]:
    experiment_id = normalize_action(str(experiment.get("id") or "next_trial"))
    idea_family = canonical_research_action(
        str(experiment.get("idea_family") or "parameter_tuning"),
        context.manifest,
    )
    run_id = str(experiment.get("run_id") or next_trial_run_id(context, experiment_id))
    return {
        "id": experiment_id,
        "run_id": run_id,
        "idea_family": idea_family,
        "change": str(experiment.get("change") or "run one manifest-defined trial"),
        "reason": str(experiment.get("reason") or (diagnostics[0] if diagnostics else "")),
    }


def next_trial_run_id(context: AutoresearchContext, slug: str) -> str:
    trials = [
        int(row["trial"])
        for row in read_ledger(context)
        if metric_float(row.get("trial")) is not None
    ]
    run_root = context.run_root / context.manifest.run_tag
    for path in run_root.glob("*/metrics.json") if run_root.exists() else []:
        metadata = json.loads(path.read_text())
        trial = metric_float(metadata.get("trial"))
        if trial is not None:
            trials.append(int(trial))
    return f"{(max(trials) + 1) if trials else 1:03d}_{normalize_action(slug)}"


def write_next_steps(context: AutoresearchContext, analysis: dict[str, Any]) -> None:
    current = analysis["current_best"]
    experiment = analysis["next_experiment"]
    state = analysis.get("loop", loop_state(read_ledger(context), context.manifest))
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
            f"- change: {experiment['change']}",
            f"- reason: {experiment['reason']}",
            "",
            "## Commands",
            "",
        ]
    )
    if state.get("terminal"):
        lines.extend(["No next run is required because the stop gate is terminal.", ""])
        context.next_steps_path.write_text("\n".join(lines))
        return
    trial = trial_from_run_id(experiment["run_id"])
    manifest_rel = rel_path(context.manifest_path, context.repo_root)
    lines.extend(
        [
            "```bash",
            ".venv/bin/python -m agentic_quant_researcher run \\",
            f"  {manifest_rel} \\",
            f"  --trial {trial} \\",
            f"  --run-id {experiment['run_id']}",
            "",
            ".venv/bin/python -m agentic_quant_researcher record \\",
            f"  {manifest_rel} \\",
            f"  --trial {trial} \\",
            f"  --run-id {experiment['run_id']} \\",
            f"  --idea-family {experiment['idea_family']} \\",
            f"  --description \"{experiment['change']}\"",
            "```",
            "",
        ]
    )
    context.next_steps_path.write_text("\n".join(lines))


def error_tail(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""
    content = path.read_text(errors="replace").splitlines()
    return "\n".join(content[-lines:])


def markdown_table_cell(value: Any) -> str:
    return clean_cell(value).replace("|", r"\|")


def trial_from_run_id(run_id: str) -> int:
    import re

    match = re.match(r"^(\d+)_", run_id)
    return int(match.group(1)) if match else 1
