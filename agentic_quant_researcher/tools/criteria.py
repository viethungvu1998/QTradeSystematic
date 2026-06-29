"""Objective, criteria, and loop-state evaluation."""

from __future__ import annotations

import operator
import sys
from typing import Any

from agentic_quant_researcher.schemas.autoresearch import ResearchManifest
from agentic_quant_researcher.tools.metrics import metric_float

OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
}
TERMINAL_STOP_REASONS = {"criteria_met", "quota_exhausted"}


def evaluate_criteria(
    metrics: dict[str, Any],
    manifest: ResearchManifest,
) -> tuple[bool, list[str]]:
    """Evaluate manifest criteria against a metric mapping."""
    if not manifest.criteria.targets:
        return False, []
    hits = []
    evaluated = []
    for target in manifest.criteria.targets:
        actual = metric_float(metrics.get(target.metric))
        ok = actual is not None and OPS[target.op](actual, target.value)
        evaluated.append(ok)
        if ok:
            hits.append(target.metric)
    if manifest.criteria.mode == "all":
        return all(evaluated), hits
    return any(evaluated), hits


def is_improvement(value: float, current: float | None, manifest: ResearchManifest) -> bool:
    """Return whether ``value`` improves the objective versus ``current``."""
    if current is None:
        return True
    min_delta = manifest.objective.min_delta
    if manifest.objective.direction == "minimize":
        return value < current - min_delta
    return value > current + min_delta


def select_best_value(current: float, value: float, manifest: ResearchManifest) -> float:
    if manifest.objective.direction == "minimize":
        return min(current, value)
    return max(current, value)


def row_objective_value(row: dict[str, Any], manifest: ResearchManifest) -> float | None:
    metric = manifest.objective.metric
    if row.get("objective_metric") == metric:
        return metric_float(row.get("objective_value"))
    return metric_float(row.get(metric))


def best_kept_value(rows: list[dict[str, Any]], manifest: ResearchManifest) -> float | None:
    values = [
        row_objective_value(row, manifest)
        for row in rows
        if row.get("status") == "keep"
    ]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return min(values) if manifest.objective.direction == "minimize" else max(values)


def latest_result_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def sort_key(row: dict[str, Any]) -> tuple[int, str]:
        trial = metric_float(row.get("trial"))
        return (int(trial) if trial is not None else -1, str(row.get("run_id") or ""))

    return sorted(rows, key=sort_key)[-1]


def loop_state(rows: list[dict[str, Any]], manifest: ResearchManifest) -> dict[str, Any]:
    """Return the current stop/continue state for a ledger."""
    quota = manifest.quota.max_trials
    if not rows:
        return {
            "terminal": False,
            "can_stop": False,
            "stop_reason": "no_trials",
            "trials_recorded": 0,
            "max_trials": quota,
            "remaining_trials": quota,
            "next_action": "run baseline",
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


def idea_family_detail(rows: list[dict[str, Any]], manifest: ResearchManifest) -> str:
    details: list[str] = []
    for row in sorted(rows, key=idea_family_row_key):
        run_id = str(row.get("run_id") or "trial")
        status = str(row.get("status") or "unknown")
        value = row_objective_value(row, manifest)
        value_text = format_metric(value) if value is not None else "n/a"
        description = clean_cell(row.get("description"))
        prefix = f"{run_id} ({status}, {manifest.objective.metric} {value_text})"
        details.append(f"{prefix}: {description}" if description else prefix)
    return "; ".join(details) if details else "No recorded idea details."


def idea_family_row_key(row: dict[str, Any]) -> tuple[int, str]:
    trial = metric_float(row.get("trial"))
    trial_index = int(trial) if trial is not None else sys.maxsize
    return (trial_index, str(row.get("run_id") or ""))


def format_metric(value: float | None) -> str:
    if value is None:
        return ""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4g}"


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()
