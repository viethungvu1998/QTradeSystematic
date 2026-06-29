"""Metric extraction for autoresearch trial artifacts."""

from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path
from typing import Any

from agentic_quant_researcher.schemas.autoresearch import ObjectiveSpec, ResearchManifest

LOG_METRIC_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)


def extract_metrics(run_dir: Path, manifest: ResearchManifest) -> tuple[dict[str, float], str]:
    """Extract metrics from known trial artifact formats."""
    summary_path = run_dir / "results_summary.csv"
    if summary_path.exists():
        metrics = parse_results_summary(summary_path, manifest.objective)
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


def parse_results_summary(path: Path, objective: ObjectiveSpec) -> dict[str, float]:
    """Parse a CSV summary and select the best candidate row by objective."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    candidates = [
        row
        for row in rows
        if str(row.get("status", "ok")).lower() in {"ok", "keep", "success", ""}
    ]
    candidates = candidates or rows
    numeric_rows = [(row, metric_float(row.get(objective.metric))) for row in candidates]
    numeric_rows = [(row, value) for row, value in numeric_rows if value is not None]
    if numeric_rows:
        reverse = objective.direction != "minimize"
        selected = sorted(numeric_rows, key=lambda item: item[1], reverse=reverse)[0][0]
    else:
        selected = candidates[-1]
    return numeric_values(selected)


def parse_metrics_json(path: Path) -> dict[str, float]:
    """Parse plain metrics JSON or wrapped run metadata JSON."""
    data = json.loads(path.read_text())
    if isinstance(data, dict) and isinstance(data.get("metrics"), dict):
        return numeric_values(data["metrics"])
    if isinstance(data, dict):
        return numeric_values(data)
    return {}


def parse_log_metrics(path: Path) -> dict[str, float]:
    """Parse simple ``metric: value`` lines and old ``Baseline metrics: {...}`` logs."""
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
