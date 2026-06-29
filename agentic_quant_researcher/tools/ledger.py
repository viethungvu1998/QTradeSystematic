"""Result and event ledger persistence."""

from __future__ import annotations

import csv
import json
import sys
from datetime import UTC, datetime
from typing import Any

from agentic_quant_researcher.schemas.autoresearch import AutoresearchContext
from agentic_quant_researcher.tools.criteria import clean_cell
from agentic_quant_researcher.tools.metrics import metric_float

csv.field_size_limit(sys.maxsize)

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


def init_ledgers(context: AutoresearchContext) -> None:
    context.artifact_root.mkdir(parents=True, exist_ok=True)
    context.run_root.mkdir(parents=True, exist_ok=True)
    if not context.results_path.exists():
        context.results_path.write_text("\t".join(RESULT_COLUMNS) + "\n")
    if not context.events_path.exists():
        context.events_path.write_text("")
    if not context.summary_path.exists():
        context.summary_path.write_text("# Autoresearch Summary\n\nNo trials recorded yet.\n")


def read_ledger(context: AutoresearchContext) -> list[dict[str, Any]]:
    if not context.results_path.exists():
        return []
    with context.results_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        for key in NUMERIC_LEDGER_COLUMNS:
            if key in row:
                row[key] = metric_float(row[key])
    return rows


def append_result_row(context: AutoresearchContext, row: dict[str, Any]) -> None:
    if not context.results_path.exists():
        init_ledgers(context)
    with context.results_path.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=RESULT_COLUMNS,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writerow({column: clean_cell(row.get(column, "")) for column in RESULT_COLUMNS})


def append_event(context: AutoresearchContext, event: dict[str, Any]) -> None:
    if not context.events_path.exists():
        init_ledgers(context)
    with context.events_path.open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")


def result_row_exists(context: AutoresearchContext, run_id: str) -> bool:
    return any(str(row.get("run_id")) == str(run_id) for row in read_ledger(context))


def read_run_metadata(run_dir) -> dict[str, Any]:
    path = run_dir / "metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"metrics": {}, "returncode": 1, "wall_seconds": None, "artifact_dir": str(run_dir)}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
