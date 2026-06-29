"""Command rendering and execution for autoresearch trials."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from agentic_quant_researcher.schemas.autoresearch import AutoresearchContext
from agentic_quant_researcher.tools.ledger import utc_now
from agentic_quant_researcher.tools.metrics import extract_metrics
from agentic_quant_researcher.utils.paths import resolve_repo_path


def resolve_run_dir(
    context: AutoresearchContext,
    run_id: str,
    explicit: Path | None = None,
) -> Path:
    if explicit is not None:
        return resolve_repo_path(context.repo_root, explicit)
    return context.run_root / context.manifest.run_tag / run_id


def render_command(context: AutoresearchContext, run_id: str, run_dir: Path) -> list[str]:
    mapping = {
        "run_dir": str(run_dir),
        "run_id": run_id,
        "artifact_root": str(context.artifact_root),
        "run_root": str(context.run_root),
        "repo_root": str(context.repo_root),
        "manifest_dir": str(context.manifest_dir),
    }
    return [item.format(**mapping) for item in context.manifest.command.argv]


def run_manifest_command(
    context: AutoresearchContext,
    *,
    trial: int,
    run_id: str,
    run_dir: Path | None = None,
    timeout: float | None = None,
) -> tuple[dict[str, object], int]:
    """Run the manifest command with stdout/stderr redirected to ``run.log``."""
    resolved_run_dir = resolve_run_dir(context, run_id, run_dir)
    resolved_run_dir.mkdir(parents=True, exist_ok=True)
    argv = render_command(context, run_id, resolved_run_dir)
    log_path = resolved_run_dir / "run.log"
    command_path = resolved_run_dir / "command.json"
    started_at = utc_now()
    start = time.monotonic()
    returncode = 0
    timed_out = False

    command_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "trial": trial,
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
                cwd=context.repo_root,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False,
            )
            returncode = int(completed.returncode)
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = 124
            log_file.write(f"\nTIMEOUT: exceeded {timeout} seconds\n")

    wall_seconds = time.monotonic() - start
    metrics, source = extract_metrics(resolved_run_dir, context.manifest)
    metadata: dict[str, object] = {
        "run_id": run_id,
        "trial": trial,
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_seconds": wall_seconds,
        "metrics": metrics,
        "metric_source": source,
        "artifact_dir": str(resolved_run_dir),
        "finished_at": utc_now(),
    }
    (resolved_run_dir / "metrics.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata, returncode
