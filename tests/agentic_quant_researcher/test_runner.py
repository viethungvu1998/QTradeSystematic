"""Tests for command rendering and execution."""

from __future__ import annotations

import sys

from agentic_quant_researcher.tools.runner import (
    render_command,
    resolve_run_dir,
    run_manifest_command,
)


def test_render_command_replaces_placeholders(context):
    run_dir = resolve_run_dir(context, "001_trial")

    argv = render_command(context, "001_trial", run_dir)

    assert argv[0] == sys.executable
    assert argv[-1] == "001_trial"
    assert str(run_dir) in argv


def test_run_manifest_command_writes_metadata(context, repo_root):
    script = repo_root / "run_trial.py"
    script.write_text(
        "import json, pathlib, sys\n"
        "run_dir = pathlib.Path(sys.argv[1])\n"
        "(run_dir / 'metrics.json').write_text(json.dumps({'sharpe': 1.25}))\n"
        "print('sharpe: 1.25')\n"
    )

    metadata, returncode = run_manifest_command(context, trial=1, run_id="001_trial")

    assert returncode == 0
    assert metadata["metrics"]["sharpe"] == 1.25
    assert (resolve_run_dir(context, "001_trial") / "run.log").exists()
