"""CLI for the V1 agentic quant autoresearch port."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agentic_quant_researcher.commands import (
    baseline,
    preflight,
    record,
    run_trial,
    status,
    summarize,
)
from agentic_quant_researcher.commands import (
    next as next_command,
)
from agentic_quant_researcher.coordinators.autoresearch import check_edits
from agentic_quant_researcher.schemas.autoresearch import AutoresearchError


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
        if isinstance(result, tuple):
            payload, returncode = result
            print_json(payload)
            return int(returncode)
        print_json(result)
        return 0
    except AutoresearchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agentic quant autoresearch helper")
    parser.add_argument("--repo-root", type=Path, help="QTradeSystematic project root override")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight", help="Check manifest readiness")
    add_manifest_arg(preflight_parser)
    preflight_parser.add_argument("--strict", action="store_true")
    preflight_parser.set_defaults(func=preflight.run)

    baseline_parser = subparsers.add_parser("baseline", help="Run and record baseline")
    add_manifest_arg(baseline_parser)
    baseline_parser.add_argument("--trial", type=int, default=0)
    baseline_parser.add_argument("--run-id", default="000_baseline")
    baseline_parser.add_argument("--idea-family", default="parameter_tuning")
    baseline_parser.add_argument("--description", default="baseline run before strategy edits")
    baseline_parser.add_argument("--timeout", type=float)
    baseline_parser.add_argument("--rerun", action="store_true")
    baseline_parser.set_defaults(func=baseline.run)

    check_parser = subparsers.add_parser("check-edits", help="Validate dirty edit scope")
    add_manifest_arg(check_parser)
    check_parser.add_argument("--edit-roots-only", action="store_true")
    check_parser.add_argument("--strict", action="store_true")
    check_parser.set_defaults(func=check_edits)

    run_parser = subparsers.add_parser("run", help="Run manifest command into run.log")
    add_manifest_arg(run_parser)
    run_parser.add_argument("--trial", type=int, required=True)
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--run-dir", type=Path)
    run_parser.add_argument("--timeout", type=float)
    run_parser.set_defaults(func=run_trial.run)

    record_parser = subparsers.add_parser("record", help="Record a trial")
    add_manifest_arg(record_parser)
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
    record_parser.set_defaults(func=record.run)

    next_parser = subparsers.add_parser("next", help="Write next_steps.md")
    add_manifest_arg(next_parser)
    next_parser.add_argument("--run-id")
    next_parser.set_defaults(func=next_command.run)

    status_parser = subparsers.add_parser("status", help="Show loop status")
    add_manifest_arg(status_parser)
    status_parser.set_defaults(func=status.run)

    summarize_parser = subparsers.add_parser("summarize", help="Rebuild summary.md")
    add_manifest_arg(summarize_parser)
    summarize_parser.set_defaults(func=summarize.run)
    return parser


def add_manifest_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("manifest", type=Path)


def print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
