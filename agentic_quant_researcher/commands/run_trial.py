"""Run trial command."""

from __future__ import annotations

import argparse

from agentic_quant_researcher.coordinators.autoresearch import run_trial


def run(args: argparse.Namespace) -> tuple[dict[str, object], int]:
    return run_trial(args)
