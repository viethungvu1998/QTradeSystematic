"""Record trial command."""

from __future__ import annotations

import argparse

from agentic_quant_researcher.coordinators.autoresearch import record


def run(args: argparse.Namespace) -> dict[str, object]:
    return record(args)
