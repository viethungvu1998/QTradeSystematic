"""Preflight command."""

from __future__ import annotations

import argparse

from agentic_quant_researcher.coordinators.autoresearch import preflight


def run(args: argparse.Namespace) -> dict[str, object]:
    return preflight(args)
