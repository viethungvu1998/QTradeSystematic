"""Next-step command."""

from __future__ import annotations

import argparse
from typing import Any

from agentic_quant_researcher.coordinators.autoresearch import next_step


def run(args: argparse.Namespace) -> dict[str, Any]:
    return next_step(args)
