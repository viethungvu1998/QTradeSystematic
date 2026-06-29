"""Summarize command."""

from __future__ import annotations

import argparse

from agentic_quant_researcher.coordinators.autoresearch import summarize


def run(args: argparse.Namespace) -> dict[str, str]:
    return summarize(args)
