"""Tests for trial metric extraction."""

from __future__ import annotations

import json
from pathlib import Path

from agentic_quant_researcher.schemas.autoresearch import ResearchManifest
from agentic_quant_researcher.tools.metrics import (
    parse_log_metrics,
    parse_metrics_json,
    parse_results_summary,
)


def test_metrics_json_extraction_plain(tmp_path: Path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"sharpe": 1.2, "ignored": "text"}))

    assert parse_metrics_json(path) == {"sharpe": 1.2}


def test_metrics_json_extraction_wrapped(tmp_path: Path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"metrics": {"sharpe": 1.4, "max_drawdown": 0.2}}))

    assert parse_metrics_json(path) == {"sharpe": 1.4, "max_drawdown": 0.2}


def test_results_summary_picks_best_row(tmp_path: Path, manifest_data):
    manifest = ResearchManifest.from_mapping(manifest_data)
    path = tmp_path / "results_summary.csv"
    path.write_text("status,sharpe,max_drawdown\nok,0.5,0.1\nok,1.3,0.3\n")

    metrics = parse_results_summary(path, manifest.objective)

    assert metrics["sharpe"] == 1.3
    assert metrics["max_drawdown"] == 0.3


def test_log_metrics_parses_simple_lines_and_baseline_dict(tmp_path: Path):
    path = tmp_path / "run.log"
    path.write_text("Baseline metrics: {'sharpe': 0.8}\nmax_drawdown: 0.22\n")

    assert parse_log_metrics(path) == {"sharpe": 0.8, "max_drawdown": 0.22}
