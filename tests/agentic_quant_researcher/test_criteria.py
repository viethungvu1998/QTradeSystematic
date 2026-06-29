"""Tests for objective, criteria, and loop state."""

from __future__ import annotations

from agentic_quant_researcher.schemas.autoresearch import ResearchManifest
from agentic_quant_researcher.tools.criteria import (
    evaluate_criteria,
    is_improvement,
    loop_state,
)


def test_criteria_all_mode(manifest_data):
    manifest = ResearchManifest.from_mapping(manifest_data)

    assert evaluate_criteria({"sharpe": 1.1, "max_drawdown": 0.2}, manifest) == (
        True,
        ["sharpe", "max_drawdown"],
    )
    assert evaluate_criteria({"sharpe": 1.1, "max_drawdown": 0.5}, manifest) == (
        False,
        ["sharpe"],
    )


def test_criteria_any_mode(manifest_data):
    manifest_data["criteria"]["mode"] = "any"
    manifest = ResearchManifest.from_mapping(manifest_data)

    assert evaluate_criteria({"sharpe": 1.1, "max_drawdown": 0.5}, manifest) == (
        True,
        ["sharpe"],
    )


def test_objective_maximize_and_minimize(manifest_data):
    manifest = ResearchManifest.from_mapping(manifest_data)
    assert is_improvement(1.1, 1.0, manifest)

    manifest_data["objective"] = {"metric": "max_drawdown", "direction": "minimize"}
    manifest = ResearchManifest.from_mapping(manifest_data)
    assert is_improvement(0.2, 0.3, manifest)


def test_loop_state_continue_and_terminal(manifest_data):
    manifest = ResearchManifest.from_mapping(manifest_data)
    rows = [
        {
            "trial": "0",
            "run_id": "000_baseline",
            "status": "keep",
            "stop_reason": "continue",
            "criteria_hit": "false",
        }
    ]

    assert loop_state(rows, manifest)["stop_reason"] == "continue"

    rows[0]["criteria_hit"] = "true"
    assert loop_state(rows, manifest)["stop_reason"] == "criteria_met"
