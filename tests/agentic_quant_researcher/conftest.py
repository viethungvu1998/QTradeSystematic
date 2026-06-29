"""Fixtures for agentic_quant_researcher tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from agentic_quant_researcher.schemas.autoresearch import load_context


@pytest.fixture
def manifest_data() -> dict[str, Any]:
    return {
        "version": 1,
        "run_tag": "unit_run",
        "strategy": "unit_strategy",
        "algorithm": "unit_algorithm",
        "quota": {"max_trials": 3},
        "objective": {"metric": "sharpe", "direction": "maximize", "min_delta": 0.0},
        "criteria": {
            "mode": "all",
            "targets": [
                {"metric": "sharpe", "op": ">=", "value": 1.0},
                {"metric": "max_drawdown", "op": "<=", "value": 0.25},
            ],
        },
        "paths": {
            "artifact_root": "artifacts/autoresearch",
            "run_root": "artifacts/runs",
            "allowed_edit_roots": ["configs/strategies/unit"],
        },
        "command": {"argv": [sys.executable, "run_trial.py", "{run_dir}", "{run_id}"]},
        "research_limits": {
            "allowed_actions": [
                {"id": "parameter_tuning", "aliases": ["params"]},
                {"id": "hyperparameter_optimization"},
            ]
        },
        "next_step_policy": {
            "first_experiments": [
                {
                    "id": "threshold_test",
                    "idea_family": "parameter_tuning",
                    "change": "try a threshold",
                }
            ]
        },
    }


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='test'\n")
    (root / "qts").mkdir()
    (root / "configs" / "strategies" / "unit").mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=root, check=True, stdout=subprocess.DEVNULL)
    return root


@pytest.fixture
def manifest_path(repo_root: Path, manifest_data: dict[str, Any]) -> Path:
    path = repo_root / "autoresearch.yaml"
    path.write_text(yaml.safe_dump(manifest_data, sort_keys=False))
    return path


@pytest.fixture
def context(manifest_path: Path, repo_root: Path):
    return load_context(manifest_path, repo_root)
