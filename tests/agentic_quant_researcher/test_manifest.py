"""Tests for autoresearch manifest validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agentic_quant_researcher.schemas.autoresearch import AutoresearchError, load_manifest


def test_valid_manifest_loads(manifest_path):
    manifest = load_manifest(manifest_path)

    assert manifest.run_tag == "unit_run"
    assert manifest.objective.metric == "sharpe"
    assert manifest.criteria.mode == "all"
    assert manifest.command.argv[-1] == "{run_id}"


def test_missing_required_manifest_key_fails(tmp_path: Path, manifest_data):
    del manifest_data["run_tag"]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(manifest_data))

    with pytest.raises(AutoresearchError, match="manifest missing keys: run_tag"):
        load_manifest(path)


def test_future_predictor_name_fails(tmp_path: Path, manifest_data):
    manifest_data["predictor_cols"] = ["forward_return_21"]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(manifest_data))

    with pytest.raises(AutoresearchError, match="future-looking"):
        load_manifest(path)
