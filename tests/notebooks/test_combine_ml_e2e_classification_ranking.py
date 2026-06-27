from __future__ import annotations

# ruff: noqa: E402, I001

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

NOTEBOOKS_DIR = Path(__file__).resolve().parents[2] / "notebooks"
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

import combine_ml_e2e_ranking as ranker_flow
from combine_ml_e2e_classification_ranking import (
    build_classifier_gated_ranker_targets,
    build_classifier_ranker_score_tilt_targets,
)


def test_classifier_gate_keeps_only_confirmed_ranker_weights():
    sessions = pd.date_range("2024-01-01", periods=2, freq="D")
    symbols = ["A", "B", "C"]
    ranker = pd.DataFrame(
        {"A": [0.6, 0.0], "B": [0.4, 0.5], "C": [0.0, 0.5]},
        index=sessions,
    )
    classifier = pd.DataFrame(
        {"A": [0.2, 0.0], "B": [0.0, 0.3], "C": [0.0, 0.0]},
        index=sessions,
    )

    hybrid = build_classifier_gated_ranker_targets(
        ranker_targets=ranker,
        classifier_targets=classifier,
        sessions=sessions,
        symbols=symbols,
    )

    expected = pd.DataFrame(
        {"A": [0.6, 0.0], "B": [0.0, 0.5], "C": [0.0, 0.0]},
        index=sessions,
    )
    pd.testing.assert_frame_equal(hybrid.targets, expected)
    assert hybrid.eval_summary["confirmed_position_ratio"] == pytest.approx(0.5)
    assert hybrid.eval_summary["retained_ranker_gross_ratio"] == pytest.approx(0.55)


def test_classifier_gate_can_renormalize_to_ranker_gross():
    sessions = pd.date_range("2024-01-01", periods=2, freq="D")
    symbols = ["A", "B", "C"]
    ranker = pd.DataFrame(
        {"A": [0.6, 0.0], "B": [0.4, 0.5], "C": [0.0, 0.5]},
        index=sessions,
    )
    classifier = pd.DataFrame(
        {"A": [0.2, 0.0], "B": [0.0, 0.3], "C": [0.0, 0.0]},
        index=sessions,
    )

    hybrid = build_classifier_gated_ranker_targets(
        ranker_targets=ranker,
        classifier_targets=classifier,
        sessions=sessions,
        symbols=symbols,
        renormalize_gross=True,
    )

    expected = pd.DataFrame(
        {"A": [1.0, 0.0], "B": [0.0, 1.0], "C": [0.0, 0.0]},
        index=sessions,
    )
    pd.testing.assert_frame_equal(hybrid.targets, expected)
    assert hybrid.eval_summary["avg_hybrid_gross"] == pytest.approx(1.0)


def test_score_tilt_keeps_classifier_universe_and_caps_weights():
    sessions = pd.date_range("2024-01-01", periods=3, freq="D")
    symbols = ["A", "B", "C"]
    classifier = pd.DataFrame(
        {"A": [0.2, 0.2, 0.0], "B": [0.0, 0.2, 0.0], "C": [0.0, 0.0, 0.2]},
        index=sessions,
    )
    raw_predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 3),
            "symbol": ["A", "B", "C"],
            "score": [3.0, 1.0, 2.0],
        }
    )

    hybrid = build_classifier_ranker_score_tilt_targets(
        classifier_targets=classifier,
        ranker_runs=[SimpleNamespace(raw_predictions=raw_predictions)],
        ranker_weights=[1.0],
        sessions=sessions,
        symbols=symbols,
        ranker_tilt_strength=0.5,
        max_position_weight=0.2,
    )

    expected = pd.DataFrame(
        {
            "A": [0.2, 0.2, 0.0],
            "B": [0.0, 0.18333333333333335, 0.0],
            "C": [0.0, 0.0, 0.2],
        },
        index=sessions,
    )
    pd.testing.assert_frame_equal(hybrid.targets, expected)
    assert hybrid.eval_summary["uses_benchmark_regime"] == 0.0
    assert hybrid.eval_summary["uses_min_top_score"] == 0.0
    assert hybrid.eval_summary["ranker_score_coverage_ratio"] == pytest.approx(0.75)
    assert hybrid.targets["C"].iloc[1] == 0.0
    assert hybrid.targets.abs().max().max() == pytest.approx(0.2)

    signals = ranker_flow.target_matrix_to_signals(hybrid.targets)
    assert not signals.is_empty()
    assert set(signals.columns) == {"date", "symbol", "signal", "weight"}
