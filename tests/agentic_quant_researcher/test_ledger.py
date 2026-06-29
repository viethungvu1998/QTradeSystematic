"""Tests for result ledgers."""

from __future__ import annotations

from agentic_quant_researcher.tools.ledger import append_result_row, init_ledgers, read_ledger


def test_ledger_appends_trial_row(context):
    init_ledgers(context)
    append_result_row(
        context,
        {
            "trial": 0,
            "run_id": "000_baseline",
            "status": "keep",
            "objective_metric": "sharpe",
            "objective_value": 1.2,
            "best_value": 1.2,
        },
    )

    rows = read_ledger(context)

    assert len(rows) == 1
    assert rows[0]["run_id"] == "000_baseline"
    assert rows[0]["objective_value"] == 1.2
