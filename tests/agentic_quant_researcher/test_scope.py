"""Tests for manifest edit-scope checks."""

from __future__ import annotations

import pytest

from agentic_quant_researcher.schemas.autoresearch import AutoresearchError
from agentic_quant_researcher.tools.ledger import init_ledgers
from agentic_quant_researcher.tools.scope import check_edit_scope, write_worktree_snapshot


def test_scope_allows_allowed_edit_root(context, repo_root):
    init_ledgers(context)
    write_worktree_snapshot(context)
    (repo_root / "configs" / "strategies" / "unit" / "x.yaml").write_text("x: 1\n")

    result = check_edit_scope(context)

    assert result["scope"] == "ok"


def test_scope_rejects_outside_allowed_roots(context, repo_root):
    init_ledgers(context)
    write_worktree_snapshot(context)
    (repo_root / "README.md").write_text("changed\n")

    with pytest.raises(AutoresearchError, match="changes escape manifest scope"):
        check_edit_scope(context)
