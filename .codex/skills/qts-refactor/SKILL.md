---
name: qts-refactor
description: Use when refactoring QTradeSystematic to remove navigation tax, compatibility shims, pass-through modules, duplicate helpers, one-use abstractions, stale public import paths, and speculative layers while preserving earned architecture seams and verified behavior.
---

# QTS Refactor

Read the QTradeSystematic codebase carefully and identify unnecessary "navigation tax": compatibility shims, pass-through modules, wrapper files, duplicate helpers, one-use abstractions, speculative layers, stale public import paths, and modules that only exist to preserve backward compatibility.

I explicitly allow breaking backward compatibility if it makes the codebase smaller, clearer, and more direct. Prefer the minimal current architecture over legacy import paths.

Use the repo rules in AGENTS.md, CODING_RULES.md, ARCHITECTURE.md, and WORKFLOWS.md, but prioritize removing unnecessary layers where the rules permit it.

Scope:
- Inspect the whole codebase, especially `qts/research/strategies/`, `qts/config/`, `qts/orchestration/`, and tests.
- Do not import from or depend on `reference/`; it is design reference only.
- Do not touch live broker behavior or add paper/live API calls.
- Do not invent new strategy logic, signal rules, risk rules, schema changes, or broker behavior.
- It is okay to update tests/configs/notebooks to the new canonical paths.
- It is okay to delete old compatibility tests if they only protect legacy import paths.

Refactor goals:
1. Remove unnecessary compatibility modules such as shim-only `model.py`, `strategy.py`, `base_model.py`, or similar files when they add no behavior.
2. Rename concrete implementation files to the simplest canonical names when that reduces navigation.
3. Move imports to the actual owning modules instead of re-export barrels when clearer.
4. Collapse single-use wrappers and helpers into their only caller.
5. Delete dead code, stale tests, and duplicate paths.
6. Keep real earned seams: registry plugins, ABCs, strategy `generate_signals`, engine `run`, broker/data/storage/feature abstractions.
7. Preserve behavior unless the behavior only exists for backward compatibility.

Process:
- First produce a short audit table: file/module, why it is navigation tax, proposed action, risk.
- Then implement the changes surgically.
- Update imports, package `__init__.py`, configs, and tests.
- Remove obsolete tests that only assert backward compatibility.
- Keep behavioral tests that protect real strategy/data/backtest behavior.
- Run focused tests with `.venv/bin/pytest`.
- Run lint/format checks with `.venv/bin/ruff`.
- Finish with a concise summary of what was removed, what was renamed, what compatibility broke intentionally, and what verification passed.

Important:
- Use `rg` to trace imports before deleting anything.
- Do not reformat unrelated files.
- Do not revert unrelated existing user changes.
- Prefer deletion and directness over new abstraction.
