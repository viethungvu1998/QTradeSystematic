---
description: Parallel deep refactor audit — code-reviewer detects file size, anti-patterns, duplicates, legacy/inconsistent API surfaces, and dead code; architect maps layer violations and builds execution plan. Both run in parallel, results merged into a single prioritized report.
argument-hint: [scope: module path or leave empty for full repo]
allowed_tools: ["Bash", "Read", "Write", "Edit", "Grep", "Glob"]
---

# Deep Refactor

**Scope**: $ARGUMENTS (default: `qts/` if empty)
**Rules reference**: `CODING_RULES.md`, `CLAUDE.md` critical-rules section

---

## Step 1 — Resolve scope

```
If $ARGUMENTS is empty → scope = "qts/"
Else → scope = $ARGUMENTS
```

Verify scope exists:
```bash
test -e "$SCOPE" || { echo "Path not found: $SCOPE"; exit 1; }
```

Forbidden zones (flag findings, exclude from fix plan):
- `execution/` — real-money territory, never modify without explicit user approval

---

## Step 2 — Launch two subagents IN PARALLEL

Send both Agent calls in the same message. Do not wait for one before launching the other.

---

### Subagent A — `code-reviewer`

Prompt:

```
You are performing a deep refactor audit. Do NOT fix anything — produce a findings report only.

Scope: [resolved scope]
Rules: CODING_RULES.md and CLAUDE.md critical-rules section

---

## PHASE 1 — File Size Audit

Scan every file in scope. For each file exceeding 300 lines:
- Report: path, line count, one-line description of current responsibility
- Identify natural split boundaries by cohesion (not by file type)
- Propose child module names and responsibilities
- State which imports move where

Rule: if you need "and" to describe a file's responsibility, it should be split.

---

## PHASE 3 — Anti-Pattern Detection

Compare every class, function, and module against these rules:

### Pass-through delegation
  Flag: method body is only `return self.other.same_method(args)`
  Flag: class has exactly 1 method and exactly 1 caller

### Future-proofing bloat
  Flag: abstraction with only 1 concrete implementation and no second planned
  Flag: config/factory/registry with a single registered item

### Premature abstraction
  Flag: shared utility used in only 1 place
  Flag: base class with only 1 subclass where the base adds no contract enforcement

### Magic routing / hidden branching
  Flag: `if type == "x"` or `if asset_type == "..."` in business logic
        (should route via registry or polymorphism)

### Mutation patterns
  Flag: methods that modify state in-place when they could return new values

### Deep nesting
  Flag: nesting depth > 3 levels where early return would flatten it

For each finding: file, line range, pattern name, one-line impact statement.

---

## PHASE 4 — Duplicate / Parallel Implementation Detection

Find cases where similar logic is implemented multiple times independently.

Look for:
- Same algorithmic pattern applied to different types separately
- Similar class structures differing only in a type parameter or a single method
- Copy-pasted error handling, validation, or retry logic across modules
- Multiple functions doing the same transformation on structurally identical inputs

For each duplicate group:
- List all N implementations with file:line references
- Identify the invariant core (what is identical)
- Identify the variant (what legitimately differs)
- Propose: abstract base / protocol / generic function that replaces N with 1 + N small overrides

Polymorphism candidates:
  Flag any `isinstance` check or type-string branch that could become method dispatch on the object itself.

---

## PHASE 5 — Legacy Code & API Surface Unification

Find places where the same concept has multiple representations, causing callers to
use different syntax for identical intent.

### Fragmented parameter surfaces
  Flag: two config keys that express the same concept at different levels of indirection
    Example: `rebalance_frequency: "monthly"` AND `rebalance: "monthly"` coexisting —
    the wrapper key is legacy; the direct string form should be the only API.
  Flag: a parameter that only accepts a rigid type (int, enum) when it could accept
    semantic string shortcuts ("daily", "weekly", "monthly") and convert internally.
  Flag: separate boolean flags that combine to express what a single string value
    could express (e.g., `is_daily=True, is_monthly=False` → `frequency="daily"`).

### Inconsistent naming for the same concept
  Flag: the same domain concept named differently across modules
    (e.g., `rebalance_freq`, `rebalance_frequency`, `rebalance_period` for the same thing).
  Flag: config schema keys that differ from the runtime attribute they populate.

### Special-case accumulation
  Flag: if/elif chains that dispatch on string values where a lookup table or
    direct parameter passing would remove the branching entirely.
  Flag: caller-side translation layers (e.g., converting "monthly" → 21 days in 3
    different callers instead of once at the boundary).

For each finding:
- Show the current fragmented forms (all variants found, with file:line)
- Show the unified target form (single canonical API)
- State what breaks at call sites and how to migrate (rename, default, deprecation shim)

---

## PHASE 6 — Dead Code Detection

Find code that is never executed, never referenced, or permanently disabled.

### Unreachable code
  Flag: statements after `return`, `raise`, `break`, or `continue`
  Flag: branches where the condition is always True or always False
  Flag: exception handlers that catch a type that can never be raised in that block

### Unused symbols
  Flag: functions and methods with zero call sites in scope
  Flag: classes that are never instantiated and never subclassed
  Flag: module-level variables assigned but never read
  Flag: function parameters that are never used inside the function body

### Unused imports
  Flag: `import X` where X is never referenced in the file
  Flag: `from X import Y` where Y is never used

### Commented-out code blocks
  Flag: blocks of commented-out code (>3 lines) that have survived more than one commit
  Note: short single-line comments explaining why something was removed are fine — flag blocks only

### Orphaned config keys
  Flag: keys defined in config schemas or dataclasses that are never read at runtime
  Flag: YAML config fields documented but ignored by the parser

### Dead feature flags
  Flag: boolean flags or constants that are hardcoded to True/False and never toggled

For each finding:
- File, line range
- Symbol or block name
- Confidence: CERTAIN (provably unreachable) / LIKELY (no references found in scope) / SUSPECT (dynamic usage possible)
- Safe to delete: yes / needs verification

Do NOT flag:
- Public API symbols that may be called by external consumers outside the scanned scope
- Abstract methods on ABCs (they are contracts, not dead code)
- `__all__` exports
- Test fixtures and pytest helpers (they are called by the framework, not directly)

---

## PHASE 8 — Misplaced File Detection

Find files that sit at the wrong level of the directory hierarchy — typically a
narrowly-scoped file co-located with generic/base files at the same level.

### Signal: scope mismatch within a directory

For each directory, classify its files into two groups:
  - **Generic files**: base classes, managers, registries, interfaces, `__init__.py`
    (they define contracts or orchestrate — they belong at that level)
  - **Specific files**: files that serve exactly one market, one broker, one asset type,
    one data source, or one country
    (they are implementations — they should live one level deeper or in a dedicated subpackage)

If specific files and generic files share the same directory level → flag the specific ones as misplaced.

Example:
```
qts/data/
├── base.py          ← generic (belongs here)
├── manager.py       ← generic (belongs here)
├── vn_symbols.py    ← MISPLACED: VN-specific, should not share level with base/manager
└── sources/
    └── binance.py   ← specific but already in a subpackage (correct)
```

### Signals that identify a file as "specific" (not generic)

- Filename contains a market, country, or broker prefix/suffix:
  `vn_`, `crypto_`, `stock_`, `binance_`, `fmp_`, `_vn`, `_futures`, etc.
- File imports or instantiates only one concrete data source, broker, or market adapter
- File contains hardcoded symbols, market codes, country identifiers, or venue constants
- File's public API is only consumed by one caller that is itself market-specific

### Resolution logic

For each misplaced file, apply this decision tree:

```
1. Does a fitting subdirectory already exist?
   (e.g., sources/, adapters/, markets/, vn/, crypto/)
   → Yes: propose moving there

2. Would moving it require creating a new subdirectory
   that would also house ≥2 other related files?
   → Yes: propose creating the subdirectory + moving all related files

3. Is the file a one-off helper with no natural sibling group?
   → Yes: propose moving to the nearest utils/ or helpers/ module

4. None of the above?
   → Flag as misplaced with "no clear target — escalate to architect"
```

For each finding:
- Current path
- Why it is misplaced (one sentence: "VN-market-specific file at generic data layer level")
- Proposed target path
- Files that import it (must update their import paths)
- Whether an `__init__.py` re-export can preserve the old import for a transition period

---

## OUTPUT FORMAT

### File Size Findings
[table: path | lines | split proposal]

### Anti-Pattern Findings
[table: severity | file:line | pattern | impact]

### Duplicate Findings
[table: group | files | invariant core | proposed abstraction]

### Legacy / API Surface Findings
[table: severity | concept | current variants (file:line) | unified target form | migration note]

### Dead Code Findings
[table: confidence | file:line | symbol/block | safe to delete]

### Misplaced File Findings
[table: current path | reason misplaced | proposed target path | importers to update]

Severity scale: CRITICAL / HIGH / MEDIUM / LOW
```

---

### Subagent B — `architect`

Prompt:

```
You are performing a deep refactor audit. Do NOT fix anything — produce a findings report only.

Scope: [resolved scope]
Rules: CODING_RULES.md and CLAUDE.md critical-rules section

---

## PHASE 2 — Layer Depth Audit (Over-Engineering Detection)

For each significant feature or operation, trace the full call path from entry
point to actual side effect (DB write, API call, computation, state mutation).

For each layer in the chain, answer:
  (a) Testability: does removing this layer make the next layer untestable?
  (b) Isolation: does this layer hide an implementation detail the caller must not know?
  (c) Reuse: is this called from ≥2 independent callers?
  (d) Translation: does this layer change data shape, type, or protocol?

If ALL four answers are NO → mark layer as REMOVABLE.

Output per chain:
  entry_point → [hop1 ✓/✗] → [hop2 ✓/✗] → ... → side_effect
  Removable hops: list with reason

Signs of over-engineering to flag:
  - Method body is only `return self.other.same_method(args)`
  - Class has 1 method and 1 caller
  - Abstraction with 1 concrete implementation and no second planned
  - Config/factory/registry with a single registered item
  - 5+ file hops to trace a single feature end-to-end

---

## PHASE 7 — Prioritized Refactor Plan

Based on Phase 2 findings plus any cross-cutting structural observations, produce a ranked list.

For each item:
  - Severity: CRITICAL / HIGH / MEDIUM / LOW
  - File(s) affected
  - Problem (one sentence)
  - Proposed fix (one sentence)
  - Risk of change: low / medium / high + reason
  - Dependencies: must be done before/after which other items

Sort by: CRITICAL first, then risk-adjusted impact (high impact + low risk = do first).

Constraints to enforce in the plan:
  - Do not propose new abstractions unless they replace ≥2 existing implementations
  - Do not add layers — only remove or merge
  - Every proposed file split must result in files more cohesive than the original
  - Preserve all ABC boundaries used as mock seams in tests
  - Flag but exclude execution/ changes from the plan

---

## OUTPUT FORMAT

### Layer Depth Findings
[per chain: entry → hops → side_effect, with removable hops annotated]

### Prioritized Refactor Plan
[ranked table: severity | files | problem | fix | risk | dependencies]
```

---

## Step 3 — Merge and present

When both agents return, synthesize into one report:

```
# Deep Refactor Report — [scope] — [date]

## Summary
- Files audited: N
- Total findings: N (CRITICAL: N, HIGH: N, MEDIUM: N, LOW: N)
- Top 3 structural problems: [list]

## Merged Findings

### CRITICAL
[items from both agents, deduplicated — if confirmed by both, note it]

### HIGH
...

### MEDIUM
...

### LOW
...

## Execution Order

Phase 1 — Safe deletes (dead code, unused imports, commented-out blocks):
[list — zero behavior risk, do first]

Phase 2 — File relocations (misplaced files moved to correct subdirectory or utils):
[list — import path changes only, update importers, optionally add __init__.py re-export shim]

Phase 3 — API surface unification (rename params, merge legacy keys, add string shortcuts):
[list — call-site changes required, needs migration note per item]

Phase 4 — Structural (split files, collapse pass-through layers, merge duplicates):
[list — behavior-preserving but touches many files, requires test run after]

Phase 5 — Architectural (layer boundary violations, polymorphism refactor):
[list — highest risk, do last, one item at a time]

## Do Not Touch
[flagged-but-excluded areas with reason]
```

Deduplication rule: if both agents flag the same file/line, keep the more specific finding and mark it "confirmed by both agents".

Migration note rule: every API surface unification item must include a one-line migration note
— what callers need to change and whether a deprecation shim is needed before removal.
