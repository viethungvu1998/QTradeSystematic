---
description: Run QTradeSystematic alpha autoresearch loop — fully autonomous from preflight through criteria_met or quota_exhausted.
argument-hint: qts/research/<algorithm>/autoresearch/autoresearch.yaml [--max-steps N]
allowed_tools: ["Bash", "Read", "Write", "Edit", "Grep", "Glob"]
---

# QTS Alpha Autoresearch

**Input**: $ARGUMENTS

---

## Step 1 — Resolve the manifest path

Parse `$ARGUMENTS`:

| Input form | Resolution |
|---|---|
| Empty | `find qts/research -name "autoresearch.yaml"` — list all; ask user to pick if >1 |
| Algorithm name only (no `/`, no `.yaml`) | `qts/research/<name>/autoresearch/autoresearch.yaml` |
| Relative or absolute path ending in `.yaml` | Use as-is |
| Path + `--max-steps N` | Strip flag; set `MAX_STEPS=N` for Step 3 |

```bash
# Verify it exists
test -f "$MANIFEST" || { echo "Manifest not found: $MANIFEST"; exit 1; }
```

---

## Step 2 — Check ALGORITHM.md (advisory only)

```bash
ALGO_DIR=$(dirname "$(dirname "$MANIFEST")")
if [ ! -f "$ALGO_DIR/ALGORITHM.md" ]; then
  echo "WARNING: ALGORITHM.md missing at $ALGO_DIR/ALGORITHM.md"
  echo "Preflight will run without invariant contract checks."
  echo "Consider creating ALGORITHM.md before the first baseline."
fi
```

Do **not** block on a missing ALGORITHM.md — preflight handles this gracefully.

---

## Step 3 — Run the loop immediately. Do not ask for confirmation.

```bash
SCRIPT=".codex/skills/qts-alpha-autoresearch/scripts/qts_autoresearch.py"

# With explicit quota
python "$SCRIPT" loop "$MANIFEST" --max-steps "$MAX_STEPS"

# Or without (loop reads quota.max_trials from the manifest)
python "$SCRIPT" loop "$MANIFEST"
```

`loop` handles the full pipeline automatically:

1. **Preflight** — validates manifest, checks ALGORITHM.md invariants, reports `contract_warnings`
2. **Baseline** — runs the unmodified strategy and records it as trial 000 (skipped if already done)
3. **Experiment loop** — for each trial:
   - Picks the next experiment from `first_experiments` / `allowed_actions`
   - Applies the change (edits config, sweep YAML, or portfolio name)
   - For `sweep_hyperparams`: runs Optuna TPE sweep via `--sweep-config`; self-contained
   - For all other experiments: runs the notebook, reads `metrics.json`
   - Records result in `results.tsv` / `events.jsonl`
   - Keeps improvement; rolls back failure (preserves ledger artifacts)
   - Writes updated `summary.md` and `next_steps.md`
4. **Stops** when: `criteria_met`, `quota_exhausted`, `--max-steps` reached, or fatal error

---

## Step 4 — If the loop exits with a non-zero code

Run preflight to diagnose:

```bash
python "$SCRIPT" preflight "$MANIFEST"
```

Inspect `blocked_by` in the JSON output. Common causes and fixes:

| `blocked_by` value | Fix |
|---|---|
| `manifest_missing` | Create the autoresearch.yaml |
| `no_allowed_actions` | Add entries to `research_limits.allowed_actions` |
| `command_not_found` | Check `command.module` path and venv |
| `dirty_out_of_scope_files` | Commit or stash unexpected edits |

After fixing, re-run the loop:

```bash
python "$SCRIPT" loop "$MANIFEST"
```

---

## Step 5 — Report when done

Print a summary:

```bash
python "$SCRIPT" status "$MANIFEST"
```

Report to the user:
- `stop_reason` — why the loop stopped
- `executed_steps` — how many trials ran this session  
- Best result so far (Sharpe, CAGR, max_dd from `results.tsv`)
- Paths: `summary.md`, `next_steps.md`, `results.tsv`

---

## Guardrails (never violate)

- Edit only files under `paths.allowed_edit_roots` in the manifest
- Do not add features named `forward_return_*`, `future_*`, or `target_*`
- Do not change `portfolio.name` to `equal_weight` for classifier strategies (ALGORITHM.md invariant)
- Do not commit run artifacts (logs, sweep CSVs) unless the user explicitly asks
- Do not pause between trials to ask whether to continue
