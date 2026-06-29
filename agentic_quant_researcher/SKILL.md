---
name: agentic-quant-researcher
description: Run the V1 manifest-driven QTS autoresearch helper port with baseline, trial, metrics, ledgers, criteria, summaries, and scope checks.
---

# Agentic Quant Researcher

Use this skill for the V1 autoresearch helper under `agentic_quant_researcher/`.

V1 intentionally ports only the existing `qts-alpha-autoresearch` capability:

- manifest-driven baseline and trial runs
- redirected run logs
- metric extraction from `results_summary.csv`, `metrics.json`, or `run.log`
- `results.tsv` and `events.jsonl` ledgers
- keep/discard/crash recording
- objective, criteria, quota, and status evaluation
- `summary.md` and `next_steps.md`
- manifest-scoped edit checks

Do not use V1 for alpha mining, feature mining, collections, or ML feature-set selection.

## Commands

Run all commands from the repository root through the repo-local virtual environment.

```bash
.venv/bin/python -m agentic_quant_researcher preflight MANIFEST
.venv/bin/python -m agentic_quant_researcher baseline MANIFEST
.venv/bin/python -m agentic_quant_researcher check-edits MANIFEST
.venv/bin/python -m agentic_quant_researcher run MANIFEST --trial N --run-id RUN_ID
.venv/bin/python -m agentic_quant_researcher record MANIFEST --trial N --run-id RUN_ID --idea-family FAMILY --description "short description"
.venv/bin/python -m agentic_quant_researcher next MANIFEST
.venv/bin/python -m agentic_quant_researcher status MANIFEST
.venv/bin/python -m agentic_quant_researcher summarize MANIFEST
```

## Guardrails

- Use the current manifest schema with `run_tag`, `strategy`, `algorithm`, `objective`, `criteria`, `paths`, and `command.argv`.
- Do not mutate strategy/config code automatically in V1.
- Do not add predictors beginning with `forward_return`, `future_`, or `target_`.
- Keep run artifacts outside commits unless explicitly requested.
- Never touch live broker or production execution paths.
