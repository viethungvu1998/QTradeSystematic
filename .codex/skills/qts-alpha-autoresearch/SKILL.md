---
name: qts-alpha-autoresearch
description: Run QTradeSystematic alpha autoresearch loops with manifest-scoped edits, baseline-first trials, redirected logs, result ledgers, and keep/discard advancement.
metadata:
  short-description: QTS alpha autoresearch loops
---

# QTS Alpha Autoresearch

Use this when the user asks to set up, run, or continue autonomous alpha research for a QTradeSystematic strategy.

Keep it simple, following `karpathy/autoresearch`: one manifest, one objective, one loop, every trial logged, keep only what works.

## Strategy Goals
- Check strategy goal from `qts/research/<algorithm>/autoresearch/autoresearch.yaml` in `criteria/targets`. The goal is to meet all of the criteria targets.

## Contract

- Work from `qts/research/<algorithm>/autoresearch/autoresearch.yaml`.
- Read only the manifest, allowed strategy/config files, and the fixed command target.
- Edit only `paths.allowed_edit_roots`.
- Run and record the unmodified baseline before trying ideas.
- Redirect run output to logs; do not flood the conversation with live output.
- Record every result, including crashes, in `results.tsv` / `events.jsonl`.
- Keep a candidate only when the objective improves, a criterion is hit, or equivalent results become simpler. Otherwise discard after artifacts are saved.
- Trial logs and ledgers are evidence, not candidate code. Keep them outside commits and preserve them when reverting failed code.
- Once the loop starts, do not ask whether to continue. Stop only when `status` is terminal for `criteria_met` or `quota_exhausted`, or an explicit `--max-steps` limit is reached.

## Commands

Set the manifest path once:

```bash
MANIFEST=qts/research/<algorithm>/autoresearch/autoresearch.yaml
```

Start with preflight and baseline:

```bash
python .codex/skills/qts-alpha-autoresearch/scripts/qts_autoresearch.py preflight "$MANIFEST"
python .codex/skills/qts-alpha-autoresearch/scripts/qts_autoresearch.py baseline "$MANIFEST"
```

Use the helper for the loop:

```bash
python .codex/skills/qts-alpha-autoresearch/scripts/qts_autoresearch.py status "$MANIFEST"
python .codex/skills/qts-alpha-autoresearch/scripts/qts_autoresearch.py next "$MANIFEST"
python .codex/skills/qts-alpha-autoresearch/scripts/qts_autoresearch.py check-edits "$MANIFEST"
python .codex/skills/qts-alpha-autoresearch/scripts/qts_autoresearch.py run "$MANIFEST" --trial N --run-id NNN_short_name
python .codex/skills/qts-alpha-autoresearch/scripts/qts_autoresearch.py record "$MANIFEST" --trial N --run-id NNN_short_name --idea-family FAMILY --description "short description"
```

If the user asks for autonomous execution, prefer:

```bash
python .codex/skills/qts-alpha-autoresearch/scripts/qts_autoresearch.py loop "$MANIFEST"
```

Add `--max-steps` only when the user gives a quota.

## Loop

1. Check branch, current best result, `summary.md`, and `next_steps.md`.
2. Run `next`; choose a manifest-allowed idea.
3. Edit only allowed files.
4. Run `check-edits`.
5. Commit the candidate if git advancement is authorized.
6. Run and record the trial.
7. Keep improvements; for failed ideas, preserve `results.tsv`, `events.jsonl`, and run artifacts, then restore only the candidate code to the previous kept state.
8. Run `status`; if `stop_reason=continue`, immediately start the next trial.

## QTS Guardrails

- Use only `research_limits.allowed_actions`.
- Do not commit run artifacts unless the user asks.
- Do not add predictors beginning with `forward_return`, `future_`, or `target_`.
- Avoid leakage: use only data available at prediction time, keep rolling features per-symbol, and lag fundamentals conservatively.
- Do not edit production modules unless the manifest and user explicitly expand scope.

## References

- Read `references/manifest-example.md` only when creating or fixing a manifest.
- Read `references/leakage-safe-alpha.md` before feature-engineering changes.
- Run `python .codex/skills/qts-alpha-autoresearch/scripts/qts_autoresearch.py --help` for full helper details.
