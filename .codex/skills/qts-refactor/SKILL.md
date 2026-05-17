---
name: qts-refactor
description: Deeply refactor QTradeSystematic Python and YAML code so it complies with `QTradeSystematic/CODING_RULES.md`, stays feature-complete, becomes lighter-weight, and keeps polymorphism at the real extension points. Use for architecture cleanup, simplification, adapter isolation, dependency injection, registry/plugin migration, strategy-engine seam fixes, flow/config cleanup, and repo-specific compliance reviews.
---

# QTS Refactor

Use this skill when the task is to refactor existing QTradeSystematic code to match the repo's architecture guide. Treat "refactor" as a deep refactor by default: preserve features, reduce weight, simplify control flow, and keep polymorphism only where it carries real architectural value. Start by reading `QTradeSystematic/CODING_RULES.md`, the target files, and the closest owning abstractions before editing.

## Workflow

1. Build context
- Read the target files plus the nearest `Base*` contract, registry entrypoint, config builder, and tests.
- For broad cleanup or review requests, also read [references/refactor-checklist.md](references/refactor-checklist.md).
- If the change touches orchestration or YAML, inspect the registered component being configured before editing the flow.

2. Diagnose rule violations
- Remove forbidden imports from reference systems such as `omega`, `qsconnect`, `qsresearch`, and `qsautomate`.
- Keep vendor libraries behind adapters only. No direct `futu-api`, `binance-connector`, `vectorbtpro`, `zipline`, or `prefect` imports outside the owning adapter module.
- Replace inline asset-type branching with registry or router dispatch keyed by `AssetType.from_symbol(symbol)`.
- Replace concrete instantiation in consumers with dependency injection or registry lookup.
- Preserve the public seams: strategies expose `BaseStrategy.generate_signals(data)` and engines expose `BaseEngine.run(...)`.
- Remove hardcoded asset types, engines, brokers, commissions, schedules, and constructors from orchestration when config or registry seams already exist.
- Enforce module boundaries and the repo typing rules.
- Identify code weight that does not earn its complexity: pass-through wrappers, duplicate helpers, one-off abstractions, dead extension points, and conditionals that should become a shared polymorphic seam.

3. Refactor with the preferred patterns
- For a new broker, engine, feature, strategy, or data source: extend the relevant ABC, register it with a lowercase key, and reference the key from YAML.
- For asset-specific behavior: derive the asset type once from the symbol and hand off to a registry or router. Do not pattern-match raw symbols elsewhere.
- For strategy cleanup: move shared helpers to the base class or an earned family module instead of adding strategy-specific backtest runners.
- For feature cleanup: append columns only, keep per-symbol logic inside `.over("symbol")`, and preserve the input schema.
- For orchestration cleanup: wire dependencies from config and registry rather than constructing concrete implementations inline.
- For credentials: read env vars inside adapter `connect()` paths and keep secrets out of code and YAML.
- Prefer deleting code over moving code when behavior can be preserved with fewer layers.
- Keep polymorphism at true variation points such as brokers, data sources, features, strategies, and engines. Remove indirection that is not buying reuse, isolation, or substitutability.
- Collapse one-off adapters, helper classes, and wrapper functions when they only forward calls without owning policy.
- Prefer smaller public surfaces, simpler data flow, and fewer cross-module hops as long as all existing features remain intact.

4. Verify the refactor
- Re-read imports and constructors to confirm the architecture seam is actually restored.
- Run repo-local tooling from `QTradeSystematic/.venv` when verification is needed.
- Update or add tests at the correct tier: unit at the ABC boundary, integration with real in-memory storage, paper tests only against paper or testnet endpoints.
- In the final report, name the rules addressed and call out any remaining assumptions or follow-up migrations.

## Lightweight Refactor Heuristics

- Preserve externally visible behavior, supported config paths, and existing feature coverage unless the user explicitly asks for a behavioral change.
- Prefer one clear abstraction over a stack of thin abstractions.
- Inline trivial single-use helpers when they hide simple logic and make navigation harder.
- Merge split responsibilities only when they are not independent variation points.
- Convert repeated branching into polymorphism, but do not introduce new abstract layers for code that has only one realistic implementation.
- If two components differ only by configuration, prefer one implementation plus config over sibling classes.
- Keep registries and ABCs where the architecture already depends on them, but prune unused hooks and speculative seams.

## Non-Negotiables

- Do not introduce imports from reference systems.
- Do not bypass a registry with a hardcoded constructor when the registry already exists.
- Do not branch on `asset_type` inline in business logic.
- Do not expose alternate public backtest entrypoints from strategy packages.
- Do not hardcode credentials, broker names, engine names, schedules, or commission rates in flow code.
- Do not replace `Decimal` monetary values with `float`.
- Do not add explanatory comments unless the reason is non-obvious.

## Output Expectations

When using this skill, report:
- the rule violations found,
- the refactor applied,
- what was simplified or removed to make the code lighter,
- how feature parity was preserved,
- the verification run or skipped,
- any remaining migration risk if the codebase is only partially aligned.
