# QTradeSystematic — Codex Context

This file is the short contributor-facing snapshot of the repository's actual current state.

For fuller detail, use:

- `README.md`
- `ARCHITECTURE.md`
- `WORKFLOWS.md`
- `FEATURES.md`
- `AGENTS.md`

## Current Reality

- No CLI entry point exists yet.
- `qts_flow` is async and accepts a config path string.
- `validation` is a config mode, not an automated promotion gate.
- `crypto_futures` is wired through both top-level orchestration flows.
- `Config.build()` resolves registry keys, but several adapters still need explicit real-client wiring for live use.

## Runtime Surface

### Main entry points

- `qts.orchestration.flow.qts_flow`
- `qts.orchestration.flows.data_fetch_flow`
- `qts.orchestration.serve`

### Registered strategy keys

- `factor`
- `ml_factor`
- `stat_arb`

### Registered engine keys

- `vectorbt`
- `fast`
- `zipline`
- `normal`

### Registered broker keys

- `moomoo`
- `binance`

## Notes for Contributors

- Update docs when changing runtime behavior, config keys, orchestration, or registry names.
- Prefer describing implemented behavior, not intended roadmap behavior.
- When in doubt, use the tests and the code paths in `qts/orchestration/flow.py` and `qts/config/builder.py` as the source of truth.
