# AGENTS.md — QTradeSystematic

Single source-of-truth for AI agents and contributors working on this codebase.

Detailed references: [ARCHITECTURE.md](ARCHITECTURE.md) · [WORKFLOWS.md](WORKFLOWS.md) · [CODING_RULES.md](CODING_RULES.md)

---

## Core Principles

### 1. Think Before Coding

Before implementing:
- Identify the smallest reasonable interpretation of the request.
- State assumptions when they affect scope, behavior, data, security, or compatibility.
- Ask for clarification only when ambiguity blocks safe progress.
- Surface tradeoffs when there are multiple viable approaches.

Do not silently invent:
- strategy logic or signal generation rules
- order routing behavior
- risk limits or position sizing rules
- data schema changes
- broker API behavior
- backtest assumptions

### 2. Simplicity First

Build the minimum solution that solves the current problem.

Avoid:
- speculative features
- premature abstractions
- unused configuration
- unnecessary frameworks
- broad error handling for impossible cases
- "future-proofing" without real future requirements

If the solution feels clever, make it clearer.

### 3. Reuse Before Rebuild

Prefer adapting existing code over creating new systems.

Default order:
1. Reuse existing behavior as-is.
2. Configure existing behavior via YAML.
3. Add a small adapter or wrapper.
4. Extract shared logic when duplication is real.
5. Introduce abstraction only when multiple real variants exist.
6. Build new code only when reuse would be harder to understand or maintain.

Do not copy large blocks of code when a shared function, module, interface, or adapter would be clearer.

### 4. Keep Changes Surgical

When editing existing code:
- Touch only files required for the task.
- Match existing style and conventions.
- Do not reformat unrelated code.
- Do not refactor adjacent code unless required.
- Remove only unused code created by your own change.
- Mention unrelated problems instead of fixing them opportunistically.

Every changed line should connect directly to the request.

### 5. Define Verifiable Success

For non-trivial work, write a short plan:

```md
## Plan
1. Change: ...
   Verify: ...
2. Change: ...
   Verify: ...
```

Good verification includes:
- a failing test that now passes
- existing tests still passing
- a manual command with expected output
- logs showing expected behavior
- backtest metric comparison (research vs. validation)
- broker dry-run or paper trading confirmation

A task is not done until it is verified.

### 6. Test Behavior, Not Implementation Details

When fixing bugs:
1. Reproduce with a test or clear manual steps.
2. Make the smallest fix.
3. Verify the fix.
4. Run nearby regression tests.

When refactoring:
1. Confirm current behavior.
2. Refactor in small steps.
3. Confirm behavior is unchanged.

Testing rules specific to this project (three tiers):

| Tier | May touch | Must not touch |
|---|---|---|
| **Unit** | `MockBroker(BaseBroker)`, in-memory data | Any real API, DB file, network |
| **Integration** | Real DuckDB in-memory instance, Zipline bundle on disk | Any broker API (live or paper) |
| **Paper** | Paper/sandbox broker API (Futu paper, Binance testnet) | Live (real-money) broker account |

- Unit tests mock at the ABC boundary (`MockBroker(BaseBroker)`) — never mock internal methods.
- Integration tests use a real DuckDB in-memory instance, never a mocked DB.
- Backtest engine tests: run both `VectorBTProEngine` and `ZiplineEngine` on the same fixture data and assert `BacktestResult` schema is identical.
- Paper tests connect to a paper/sandbox account only (Futu OpenD paper mode, Binance testnet). They verify `place_order` → `Fill` round-trips without spending real money. Mark with `@pytest.mark.paper`.
- No test ever touches a live (real-money) broker account or production exchange API.

### 7. Respect High-Risk Areas

These areas require extra caution in this codebase:

| Area | Risk |
|---|---|
| `execution/` — order placement | Real money, irreversible |
| `orchestration/` — live flows | Scheduled, runs without confirmation |
| `data/` — schema normalization | Silent schema drift breaks downstream |
| `research/backtest/` — look-ahead | Future leakage invalidates all results |
| `config/` — YAML resolution | Wrong name silently uses wrong plugin |

For high-risk changes:
- do not guess broker API behavior — verify against paper account first
- preserve auditability (log fills, position diffs)
- add edge-case tests; use `@pytest.mark.paper` tests for broker round-trips
- document assumptions
- prefer explicit code over clever abstractions

### 8. Use Abstraction Only When Earned

Use shared interfaces, polymorphism, or generic utilities when they simplify real repeated patterns.

In this codebase, abstraction is already earned at these seams:

| Extension point | ABC | Decorator |
|---|---|---|
| Data source | `BaseDataSource` | `@Registry.register_data_source` |
| Storage | `BaseStorage` | `@Registry.register_storage` |
| Feature | `BaseFeature` | `@Registry.register_feature` |
| Strategy | `BaseStrategy` | `@Registry.register_strategy` |
| Backtest engine | `BaseEngine` | `@Registry.register_engine` |
| Broker | `BaseBroker` | `@Registry.register_broker` |

New abstraction is only warranted when a second real implementation exists. One implementation → no abstraction.

### 9. Keep Documentation Close to Decisions

Update documentation when a change affects:
- architecture or layer boundaries
- setup or deployment
- YAML config keys or CLI commands
- public ABCs or their contracts
- data schemas
- workflow state transitions (research → validation → live)
- operational procedures or scheduling

Documentation sync rule:
- `ARCHITECTURE.md`, `WORKFLOWS.md`, `CODING_RULES.md`, and this file must stay in sync with code.
- If a change affects architecture, workflows, ABCs, naming, or layer responsibilities, update the matching doc in the same change.

### 10. Maintain Compatibility

Before changing public behavior, check for:
- existing callers of the ABC method being changed
- YAML config key renames (break existing config files)
- DuckDB schema changes (require migration)
- `SignalFrame` schema changes (break both engines)
- `BacktestResult` schema changes (break metrics.py and comparisons)

Prefer additive changes when compatibility matters.

---

## Project-Specific Rules

### Layer Boundaries

No layer imports from a layer to its right.

| Layer | May import from |
|---|---|
| `core/` | stdlib, third-party only |
| `data/` | `core/` |
| `research/` | `core/`, `data/` |
| `execution/` | `core/` |
| `orchestration/` | `core/`, `data/`, `research/`, `execution/` |
| `config/` | `core/`, registry only |

`execution/` never imports from `research/`. Strategy code never imports broker adapters.

### External Dependencies

No direct imports of `futu`, `binance-connector`, `vectorbt`, or `prefect` outside their adapter module. All external dependencies are behind an ABC.

### Local Python Environment

Always use the repo-local virtual environment at `QTradeSystematic/.venv`.
Install with `.venv/bin/pip`, and run Python, pytest, and tooling through `.venv/bin/...` so the project never depends on global packages.

### Polymorphism Over Conditionals

No `if asset_type == "crypto"` in business logic. Route via the registry or the router. Add a new plugin; do not add a branch.

### Dependency Injection

Concrete implementations are passed in — never instantiated inside a class that does not own them.

### Key Schemas

**OHLCV output** (all data sources must conform):
```
[date, symbol, open, high, low, close, volume]
```

**SignalFrame** (all strategies must produce):
```
[date, symbol, signal: int ∈ {-1, 0, 1}, weight: float ∈ [0, 1]]
```

**Monetary values:** always `Decimal`, never `float`.  
**Bar timestamps:** `date` for calendar-aligned bars; `datetime` only for tick-level data.

### Strategy Promotion Gate

A strategy is only promoted to live after passing both gates:

```
research.yaml  →  (metrics pass threshold)
validation.yaml  →  (Sharpe degrades < 30%)
live.yaml
```

Promotion is a code review, not a runtime decision. Config files are version-controlled.

### Error Handling at Boundaries

- Raise at system boundaries (API calls, DB writes, broker connections). Do not swallow exceptions.
- `BrokerError(message, order)` — callers handle retry logic.
- `DataSourceError(message, symbol, date_range)` — callers handle retry logic.
- No fallback logic inside adapters.

---

## Working Standard

Deliver clear, boring, verified code. Prefer small changes that are easy to review, easy to test, and easy to undo.

## Anti-Patterns

Avoid:
- rewriting working systems without need
- drive-by refactoring
- style churn
- speculative extensibility
- large unverified changes
- hidden assumptions
- copying code instead of sharing it
- abstractions with only one implementation
- adding dependencies for small problems
- changing behavior without tests or explicit verification
- `if asset_type == ...` branches in business logic
- importing from `omega`, `qsconnect`, `qsresearch`, or `qsautomate` — those are design references, not dependencies; rewrite everything from scratch
- direct imports of vendor SDKs (`futu-api`, `binance-connector`, `vectorbtpro`, `zipline`, `prefect`) outside their designated adapter module
- touching a live broker in tests
