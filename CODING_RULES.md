# CODING RULES

## Core Principles

1. **All code is original to this codebase.** `omega`, `qsconnect`, `qsresearch`, and `qsautomate` are design references only — they are not dependencies and must never appear in an import statement. Every pattern drawn from those systems is rewritten here from scratch.
2. **Every external vendor dependency is behind an ABC.** No direct imports of `futu-api`, `binance-connector`, `vectorbtpro`, `zipline`, or `prefect` outside their adapter module.
3. **Polymorphism over conditionals.** No `if asset_type == "crypto"` in business logic — route via the registry or the router.
4. **Dependency injection.** Concrete implementations are passed in — never instantiated inside a class that does not own them.
5. **Signal-based strategy interface.** Strategies produce a `SignalFrame`; engines consume it. Strategy code is engine-agnostic.
6. **No hardcoded parameters in orchestration.** `flow.py` contains no literal asset types, engine names, commission rates, schedules, or broker names. Every axis of behaviour is read from `BacktestConfig` built from YAML. If a value is not in the YAML schema, it does not belong in the flow.
7. **Use the repo-local environment.** Install dependencies into `QTradeSystematic/.venv` and run Python, pytest, and tooling from that environment instead of the global interpreter.
8. **Credentials come from environment variables, not code or config files.** Adapters read credentials (API keys, secrets) from env vars at `connect()` time using `python-dotenv` or OS environment. Never hardcode keys or embed them in YAML. The `.env` file at the repo root is the canonical credential store and must never be committed.

---

## Adding a Plugin

Every pluggable component follows the same three-step pattern:

```python
# 1. Inherit the ABC
class MyBroker(BaseBroker):
    async def connect(self) -> None: ...
    async def get_positions(self) -> list[Position]: ...
    async def place_order(self, order: Order) -> Fill: ...
    async def cancel_order(self, order_id: str) -> None: ...
    async def get_account_value(self) -> Decimal: ...

# 2. Register with a string key
@Registry.register_broker("mybroker")
class MyBroker(BaseBroker): ...

# 3. Reference the key in YAML
#    brokers:
#      stock: mybroker
```

Same pattern for data sources, storage, features, strategies, and engines.

---

## ABC Contracts

### `BaseDataSource`
```python
async def get_ohlcv(self, symbol: str, start: date, end: date, interval: str) -> pl.DataFrame
async def get_fundamentals(self, symbol: str) -> pl.DataFrame  # raise NotImplementedError if unsupported
async def stream_ticks(self, symbols: list[str]) -> AsyncIterator[Tick]
```
Output schema: `[date, symbol, open, high, low, close, volume]` — all sources must conform.

### `BaseFeature`
```python
def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame
```
Appends new columns. Never drops existing columns.

### `BaseStrategy`
```python
def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame
```
Output schema: `[date, symbol, signal: int ∈ {-1, 0, 1}, weight: float ∈ [0, 1]]`.

### `BaseEngine`
```python
def run(self, strategy: BaseStrategy, data: pl.DataFrame, config: BacktestConfig) -> BacktestResult
```

### `BaseBroker`
```python
async def connect(self) -> None
async def disconnect(self) -> None
async def get_positions(self) -> list[Position]
async def place_order(self, order: Order) -> Fill
async def cancel_order(self, order_id: str) -> None
async def get_account_value(self) -> Decimal
```

---

## Type Annotations

- All public functions and methods are fully annotated.
- `pl.DataFrame` (Polars) is the standard DataFrame type throughout.
- `Decimal` for all monetary values — never `float`.
- `date` for calendar-aligned bar timestamps; `datetime` only for tick-level data.

## Local Environment

- Create the environment with `python3 -m venv .venv` from the repo root.
- Install dependencies with `.venv/bin/pip install -e .[dev]` or a narrower extra set.
- Run commands with `.venv/bin/python`, `.venv/bin/pytest`, and `.venv/bin/ruff`.
- Do not rely on globally installed Python packages for development or verification.

---

## Error Handling

- Raise at system boundaries (API calls, DB writes, broker connections). Do not swallow exceptions.
- Broker adapters raise `BrokerError(message, order)` — callers handle retry.
- Data sources raise `DataSourceError(message, symbol, date_range)`.
- No fallback logic inside adapters — callers decide how to handle failures.

---

## Module Boundaries

| Layer | May import from |
|---|---|
| `core/` | stdlib, third-party only |
| `data/` | `core/` |
| `research/` | `core/`, `data/` |
| `execution/` | `core/` |
| `orchestration/` | `core/`, `data/`, `research/`, `execution/` |
| `config/` | `core/`, registry only |

`execution/` never imports from `research/`. No layer imports from a layer to its right.

---

## Naming Conventions

| Thing | Convention | Example |
|---|---|---|
| ABC | `Base` prefix | `BaseBroker`, `BaseEngine` |
| Concrete plugin | Vendor + noun | `MoomooBroker`, `VectorBTProEngine`, `ZiplineEngine` |
| Registry key | lowercase, no hyphens | `"moomoo"`, `"fast"`, `"fmp"` |
| Prefect task | verb phrase | `download_ohlcv`, `run_backtest` |
| Prefect flow | `qts_flow` | single flow, config-driven |
| Config keys | snake_case | `backtest_engine`, `initial_capital` |

---

## Comments

No comments unless the reason is non-obvious from the code. One short line max. Never describe what the code does — only why.

---

## Testing

Three tiers with hard boundaries:

| Tier | May touch | Must not touch |
|---|---|---|
| **Unit** | `MockBroker(BaseBroker)`, in-memory data | Any real API, DB file, network |
| **Integration** | Real DuckDB in-memory, Zipline bundle on disk | Any broker API (live or paper) |
| **Paper** | Futu OpenD paper mode, Binance testnet | Live (real-money) accounts |

- Mock at the ABC boundary only — never mock internal methods.
- Integration tests use a real in-memory DuckDB — never a mocked DB.
- Engine tests: run both `VectorBTProEngine` and `ZiplineEngine` on the same fixture; assert `BacktestResult` schema is identical.
- Paper tests verify `place_order` → `Fill` round-trips without real money. Mark `@pytest.mark.paper`.
- No test ever touches a live broker account or production exchange API.
