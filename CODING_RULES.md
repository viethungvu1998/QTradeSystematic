# CODING RULES

## Core Principles

1. **All code is original to this codebase.** `omega`, `qsconnect`, `qsresearch`, and `qsautomate` are design references only — they are not dependencies and must never appear in an import statement. Every pattern drawn from those systems is rewritten here from scratch.
2. **Every external vendor dependency is behind an ABC.** No direct imports of `futu-api`, `binance-connector`, `vectorbtpro`, `zipline`, or `prefect` outside their adapter module.
3. **Polymorphism over conditionals.** No `if asset_type == "crypto"` in business logic — route via the registry or the router. Asset type is always derived from `AssetType.from_symbol(symbol)` and used as a routing key; never branch on it inline.
4. **Dependency injection.** Concrete implementations are passed in — never instantiated inside a class that does not own them.
5. **Signal-based strategy interface.** `BaseStrategy.generate_signals(data)` is the public strategy seam. Strategies produce a standard `SignalFrame`; engines consume it. Strategy code is engine-agnostic.
6. **Engines own simulation.** `BaseEngine.run(strategy, data, config, *, pipeline=None, ohlcv=None)` is the only supported backtest entrypoint. Do not add strategy-specific runners under `research/backtest/`, and do not expose alternate public backtest APIs from strategy packages.
7. **No hardcoded parameters in orchestration or config resolution.** `flow.py` contains no literal asset types, engine names, commission rates, schedules, or broker names. Registry-backed seams must not be bypassed with hardcoded constructors when the registry already exists. If a value is not in the YAML schema or a registered component, it does not belong in the flow.
8. **Use the repo-local environment.** Install dependencies into `QTradeSystematic/.venv` and run Python, pytest, and tooling from that environment instead of the global interpreter.
9. **Credentials come from environment variables, not code or config files.** Adapters read credentials (API keys, secrets) from env vars at `connect()` time using `python-dotenv` or OS environment. Never hardcode keys or embed them in YAML. The `.env` file at the repo root is the canonical credential store and must never be committed.
10. **Before adding a new class/module/layer, answer:**
-  Who else calls this besides the current caller? (reuse check)
- What breaks at test time if this layer doesn't exist? (testability check)  
- Is there a data shape change happening here? (translation check)
If all three answers are "nothing/nobody/no" → implement inline, no new layer.

---

## Symbol Convention

Asset type is inferred from symbol format by `AssetType.from_symbol()` — no other code should infer it:

| Symbol format | `AssetType` | Examples |
|---|---|---|
| Starts with `PERP:` | `CRYPTO_FUTURES` | `PERP:ETH/USDT` |
| Starts with `VNF:` | `VN_FUTURES` | `VNF:VN30F2606` |
| Starts with `VNW:` | `VN_WARRANT` | `VNW:CVNM2511` |
| Contains `/` | `CRYPTO` | `BTC/USDT` |
| Starts with `VN:` | `VN_STOCK` | `VN:VNM`, `VN:VIC` |
| Starts with `CMX:` | `COMMODITY` | `CMX:CL`, `CMX:GC` |
| Bare alphabetic | `STOCK` | `AAPL`, `MSFT` |

Rules:
- Always pass raw symbol strings through `AssetType.from_symbol()` — never pattern-match against them elsewhere.
- VN stock symbols **must** carry the `VN:` prefix in all config files and code.
- `COMMODITY` is reserved. Symbols with `CMX:` prefix are silently skipped by `DataManager` until a source is registered.
- Currency defaults: `STOCK` → `USD`; `VN_STOCK` → `VND`; `COMMODITY` → `USD`; `CRYPTO` → quote currency (right of `/`). Never hardcode `"USD"` for non-`STOCK` types.

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

For indicator plugins specifically, the registry key is the indicator name referenced in the YAML `features.indicators` list:

```python
# 1. Inherit BaseFeature (no separate BaseIndicator ABC needed)
class RSIFeature(BaseFeature):
    def __init__(self, periods: list[int] = (14,)) -> None:
        self.periods = list(periods)

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df
        for p in self.periods:
            result = result.with_columns(
                _rsi_expr(p).alias(f"rsi_{p}")
            )
        return self._validate_append_only(df, result)

# 2. Register with a string key matching the YAML name
@Registry.register_feature("rsi")
class RSIFeature(BaseFeature): ...

# 3. Reference in YAML
#    features:
#      indicators:
#        - name: rsi
#          params:
#            periods: [14, 28, 63]
```

`Config.build()` iterates `features.indicators`, calls `Registry.get_feature(name)(**params)` for each entry, and appends the resulting instance to `FeaturePipeline`. An unregistered name raises `RegistryError` at build time.

---

## ABC Contracts

### `BaseDataSource`
```python
CAPABILITIES: frozenset[DataType]   # declare supported DataTypes; DataManager reads this at init

async def fetch(self, data_type: DataType, symbol: str, **kwargs) -> pl.DataFrame
async def stream_ticks(self, symbols: list[str]) -> AsyncIterator[Tick]

# backward-compatible wrappers — delegate to fetch():
async def get_ohlcv(self, symbol: str, start: date, end: date, interval: str) -> pl.DataFrame
async def get_fundamentals(self, symbol: str) -> pl.DataFrame
```
- `fetch()` is the primary interface. `get_ohlcv` / `get_fundamentals` are thin wrappers for backward compatibility.
- `CAPABILITIES` must be declared as a class attribute. `DataManager` builds its `(AssetType, DataType) → source` map from it at init — do not override `supports()`.
- Output schema for each `DataType` is defined in `data/_schemas.py`.
- OHLCV schema: `[date, symbol, open, high, low, close, volume]` — all sources must conform for `DataType.OHLCV`.
- VN futures intraday storage schema: `[bar_time, date, symbol, interval, open, high, low, close, volume]`. Daily futures stay on the standard OHLCV schema.

### `BaseFeature`
```python
def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame
```
Appends new columns. Never drops existing columns. Input is pre-processed OHLCV (already deduped, filled, eps-replaced, OHLC-consistent). Each indicator implementation uses `.over("symbol")` for per-symbol computation — no asset-type branching inside the feature code.

### `BaseStrategy`
```python
def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame
```
Output schema: `[date, symbol, signal: int ∈ {-1, 0, 1}, weight: float ∈ [0, 1]]`.
- Treat `BaseStrategy` as the single public strategy contract.
- Shared helpers such as `validate_signal_frame()` and `empty_signal_frame()` belong on the base class, not on ad hoc concrete strategies.
- Concrete strategies may add family-specific internals, but they still return the standard signal frame.

### `BaseEngine`
```python
def run(
    self,
    strategy: BaseStrategy,
    data: pl.DataFrame,
    config: BacktestConfig,
    *,
    pipeline=None,
    ohlcv: pl.DataFrame | None = None,
) -> BacktestResult
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

Strategy architecture rules:
- Each concrete strategy family under `qts/research/strategies/` uses the same spine: `base.py` for the family interface, `factories.py` for registry side effects, and one concrete strategy per descriptive module.
- Do not add compatibility import shims such as `model.py`, `strategy.py`, or `base_model.py`; concrete strategy logic belongs in descriptive modules such as `rank.py`, `classification.py`, `mean_reversion.py`, or `quantamental.py`.
- Learning-based model objects expose `fit()` / `predict()` only. They do not implement `generate_signals()`; the strategy class owns signal generation.
- Strategy packages do not own data loading. Data assembly and source-specific fetching belong in `qts/data/`, while strategies consume normalized frames.
- Do not create empty family scaffolding until a concrete strategy exists.
- `research/backtest/` holds shared engine/runtime code only. Diagnostics may live in strategy packages, but simulation orchestration stays in engines.

---

## Naming Conventions

| Thing | Convention | Example |
|---|---|---|
| ABC | `Base` prefix | `BaseBroker`, `BaseEngine` |
| Concrete plugin | Vendor + noun | `MoomooBroker`, `VectorBTProEngine`, `ZiplineReloadedEngine` |
| Registry key | lowercase, no hyphens | `"moomoo"`, `"vectorbt"`, `"fmp"` |
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
- Engine tests: run both `VectorBTProEngine` and `ZiplineReloadedEngine` on the same fixture; assert `BacktestResult` schema is identical.
- Paper tests verify `place_order` → `Fill` round-trips without real money. Mark `@pytest.mark.paper`.
- No test ever touches a live broker account or production exchange API.
