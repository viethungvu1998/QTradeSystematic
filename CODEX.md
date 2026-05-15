# QTradeSystematic — Codex Context

Single self-contained reference for AI agents and contributors. All rules, architecture, workflows, and features are inlined here.

---

## Project Goal

Build **QTradeSystematic (QTS)** from scratch — a systematic trading platform covering stocks (Moomoo/Futu) and crypto (Binance), with a config-driven research → validation → live promotion pipeline.

**Key migrations from legacy reference systems (omega, QSConnect, QSResearch, QSAutomate):**
- pandas → Polars everywhere
- IB binary protocol → Futu OpenD gateway (Moomoo) + Binance REST/WS
- float → Decimal for all monetary values
- Zipline bundle kept for stocks; crypto stays in DuckDB only

**Critical rule:** `omega`, `qsconnect`, `qsresearch`, `qsautomate` are design references only — never imported. Every file is written from scratch.
**Environment rule:** all installs, tests, and scripts run from the repo-local virtual environment at `QTradeSystematic/.venv`.

---

## Environment Variables

Credentials are read from `.env` at the repo root. Load via `python-dotenv` or export before activating the venv. Never commit `.env`.

| Variable | Purpose |
|---|---|
| `BINANCE_DEMO_TRADING_API_KEY` | Binance testnet API key — used for paper tests and demo mode |
| `BINANCE_DEMO_TRADING_SECRET_KEY` | Binance testnet secret key — paired with the above |
| `BINANCE_TRADING_KEY` | Binance production API key — live mode only |
| `BINANCE_TRADING_SECRET_KEY` | Binance production secret key — live mode only |
| `BINANCE_TESTNET_API_KEY` | `testnet.binance.vision` key — paper tests only; generate at https://testnet.binance.vision |
| `BINANCE_TESTNET_SECRET_KEY` | `testnet.binance.vision` secret — paired with the above |

**Mode selection:** set `binance_mode: demo` or `binance_mode: live` under `brokers` in the YAML config. `demo` uses the testnet keys and base URL `https://testnet.binance.vision`; `live` uses the production keys. Both broker and data source adapters read these at `connect()` time.

**Install the execution extras before wiring real clients:**
```
.venv/bin/pip install -e ".[execution]"
```
This installs `binance-connector` and `futu-api`.

---

## Repo Structure

```
QTradeSystematic/
├── qts/
│   ├── core/           # Shared ABCs, models, plugin registry
│   ├── data/           # Data acquisition, storage, bundle ingestion
│   ├── research/       # Feature engineering + backtesting
│   ├── execution/      # Live order execution
│   ├── orchestration/  # Single Prefect flow + task library
│   └── config/         # YAML parsing + registry resolution
├── examples/
├── tests/
└── pyproject.toml
```

---

## Architecture

### `core/` — Foundation

No imports from any other layer.

| File | Responsibility |
|---|---|
| `instrument.py` | `Instrument(symbol, asset_type, exchange, currency)` — unified Stock + Crypto |
| `order.py` | `Order`, `Fill`, `OrderSide`, `OrderType`, `OrderStatus` — all monetary fields `Decimal` |
| `portfolio.py` | `Position`, `Portfolio` — all monetary values `Decimal` |
| `events.py` | Async event bus — ticks and order updates |
| `registry.py` | Plugin registry — maps string names to concrete classes |

---

### `data/` — Acquisition + Storage

```
data/
├── base.py
├── sources/
│   ├── fmp.py          # Stocks — Financial Modeling Prep
│   ├── yahoo.py        # Stocks — Yahoo Finance
│   └── binance.py      # Crypto — Binance REST + WebSocket
├── storage/
│   ├── base.py         # BaseStorage ABC
│   ├── duckdb.py       # Primary DB
│   └── parquet.py      # Local cache layer
├── bundles/
│   ├── base.py         # BaseBundleAdapter ABC
│   ├── local.py        # LocalBundleAdapter — filesystem Zipline bundle
│   └── zipline_ingest.py  # Ingest DuckDB tables → Zipline bundle
└── manager.py
```

`DataManager` routes by symbol convention (`BTC/USDT` → Binance, `AAPL` → FMP), normalises all output to `[date, symbol, open, high, low, close, volume]`, writes to DuckDB, and ingests stock data into a Zipline bundle for `ZiplineEngine`.

---

### `research/` — Features + Backtesting

```
research/
├── features/
│   ├── base.py              # BaseFeature ABC
│   ├── technical.py         # RSI, MACD, ATR, Bollinger — any OHLCV
│   ├── fundamentals.py      # P/E, EV/EBITDA — stocks only, no-op for crypto
│   ├── onchain.py           # NVT, active addresses — crypto only, no-op for stocks
│   └── forward_returns.py
├── strategies/
│   ├── base.py              # BaseStrategy ABC
│   ├── factor/              # ML factor model (XGBoost)
│   └── stat_arb/            # Pairs / cointegration
└── backtest/
    ├── base.py              # BaseEngine ABC · BacktestConfig · BacktestResult
    ├── engines/
    │   ├── vectorbtpro_engine.py   # "fast" — vectorised, full-history batch
    │   └── zipline_engine.py       # "normal" — bar-by-bar, strict look-ahead guard
    ├── simulation/
    │   ├── fills.py         # ImmediateFill | NextOpenFill | VWAPFill
    │   ├── slippage.py      # FixedSlippage | VolatilityScaledSlippage
    │   ├── commission.py    # PercentageCommission | PerTradeCommission
    │   └── calendar.py      # NYSE | HKEX | Crypto (24/7)
    └── metrics.py           # Sharpe, Sortino, CAGR, max drawdown, win rate
```

- `VectorBTProEngine` ("fast") reads from DuckDB/Polars directly for all asset types.
- `ZiplineEngine` ("normal") reads from Zipline bundle — stocks only.
- Both must produce identical `BacktestResult` schema.

---

### `execution/` — Live Trading

```
execution/
├── base.py             # BaseBroker ABC
├── brokers/
│   ├── moomoo.py       # Stocks — Futu OpenD gateway (futu-api)
│   └── binance.py      # Crypto — Binance REST + user data stream
├── router.py           # OrderRouter: dict[AssetType, BaseBroker] — dispatches concurrently
└── sync.py             # PositionSync: target weights → delta orders
```

`execution/` never imports from `research/`. Strategy code never imports broker adapters.

---

### `orchestration/` — Workflows

```
orchestration/
├── flow.py             # Single qts_flow — zero hardcoded parameters
└── tasks/
    ├── data_tasks.py       # download_ohlcv, download_fundamentals
    ├── research_tasks.py   # build_features, run_backtest
    └── execution_tasks.py  # sync_positions, execute_rebalance
```

One flow, one entry point. All behaviour is resolved from `BacktestConfig` at runtime. Nothing is hardcoded in `flow.py`.

---

### `config/` — Resolution

`Config.build()` parses YAML and resolves all string names through the registry.

---

## Plugin Registry

```python
# Register
@Registry.register_broker("moomoo")
class MoomooBroker(BaseBroker): ...

# Resolve
broker = Registry.get_broker("moomoo")()
```

| Extension point | ABC | Decorator |
|---|---|---|
| Data source | `BaseDataSource` | `@Registry.register_data_source` |
| Storage | `BaseStorage` | `@Registry.register_storage` |
| Feature | `BaseFeature` | `@Registry.register_feature` |
| Strategy | `BaseStrategy` | `@Registry.register_strategy` |
| Backtest engine | `BaseEngine` | `@Registry.register_engine` |
| Broker | `BaseBroker` | `@Registry.register_broker` |

---

## Data Flow

```
FMP / Yahoo ──┐                          ┌──► Zipline bundle ──► ZiplineEngine ("normal")
              ├──► DataManager ──► DuckDB─┤
Binance ───────┘                          └──► FeaturePipeline ──► Strategy.generate_signals()
                                                                           │
                                                              VectorBTProEngine ("fast")
                                                                           │
                                                                    BacktestResult
                                                                           │
                                                               OrderRouter.execute()
                                                                │               │
                                                          MoomooBroker    BinanceBroker
```

---

## Layer Boundaries

| Layer | May import from |
|---|---|
| `core/` | stdlib, third-party only |
| `data/` | `core/` |
| `research/` | `core/`, `data/` |
| `execution/` | `core/` |
| `orchestration/` | `core/`, `data/`, `research/`, `execution/` |
| `config/` | `core/`, registry only |

No layer imports from a layer to its right. `execution/` never imports from `research/`.

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

## Key Schemas

**OHLCV** (all data sources must conform):
```
[date, symbol, open, high, low, close, volume]
```

**SignalFrame** (all strategies must produce):
```
[date, symbol, signal: int ∈ {-1, 0, 1}, weight: float ∈ [0, 1]]
```

**Monetary values:** always `Decimal`, never `float`.
**Bar timestamps:** `date` for calendar-aligned bars; `datetime` only for tick-level data.

---

## YAML Config Schema

```yaml
# ── Workflow ───────────────────────────────────────────────
workflow: research            # research | validation | live

# ── Universe ───────────────────────────────────────────────
asset_types: [stock, crypto]
universe:
  stock:  [AAPL, GOOGL, MSFT]
  crypto: [BTC/USDT, ETH/USDT]
start_date: "2018-01-01"
end_date:   "2024-12-31"
initial_capital: 100000

# ── Data ───────────────────────────────────────────────────
data_sources:
  stock:  fmp               # fmp | yahoo
  crypto: binance
storage: duckdb

# ── Features ───────────────────────────────────────────────
features:
  technical:   true
  fundamental: true         # stocks only — auto no-op for crypto
  onchain:     false        # crypto only — auto no-op for stocks
  forward_returns:
    periods: [1, 5, 20]

# ── Strategy ───────────────────────────────────────────────
strategy:
  type: factor              # factor | stat_arb
  params:
    n_factors: 5
    lookback:  60

# ── Backtest ───────────────────────────────────────────────
backtest_engine: fast       # fast (vectorbtpro) | normal (zipline)
fill_model:      next_open  # immediate | next_open | vwap
slippage_model:  volatility_scaled
commission:
  model: percentage
  rate:  0.001
calendar: nyse              # nyse | hkex | crypto

# ── Brokers (live only) ────────────────────────────────────
brokers:
  stock:  moomoo
  crypto: binance
  binance_mode: demo       # demo (testnet) | live (production)

# ── Schedule (live only) ───────────────────────────────────
schedule:
  stock:  "0 16 * * 1-5"
  crypto: "0 */4 * * *"

# ── Promotion gate (validation only) ──────────────────────
promotion_gate:
  max_sharpe_degradation: 0.30
```

---

## Workflows

### Research
**Engine:** `VectorBTProEngine` — full history vectorised.

```
1. DataManager.get_ohlcv(config.universe, config.start_date, config.end_date)
2. FeaturePipeline.fit_transform(df)
3. Strategy.generate_signals(df) → SignalFrame
4. VectorBTProEngine.run(strategy, data, config)
5. BacktestResult → metrics.py
6. (optional) Grid search / Optuna sweep over steps 3–5
```

### Validation
**Engine:** `ZiplineEngine` — bar-by-bar, look-ahead prevented.

```
1. Load Zipline bundle (from same DuckDB — no re-download)
2. FeaturePipeline.fit_transform(df)
3. Strategy.generate_signals(visible_data_only)
4. ZiplineEngine.run(strategy, data, config)
5. BacktestResult vs research: Sharpe degradation > max_sharpe_degradation → reject
```

### Live
**Orchestration:** `qts_flow` scheduled per `config.schedule`.

```
1. [parallel] download_ohlcv, download_fundamentals → DuckDB
2. build_features(df, config.features)
3. run_backtest(strategy, data, config) → final-bar target weights
4. sync_positions(config.brokers) → delta orders
5. execute_rebalance(orders, config.brokers)
     stock  → MoomooBroker
     crypto → BinanceBroker
6. Log fills → positions table
```

### Promotion Gate
```
research.yaml → metrics pass threshold
      ↓
validation.yaml → Sharpe degradation < 30%
      ↓
live.yaml → scheduled per config.schedule
```
Promotion is a code review, not a runtime decision. Config files are version-controlled.

---

## Coding Rules

1. **All code is original.** Never import from `omega`, `qsconnect`, `qsresearch`, `qsautomate`.
2. **Every vendor SDK behind an ABC.** No direct imports of `futu-api`, `binance-connector`, `vectorbtpro`, `zipline`, `prefect` outside their adapter module.
3. **Polymorphism over conditionals.** No `if asset_type == "crypto"` in business logic — route via registry or router.
4. **Dependency injection.** Concrete implementations are passed in — never instantiated inside a class that does not own them.
5. **Signal-based strategy interface.** Strategies produce `SignalFrame`; engines consume it. Strategy code is engine-agnostic.
6. **No hardcoded parameters in orchestration.** `flow.py` reads everything from `BacktestConfig`. If a value is not in the YAML schema, it does not belong in the flow.
7. **Use the local environment.** Install and run everything through `QTradeSystematic/.venv`; do not rely on global Python packages.

### Adding a Plugin
```python
# 1. Inherit the ABC
class MyBroker(BaseBroker):
    async def connect(self) -> None: ...
    async def place_order(self, order: Order) -> Fill: ...

# 2. Register with a string key
@Registry.register_broker("mybroker")
class MyBroker(BaseBroker): ...

# 3. Reference the key in YAML
#    brokers:
#      stock: mybroker
```

### Type Annotations
- All public functions/methods fully annotated.
- `pl.DataFrame` (Polars) is the standard DataFrame type.
- `Decimal` for all monetary values — never `float`.
- `date` for calendar-aligned bars; `datetime` only for tick-level data.

### Error Handling
- Raise at system boundaries (API calls, DB writes, broker connections).
- `BrokerError(message, order)` — callers handle retry.
- `DataSourceError(message, symbol, date_range)` — callers handle retry.
- No fallback logic inside adapters.

### Naming
| Thing | Convention | Example |
|---|---|---|
| ABC | `Base` prefix | `BaseBroker`, `BaseEngine` |
| Concrete plugin | Vendor + noun | `MoomooBroker`, `VectorBTProEngine` |
| Registry key | lowercase, no hyphens | `"moomoo"`, `"fast"`, `"fmp"` |
| Prefect task | verb phrase | `download_ohlcv`, `run_backtest` |
| Config keys | snake_case | `backtest_engine`, `initial_capital` |

### Comments
No comments unless the reason is non-obvious from the code. One short line max. Never describe what the code does — only why.

---

## Testing

Three tiers with hard boundaries:

| Tier | May touch | Must not touch |
|---|---|---|
| **Unit (U)** | `MockBroker(BaseBroker)`, in-memory data | Any real API, DB file, network |
| **Integration (I)** | Real DuckDB in-memory, Zipline bundle on disk | Any broker API (live or paper) |
| **Paper (P)** | Futu OpenD paper mode, Binance testnet | Live (real-money) accounts |

- Mock at the ABC boundary only — never mock internal methods.
- Integration tests use real in-memory DuckDB — never a mocked DB.
- Engine tests: run both engines on the same fixture; assert `BacktestResult` schema is identical.
- Paper tests: `place_order` → `Fill` round-trip without real money. Mark `@pytest.mark.paper`.
- No test ever touches a live broker account or production exchange API.

---

## High-Risk Areas

| Area | Risk |
|---|---|
| `execution/` — order placement | Real money, irreversible |
| `orchestration/` — live flows | Scheduled, runs without confirmation |
| `data/` — schema normalization | Silent schema drift breaks downstream |
| `research/backtest/` — look-ahead | Future leakage invalidates all results |
| `config/` — YAML resolution | Wrong name silently uses wrong plugin |

For high-risk changes: verify broker behavior against paper account first, preserve fill audit logs, add edge-case tests, prefer explicit code over clever abstractions.

---

## Anti-Patterns

- Rewriting working systems without need
- `if asset_type == ...` branches in business logic
- Importing from `omega`, `qsconnect`, `qsresearch`, `qsautomate`
- Direct imports of `futu-api`, `binance-connector`, `vectorbtpro`, `zipline`, `prefect` outside adapter modules
- Hardcoding any value in `flow.py`
- Abstractions with only one implementation
- Copying code instead of sharing it
- Touching a live broker in tests
- Swallowing exceptions inside adapters

---

## Feature Build Order (38 features)

### Phase 1 — Core (Unit tests)

**F-01 · Instrument model**
`core/instrument.py` — `AssetType(Stock | Crypto)`; `Instrument(symbol, asset_type, exchange, currency)`

**F-02 · Order and Fill models**
`core/order.py` — `OrderSide`, `OrderType`, `OrderStatus`; `Order`, `Fill`; all monetary fields `Decimal`

**F-03 · Portfolio models**
`core/portfolio.py` — `Position`, `Portfolio`; all monetary values `Decimal`

**F-04 · Plugin Registry**
`core/registry.py` — `@Registry.register_*` decorators; `Registry.get_*()` for all extension points; `RegistryError` on unknown name

**F-05 · Async Event Bus**
`core/events.py` — typed async event bus for ticks and order updates

---

### Phase 2 — Data (Unit + Integration tests)

**F-06 · Data ABCs**
`data/base.py` (`BaseDataSource`), `data/storage/base.py` (`BaseStorage`), `data/bundles/base.py` (`BaseBundleAdapter`)

**F-07 · Parquet storage**
`data/storage/parquet.py` — `write / read / exists / list_keys`

**F-08 · DuckDB storage**
`data/storage/duckdb.py` — `write / read / append / query`; append must not produce duplicates

**F-09 · FMP data source**
`data/sources/fmp.py` — `get_ohlcv`, `get_fundamentals`; raises `DataSourceError` on invalid symbol

**F-10 · Yahoo data source**
`data/sources/yahoo.py` — `get_ohlcv`

**F-11 · Binance data source**
`data/sources/binance.py` — `get_ohlcv`, `stream_ticks` → `Tick` objects using `binance-connector`; reads `BINANCE_DEMO_TRADING_API_KEY` / `BINANCE_DEMO_TRADING_SECRET_KEY` for testnet, production keys for live

**F-12 · DataManager — stock routing**
`data/manager.py` — routes stock symbols → FMP/Yahoo; writes DuckDB; caches Parquet; second call served from cache

**F-13 · Zipline bundle ingest**
`data/bundles/local.py` + `data/bundles/zipline_ingest.py` — DuckDB → Zipline bundle on disk

**F-14 · DataManager — crypto routing**
`data/manager.py` (extended) — routes crypto → Binance; skips Zipline ingest; no bundle produced

---

### Phase 3 — Research (Unit + Integration tests)

**F-15 · BaseFeature ABC**
`research/features/base.py` — `fit_transform(df) → df`; contract: append columns, never drop

**F-16 · Technical features**
`research/features/technical.py` — RSI, MACD, ATR, Bollinger; any OHLCV asset type

**F-17 · Fundamental features**
`research/features/fundamentals.py` — P/E, EV/EBITDA; stocks only, no-op for crypto

**F-18 · Onchain features**
`research/features/onchain.py` — NVT, active addresses; crypto only, no-op for stocks

**F-19 · Forward returns**
`research/features/forward_returns.py` — configurable periods; last `n` rows are NaN for each period `n`

**F-20 · BaseStrategy ABC + SignalFrame validation**
`research/strategies/base.py` — validator: `signal ∈ {-1, 0, 1}`, `weight ∈ [0, 1]`; raises `ValueError` on violation

**F-21 · Factor strategy (XGBoost)**
`research/strategies/factor/` — `generate_signals(df) → SignalFrame`

**F-22 · Stat-arb strategy**
`research/strategies/stat_arb/` — pairs/cointegration; spread crossing → opposing signals

**F-23 · Simulation models**
`research/backtest/simulation/fills.py`, `slippage.py`, `commission.py`, `calendar.py`
- `NextOpenFill` fills at next bar open
- `VolatilityScaledSlippage` scales with ATR
- `NYSE` calendar excludes holidays; `Crypto` includes weekends

**F-24 · Metrics**
`research/backtest/metrics.py` — Sharpe, Sortino, CAGR, max drawdown, win rate

**F-25 · BacktestConfig + BacktestResult**
`research/backtest/base.py` — `BaseEngine` ABC; `BacktestConfig` (typed dataclass); `BacktestResult` (frozen dataclass); identical schema regardless of engine

**F-26 · VectorBTProEngine**
`research/backtest/engines/vectorbtpro_engine.py` — reads Polars DataFrame from DuckDB; full-history vectorised

**F-27 · ZiplineEngine**
`research/backtest/engines/zipline_engine.py` — reads Zipline bundle; bar-by-bar look-ahead guard; `BacktestResult` schema identical to VectorBTProEngine; Sharpe values within 30% of each other on same fixture

---

### Phase 4 — Config (Unit tests)

**F-28 · YAML loader + BacktestConfig schema**
`config/loader.py` — parse YAML → typed `BacktestConfig`; raise `ConfigError` on missing required keys or unknown top-level keys

**F-29 · Registry resolution**
`config/builder.py` — `Config.build()` resolves every string registry key → concrete instance; `RegistryError` on unregistered key

---

### Phase 5 — Execution (Unit + Paper tests)

**F-30 · BaseBroker ABC**
`execution/base.py` — full async contract; `MockBroker` returns `Fill` with `Decimal` price and quantity

**F-31 · MoomooBroker**
`execution/brokers/moomoo.py` — Futu OpenD gateway via `futu-api`
Paper verify: connect Futu OpenD paper mode; `place_order(AAPL, qty=1, LIMIT)` → `Fill` with `Decimal` fields

**F-32 · BinanceBroker**
`execution/brokers/binance.py` — Binance REST + user data stream via `binance-connector`; reads env vars at `connect()` time; `binance_mode: demo` → testnet (`BINANCE_DEMO_TRADING_API_KEY` + `BINANCE_DEMO_TRADING_SECRET_KEY`), `binance_mode: live` → production (`BINANCE_TRADING_KEY` + `BINANCE_TRADING_SECRET_KEY`)
Paper verify: connect Binance testnet; `place_order(BTC/USDT, qty=0.001, LIMIT)` → `Fill` with `Decimal` fields

**F-33 · OrderRouter**
`execution/router.py` — `dict[AssetType, BaseBroker]`; dispatches concurrently; stock order → only stock broker, crypto → only crypto broker

**F-34 · PositionSync**
`execution/sync.py` — target weights → delta `Order` list; no order emitted for unchanged weights; all monetary values `Decimal`

---

### Phase 6 — Orchestration (Unit + Integration + Paper tests)

**F-35 · Task library**
`orchestration/tasks/` — `download_ohlcv`, `download_fundamentals`, `build_features`, `run_backtest`, `sync_positions`, `execute_rebalance`
Rule: every task accepts `BacktestConfig` — no literal values inside task bodies; retries on `DataSourceError` and `BrokerError`

**F-36 · qts_flow — research**
`orchestration/flow.py` — data → features → `VectorBTProEngine` → `BacktestResult`
- stock run: `BacktestResult` produced + Zipline bundle written
- crypto run: `BacktestResult` produced + no Zipline bundle

**F-37 · qts_flow — validation**
`orchestration/flow.py` extended — Zipline bundle + promotion gate
- `ZiplineEngine` used when `backtest_engine: normal`
- Sharpe degradation > threshold → reject

**F-38 · qts_flow — live (paper)**
`orchestration/flow.py` extended — `sync_positions` → `execute_rebalance` → paper broker
- Cron schedule registered from `config.schedule` at deploy time
- Paper verify (`@pytest.mark.paper`): full cycle → DuckDB positions table updated with fills

---

## Feature Summary

| Phase | Features | Test tiers |
|---|---|---|
| 1 — Core | F-01 → F-05 | U |
| 2 — Data | F-06 → F-14 | U, I |
| 3 — Research | F-15 → F-27 | U, I |
| 4 — Config | F-28 → F-29 | U |
| 5 — Execution | F-30 → F-34 | U, P |
| 6 — Orchestration | F-35 → F-38 | U, I, P |

**38 features total. Each is independently buildable. No feature is done until its verify step passes.**
