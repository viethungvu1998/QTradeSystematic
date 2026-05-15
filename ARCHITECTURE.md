# ARCHITECTURE

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

## Layers

### `core/` — Foundation

No imports from any other layer. Everything else imports from here.

| File | Responsibility |
|---|---|
| `instrument.py` | `Instrument(symbol, asset_type, exchange, currency)` — unified Stock + Crypto model |
| `order.py` | `Order`, `Fill`, `OrderSide`, `OrderType` |
| `portfolio.py` | `Position`, `Portfolio` |
| `events.py` | Async event bus — real-time ticks and order updates |
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

`DataManager` routes by symbol convention (`BTC/USDT` → Binance, `AAPL` → FMP), normalises all output to `[date, symbol, open, high, low, close, volume]`, writes to DuckDB, and ingests stock data into a Zipline bundle for the `ZiplineEngine`.

---

### `research/` — Features + Backtesting

```
research/
├── features/
│   ├── base.py              # BaseFeature ABC: fit_transform(df) → df
│   ├── technical.py         # RSI, MACD, ATR, Bollinger — any OHLCV
│   ├── fundamentals.py      # P/E, EV/EBITDA — stocks only, no-op for crypto
│   ├── onchain.py           # NVT, active addresses — crypto only
│   └── forward_returns.py
├── strategies/
│   ├── base.py              # BaseStrategy ABC: generate_signals(df) → SignalFrame
│   ├── factor/              # ML factor model (XGBoost)
│   └── stat_arb/            # Pairs / cointegration
└── backtest/
    ├── base.py              # BaseEngine ABC · BacktestConfig · BacktestResult
    ├── engines/
    │   ├── vectorbtpro_engine.py   # "fast"   — vectorised, full-history batch
    │   └── zipline_engine.py       # "normal" — bar-by-bar, strict look-ahead guard
    ├── simulation/
    │   ├── fills.py         # ImmediateFill | NextOpenFill | VWAPFill
    │   ├── slippage.py      # FixedSlippage | VolatilityScaledSlippage
    │   ├── commission.py    # PercentageCommission | PerTradeCommission
    │   └── calendar.py      # NYSE | HKEX | Crypto (24/7)
    └── metrics.py           # Sharpe, Sortino, CAGR, max drawdown, win rate
```

`ZiplineEngine` reads from the bundle (stocks only). `VectorBTProEngine` reads from DuckDB/Polars directly for all asset types.

---

### `execution/` — Live Trading

```
execution/
├── base.py             # BaseBroker ABC
├── brokers/
│   ├── moomoo.py       # Stocks — Futu OpenD gateway (futu-api)
│   └── binance.py      # Crypto — Binance REST + user data stream (binance-connector)
├── router.py           # OrderRouter: dict[AssetType, BaseBroker] — dispatches concurrently
└── sync.py             # PositionSync: target weights → delta orders
```

**Install execution extras before using live brokers:**
```
.venv/bin/pip install -e ".[execution]"
```

**Binance mode:** resolved from `brokers.binance_mode` in YAML. Adapters read credentials from env at `connect()` time — never stored in config files.

| YAML `binance_mode` | Env vars read | Base URL |
|---|---|---|
| `demo` | `BINANCE_DEMO_TRADING_API_KEY`, `BINANCE_DEMO_TRADING_SECRET_KEY` | `https://testnet.binance.vision` |
| `live` | `BINANCE_TRADING_KEY`, `BINANCE_TRADING_SECRET_KEY` | Binance production |

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

One flow, one entry point. All behaviour — asset types, engine, brokers, schedule, commission — is resolved from `BacktestConfig` at runtime. Nothing is hardcoded in `flow.py`.

---

### `config/` — Resolution

`Config.build()` parses YAML and resolves all string names through the registry. Full config schema:

```yaml
# ── Workflow ───────────────────────────────────────────────
workflow: research            # research | validation | live

# ── Universe ───────────────────────────────────────────────
asset_types: [stock, crypto]  # any combination
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
slippage_model:  volatility_scaled  # fixed | volatility_scaled
commission:
  model: percentage         # percentage | per_trade
  rate:  0.001
calendar: nyse              # nyse | hkex | crypto

# ── Brokers (live only) ────────────────────────────────────
brokers:
  stock:  moomoo
  crypto: binance
  binance_mode: demo       # demo (testnet) | live (production)

# ── Schedule (live only) ───────────────────────────────────
schedule:
  stock:  "0 16 * * 1-5"   # daily after NYSE close
  crypto: "0 */4 * * *"    # every 4 hours

# ── Promotion gate (validation only) ──────────────────────
promotion_gate:
  max_sharpe_degradation: 0.30
```

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

- Stocks: DuckDB + Zipline bundle. `ZiplineEngine` reads from the bundle; `VectorBTProEngine` reads from DuckDB.
- Crypto: DuckDB only. `ZiplineEngine` is not used for crypto.
