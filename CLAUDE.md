# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Commands

All commands run from `QTradeSystematic/` using the repo-local venv — never the global interpreter.

```bash
# Install (choose the extras you need)
.venv/bin/pip install -e ".[dev]"          # core + dev tools only
.venv/bin/pip install -e ".[all]"          # everything

# Tests
.venv/bin/python -m pytest                            # all unit + integration tests
.venv/bin/python -m pytest tests/data/ -v             # one module
.venv/bin/python -m pytest tests/core/test_core.py::test_instrument_model  # single test
.venv/bin/python -m pytest -m paper                   # paper/testnet tests (needs live creds)

# Lint
.venv/bin/ruff check qts/
.venv/bin/ruff check qts/ --fix

# Run a flow
.venv/bin/python qts/orchestration/serve.py           # register + start Prefect deployments
```

---

## Architecture

### Layer boundaries (strict — no exceptions)

```
core/  ←  data/  ←  research/  ←  orchestration/
core/  ←  execution/            ←  orchestration/
core/  ←  config/   (registry only)
```

`execution/` never imports from `research/`. Strategy code never imports broker adapters. Violations break the dependency graph.

### Routing: how a symbol becomes a source + table

`AssetType.from_symbol(symbol)` is the single routing function — nothing else should infer asset type from a symbol string:

| Symbol | `AssetType` | Source | DuckDB table |
|---|---|---|---|
| `PERP:ETH/USDT` | `CRYPTO_FUTURES` | `BinanceFuturesDataSource` | `futures_prices` |
| `BTC/USDT` | `CRYPTO` | `BinanceDataSource` | `crypto_prices` |
| `VN:VNM` | `VN_STOCK` | `VnstockDataSource` / `DNSEDataSource` | `vn_stock_prices` |
| `VNF:VN30F2503` | `VN_FUTURES` | `VnstockFuturesDataSource` / `DNSEDataSource` | `vn_futures_prices` |
| `VNW:VNM` | `VN_WARRANT` | `VnstockDataSource` / `DNSEDataSource` | `vn_warrant_prices` |
| `AAPL` | `STOCK` | `FMPDataSource` | `stock_prices` + Zipline bundle |

`DataManager.__init__` builds a `(AssetType, DataType) → source` map from each source's `CAPABILITIES` frozenset. Adding a new source only requires declaring `CAPABILITIES` and passing the instance to `DataManager`.

### Config → runtime resolution

`Config.build(path)` parses YAML → `BacktestConfig` (typed dataclass) → resolves every string key through `Registry` → returns `ResolvedConfig` with concrete instances. `ResolvedConfig` also exposes `data_sources()`, `brokers()`, and `with_fundamentals()` so orchestration can assemble runtime collaborators without rebuilding asset-type maps inline. YAML keys like `"binance"` or `"fmp"` map to classes via `@Registry.register_data_source("key")`. An unregistered key raises `RegistryError` at build time, not at runtime.

### Plugin registration pattern

```python
@Registry.register_data_source("my_source")
class MySource(BaseDataSource):
    CAPABILITIES = frozenset({DataType.OHLCV})
    async def fetch(self, data_type, symbol, **kwargs) -> pl.DataFrame: ...
```

Same three-step pattern (inherit ABC → register with decorator → reference key in YAML) applies to all extension points: data sources, storage, features, strategies, engines, brokers.

### Data flow

```
BinanceDataSource  ──► DataManager.get_ohlcv()         ──► crypto_prices   (DuckDB)
BinanceFuturesDataSource ──► get_futures_ohlcv()        ──► futures_prices  (DuckDB)
FMPDataSource      ──► DataManager.get_ohlcv()         ──► stock_prices    (DuckDB) + Zipline bundle
                                                                │
                                              FeaturePipeline.fit_transform(df)
                                                                │
                                              Strategy.generate_signals(df) → SignalFrame
                                                                │
                                   VectorBTProEngine / ZiplineReloadedEngine → BacktestResult
```

`VectorBTProEngine` reads from DuckDB/Polars for all asset types. `ZiplineReloadedEngine` auto-ingests a Zipline bundle per run and supports stocks, VN stocks, crypto, and crypto futures; mixed-type universes require an explicit `calendar:`. Both must produce identical `BacktestResult` schema.

### Persistent storage paths

Centralised in `utils/paths.py`. Default root `~/.qts/`; override with `QTS_ROOT`.

```
~/.qts/
├── database/qts.duckdb          ← stock_prices, crypto_prices, futures_prices,
│                                   vn_stock_prices, vn_futures_prices, vn_warrant_prices
├── cache/
│   └── vn_fundamentals/         ← {ticker}_{annual|quarterly}.parquet, TTL 24 h
└── bundles/                     ← Zipline bundles
```

### Key schemas

**OHLCV** — all sources must output exactly this:
```
[date: Date, symbol: Utf8, open: f64, high: f64, low: f64, close: f64, volume: f64]
```

**SignalFrame** — all strategies must produce exactly this:
```
[date, symbol, signal: int ∈ {-1, 0, 1}, weight: float ∈ [0, 1]]
```

Monetary values: always `Decimal`, never `float`. Bar timestamps: `date` for OHLCV; `datetime` only for ticks.

---

## Critical rules

- **No imports from `omega`, `qsconnect`, `qsresearch`, `qsautomate`** — they are design references, not dependencies.
- **Vendor SDKs only inside their adapter module** — `binance-connector`, `futu-api`, `vectorbtpro`, `zipline`, `prefect` must not appear outside their designated file.
- **No `if asset_type == "crypto"` in business logic** — always route via registry or router.
- **`flow.py` contains zero hardcoded values** — every axis of behaviour comes from `BacktestConfig`.
- **`execution/` is real-money territory** — verify against paper/testnet before changing order placement logic.
- **Look-ahead guard in backtesting** — `ZiplineReloadedEngine` enforces this at the bar level; `VectorBTProEngine` relies on the feature pipeline not leaking future data.

---

## Testing tiers

| Tier | Allowed | Forbidden |
|---|---|---|
| Unit | In-memory mocks at ABC boundary | Network, real APIs, DB files |
| Integration | Real in-memory DuckDB, Zipline bundle on disk | Any broker API |
| Paper (`@pytest.mark.paper`) | Binance testnet, Futu OpenD paper mode | Live (real-money) accounts |

Mock at the ABC boundary only — never mock internal methods. Engine tests must run both `VectorBTProEngine` and `ZiplineReloadedEngine` on the same fixture and assert identical `BacktestResult` schema.

---

## Detailed references

- `ARCHITECTURE.md` — full layer descriptions, symbol convention, config schema, deployment table
- `WORKFLOWS.md` — research → validation → live promotion pipeline with config examples
- `CODING_RULES.md` — ABC contracts, naming conventions, error handling rules
- `AGENTS.md` — agent-specific guidance, anti-patterns, high-risk areas
- `FEATURES.md` — 45-feature build checklist with verify steps per feature
