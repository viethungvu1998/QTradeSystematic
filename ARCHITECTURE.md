# ARCHITECTURE

## Repo Layout

```text
QTradeSystematic/
├── qts/
│   ├── config/
│   ├── core/
│   ├── data/
│   ├── execution/
│   ├── orchestration/
│   ├── research/
│   └── utils/
├── examples/
├── tests/
└── pyproject.toml
```

## Layer Summary

### `qts/core`

Shared models and extension points.

- `instrument.py`: `AssetType`, `Instrument`
- `order.py`: `Order`, `Fill`, enums
- `portfolio.py`: `Position`, `Portfolio`
- `events.py`: async tick/order event models
- `registry.py`: central plugin registry
- `observability.py`: `ClosedTrade`, `PositionSnapshot`, `TradeLog` result types

Asset type routing is derived from symbol strings via `AssetType.from_symbol()`:

- `PERP:*` → `CRYPTO_FUTURES`
- `VNF:*` → `VN_FUTURES`
- `VNW:*` → `VN_WARRANT`
- `*/*` → `CRYPTO`
- `VN:*` → `VN_STOCK`
- `CMX:*` → `COMMODITY`
- otherwise → `STOCK`

### `qts/data`

Acquisition, routing, persistence, and bundle support.

```text
qts/data/
├── _schemas.py
├── base.py
├── manager.py
├── bundles/
├── sources/
└── storage/
```

Key components:

- `DataType`: `ohlcv`, `fundamentals`, `options_chain`, `funding_rates`, `open_interest`, `futures_ohlcv`
- `DataManager`: DB-first, then parquet cache, then source fetch
- `DuckDBStorage`: main persistent store
- `ParquetStorage`: cache store
- `LocalBundleAdapter`: local bundle persistence
- `ZiplineBundleAdapter`: zipline-reloaded ingestion path

Default storage paths come from `qts/utils/paths.py`:

```text
~/.qts/
├── database/qts.duckdb          ← stock_prices, crypto_prices, futures_prices,
│                                   vn_stock_prices, vn_futures_prices, vn_warrant_prices
├── cache/
│   └── vn_fundamentals/         ← {ticker}_{annual|quarterly}.parquet, TTL 24 h
└── bundles/                     ← Zipline bundles
```

Current source registrations:

| Key | Module | Capabilities | Live client |
|---|---|---|---|
| `fmp` | `qts/data/sources/fmp.py` | `ohlcv`, `fundamentals` | fixture-only |
| `yahoo` | `qts/data/sources/yahoo.py` | `ohlcv` | fixture-only |
| `dnse` | `qts/data/sources/dnse.py` | `ohlcv`, `futures_ohlcv` | `from_env()` — geo-restricted to VN IP |
| `vnstock` | `qts/data/sources/vnstock.py` | `ohlcv`, `fundamentals` | `from_env()` — KBS public API, no auth |
| `vnstock_futures` | `qts/data/sources/vnstock.py` | `futures_ohlcv` | `from_env()` — KBS public API, no auth |
| `binance` | `qts/data/sources/binance.py` | `ohlcv` | `from_env()` |
| `binance_futures` | `qts/data/sources/binance.py` | `futures_ohlcv` | `from_env()` |

KBS (`vnstock` / `vnstock_futures`) details:

- Base URL: `https://kbbuddywts.kbsec.com.vn/iis-server/investment`
- No authentication required; browser User-Agent is sufficient.
- Equity/warrant prices are returned in thousands of VND (divide by 1000 → VND/1000 unit).
- Index and futures prices are returned in full points (no scaling).
- VN30 futures symbols: old format `VNF:VN30F2606` is auto-converted to KRX format `41I1G6000` for contracts expiring May 2025 or later.
- DNSE (`openapi.dnse.com.vn`) requires API key/secret auth with the current OpenAPI signature headers and is geo-restricted to Vietnamese IPs; use `vnstock` as the primary source from outside Vietnam.
- DNSE futures requests are normalized to rolling aliases such as `VNF:VN30F1M`, even when the input uses a dated symbol such as `VNF:VN30F2503`.
- DNSE warrant-underlying requests such as `VNW:VNM` are expanded into concrete warrant symbols for fetch, cache, and storage, for example `VNW:CVNM2511`.

Fundamentals pipeline (`vnstock` source):

- Fetches 4 report types: `KQKD` (income), `CDKT` (balance sheet), `LCTT` (cash flow), `CSTC` (ratios).
- Stored in tidy long format: `symbol | report_type | period | fiscal_year | quarter | report_date | item_en | value`.
- Values from `KQKD`/`CDKT`/`LCTT` are in thousands of VND. `CSTC` values are native ratio units (%, ×, VND/share).
- On-disk cache at `~/.qts/cache/vn_fundamentals/{ticker}_{annual|quarterly}.parquet`, TTL 24 h.
- Pass `force_refresh=True` to `get_fundamentals()` to bypass the cache.

Important implementation detail:

- `Config.build()` prefers `from_env()` for `dnse`, `vnstock`, and `vnstock_futures`.
- If required env vars are missing for a source that supports `from_env()`, the builder falls back to the default constructor so fixture-oriented tests still work.

### `qts/research`

Feature engineering, strategy logic, backtesting, and analysis helpers.

```text
qts/research/
├── backtest/
│   ├── engines/
│   ├── simulation/
│   ├── observability.py          ← VectorBT result extraction
│   └── zipline_observability.py  ← Zipline result extraction
├── features/
│   ├── indicators/
│   └── transforms/               ← qsmom, price_preprocessor, universe_screener
├── portfolio_analysis/
├── portfolio_construction/
├── statistical_analysis/
└── strategies/
```

#### Features

- `FeaturePipeline.fit_transform()` always starts with `preprocess_ohlcv()`
- `FeaturePipeline.requires_fundamentals()` lets orchestration ask whether external fundamentals must be downloaded
- `FeaturePipeline.with_fundamentals()` returns a bound copy instead of mutating the original pipeline in place
- Legacy `technical` feature still exists
- Fine-grained indicator plugins live under `features/indicators/`

<!-- AUTO-GENERATED from Registry decorators in qts/research/features/ -->

Registered feature keys:

- `technical`
- `fundamental`
- `vn_fundamental`
- `onchain`
- `forward_returns`
- `rsi`, `roc`, `macd`, `adx`, `atr`, `bollinger`, `hist_vol`, `obv`, `volume_ratio`, `zscore`

Registered transform keys:

- `qsmom` — QS momentum transform (`features/transforms/momentum.py`)
- `price_preprocessor` — price quality preprocessing (`features/transforms/quality.py`)
- `universe_screener` — universe filtering (`features/transforms/screener.py`)

<!-- END AUTO-GENERATED -->

#### Strategies

<!-- AUTO-GENERATED from Registry decorators in qts/research/strategies/ -->

Registered strategy keys:

- `factor`
- `ml_factor`
- `stat_arb`
- `vn100_quantamental`

Registered signal algorithm keys (`factor` family):

- `cross_sectional_rank`
- `factor_as_signal`
- `ic_weighted`

Registered spread model keys (`stat_arb` family):

- `ols`
- `rolling_ols`

Registered ML model keys (`ml_factor` family):

- `xgb_classifier`
- `xgb_regressor`
- `linear`
- `ic_composite`

<!-- END AUTO-GENERATED -->

Refactor direction for strategy families:

- Keep `BaseStrategy.generate_signals(data) -> SignalFrame` as the stable public strategy seam.
- Family-level `base.py` and `core.py` modules are justified only when a family has real shared behavior.
- `factor/` and `stat_arb/` are the two earned families for this structure today.
- Do not add `cross_sectional/base.py` or `cross_sectional/core.py` until the first concrete cross-sectional strategy exists.
- Stat-arb should conform to the same seam as every other strategy: consume a universe frame and emit a standard signal frame across traded symbols.

#### Backtesting

Backtest runtime lives in `qts/research/backtest/`.

- `BacktestConfig` and `BacktestResult` are the shared contracts
- `walk_forward_signals()` implements train-window signal generation
- `run_backtest_frame()` converts signal frames into portfolio returns
- Engines are the only supported simulation entrypoints

Registered engine keys:

| Key | Class | Notes |
|---|---|---|
| `vectorbt` | `VectorBTProEngine` | Vectorized target-percent simulation |
| `fast` | `VectorBTProEngine` | Alias for `vectorbt` |
| `zipline` | `ZiplineReloadedEngine` | Calendar-aware zipline-reloaded simulation |
| `normal` | `ZiplineEngine` | Compatibility engine class; YAML config loader currently normalizes `normal` -> `zipline` |

Simulation model registrations:

- Fill models: `immediate`, `next_open`, `vwap`
- Slippage models: `fixed`, `volatility_scaled`
- Commission models: `percentage`, `per_trade`
- Calendars: `nyse`, `hkex`, `hose`, `crypto`

### `qts/execution`

Live-order abstractions and adapters.

```text
qts/execution/
├── base.py
├── router.py
├── sync.py
└── brokers/
```

Current broker registrations:

| Key | Class | Current status |
|---|---|---|
| `moomoo` | `MoomooBroker` | Fixture-friendly unless a client is injected |
| `binance` | `BinanceBroker` | Can build a real spot client via `from_env()` |
| `dnse` | `DNSEBroker` | Implemented and decorated in-module, but not auto-imported by `qts.execution.brokers.__init__` |

Notes:

- `PositionSync` and `OrderRouter` are operational and used by `qts_flow` in `live` mode.

### `qts/orchestration`

Async orchestration and Prefect compatibility.

```text
qts/orchestration/
├── flow.py
├── runtime.py
├── prefect_compat.py
├── serve.py
├── flows/
└── tasks/
```

Current flow surface:

- `qts_flow(config_path: str)`: main async entry point
- `data_fetch_flow(config_path: str, asset_types: list[str], data_types: list[str])`: data-only async flow

Current flow behavior:

- `qts_flow()` builds runtime collaborators from `ResolvedConfig`, downloads `stock`, `vn_stock`, `vn_warrant`, and spot `crypto` OHLCV plus `vn_futures` and `crypto_futures` data when configured, binds fundamentals into a copied feature pipeline only when required, then runs research or live execution logic.
- `data_fetch_flow()` resolves requested asset classes through shared symbol-routing helpers, including `vn_warrant`, `vn_futures`, and `crypto_futures`, and uses the same `DataManager` assembly path as the main flow.

`prefect_compat.py` provides a fallback `flow`, `task`, and `serve` shim when Prefect is not installed.

Current deployment registration in `serve.py` creates six data-refresh deployments:

- `stock-ohlcv-daily`
- `vn-stock-ohlcv-daily`
- `crypto-ohlcv-daily`
- `crypto-funding-8h`
- `stock-fundamentals-weekly`
- `vn-stock-fundamentals-weekly`

### `qts/config`

Configuration parsing and runtime resolution.

- `load_config()` validates YAML and returns `BacktestConfig`
- `Config.build()` resolves registry keys into concrete runtime objects
- `ResolvedConfig.data_sources()` and `ResolvedConfig.brokers()` expose assembled asset-type maps for orchestration
- `ResolvedConfig.with_fundamentals()` binds external fundamentals into a copied feature pipeline

Current behavior to keep in mind:

- `promotion_gate` is parsed into config for `validation`, but not enforced by the flow
- `brokers.binance_mode` is stored in config, but broker construction does not yet switch on it automatically

Refactor direction:

- Package bootstrap should import strategy families, not a hand-picked set of concrete strategy modules.
- Registry-backed strategies should all resolve after package import, including `ml_factor`.
- Built-in features and storage should resolve through the same registries as the rest of the plugin surface.
- YAML compatibility stays intact during this cleanup: `strategy.type`, `strategy.params`, `backtest_engine`, and the existing feature booleans remain valid.

## End-to-End Data Path

### Research and validation

```text
config_path
  -> Config.build()
  -> build_data_manager()
  -> download_ohlcv()
  -> optional download_futures_ohlcv()
  -> optional download_fundamentals()
  -> feature_pipeline.with_fundamentals()
  -> FeaturePipeline.fit_transform()
  -> engine.run()
  -> BacktestResult
```

### Live

```text
config_path
  -> Config.build()
  -> build_data_manager()
  -> download_ohlcv() + optional download_futures_ohlcv()
  -> build_features()
  -> run_backtest()
  -> latest target weights
  -> PositionSync.compute_deltas()
  -> build_order_router()
  -> OrderRouter.execute()
```

## Current Gaps

These are the main architecture gaps between the implemented modules and the top-level flows:

- validation metadata is parsed, but there is no automatic research-vs-validation gate
- several adapters are framework stubs awaiting real client wiring
- broker package bootstrap still hand-picks imports, so `DNSEBroker` is not loaded by `import qts` alone
- there is no packaged CLI layer yet
