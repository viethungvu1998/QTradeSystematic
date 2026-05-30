# What is QTradeSystematic?

QTradeSystematic (QTS) is a config-driven systematic trading framework written in Python.
You describe a strategy, a universe, and a date range in a YAML file, then call one async
function — `qts_flow` — to fetch data, build features, run a backtest, and (optionally) execute
live orders. The same config file drives research, validation, and live trading; changing the
`workflow:` key is the only switch.

The framework targets quantitative researchers who want to iterate quickly across asset classes
without rewiring boilerplate for each new asset type or broker.

---

## Supported Asset Classes

`AssetType.from_symbol()` is the single routing function. Every symbol string maps to exactly one
asset type, which determines the data source, DuckDB table, and exchange calendar used during a
backtest.

| Symbol format | `AssetType` | DuckDB table | Example symbols |
|---|---|---|---|
| Bare alphabetic | `STOCK` | `stock_prices` | `AAPL`, `MSFT`, `GOOGL` |
| Starts with `VN:` | `VN_STOCK` | `vn_stock_prices` | `VN:VNM`, `VN:VIC` |
| Starts with `VNF:` | `VN_FUTURES` | `vn_futures_prices` | `VNF:VN30F2606` |
| Starts with `VNW:` | `VN_WARRANT` | `vn_warrant_prices` | `VNW:CVNM2511` |
| Contains `/` | `CRYPTO` | `crypto_prices` | `BTC/USDT`, `ETH/USDT` |
| Starts with `PERP:` | `CRYPTO_FUTURES` | `futures_prices` | `PERP:ETH/USDT` |
| Starts with `CMX:` | `COMMODITY` | *(reserved, not yet wired)* | `CMX:CL`, `CMX:GC` |

> **VN stock symbols must carry the `VN:` prefix** in all config files.
> `COMMODITY` symbols are silently skipped by `DataManager` until a data source is registered.

---

## Supported Data Sources

`DataManager` builds a `(AssetType, DataType) → source` map at startup from each source's
`CAPABILITIES` frozenset. Adding a new source requires only declaring `CAPABILITIES` and
registering it with `@Registry.register_data_source("key")`.

| Registry key | Capabilities | Live client | Notes |
|---|---|---|---|
| `fmp` | `ohlcv`, `fundamentals` | fixture-only | US stocks via Financial Modeling Prep |
| `yahoo` | `ohlcv` | fixture-only | US stocks via Yahoo Finance |
| `binance` | `ohlcv` | `from_env()` | Spot crypto; needs `BINANCE_API_KEY` / `BINANCE_API_SECRET` |
| `binance_futures` | `futures_ohlcv` | `from_env()` | USDT-M perps |
| `vnstock` | `ohlcv`, `fundamentals` | `from_env()` — global access | KBS public API; no auth required |
| `vnstock_futures` | `futures_ohlcv` | `from_env()` — global access | VN30 index futures via KBS |
| `dnse` | `ohlcv`, `futures_ohlcv` | `from_env()` — **geo-restricted to VN IP** | DNSE OpenAPI; needs `DNSE_API_KEY` / `DNSE_API_SECRET` |

Use `vnstock` as your primary VN source when running from outside Vietnam; `dnse` is only
reachable from Vietnamese IP addresses.

---

## Supported Strategies

Four strategy families are registered in the registry. All of them implement
`BaseStrategy.generate_signals(data) -> SignalFrame`.

| Registry key | Description |
|---|---|
| `factor` | Cross-sectional factor strategy. Ranks symbols by one or more numeric factors computed from the feature pipeline, then converts ranks to long/short signals using configurable quantile thresholds. Supports XGBoost regressors, rankers, and IC-weighted composites as signal algorithms. |
| `ml_factor` | Machine-learning classification strategy. Trains an XGBoost classifier (or linear model) on tabular features to predict the sign of forward returns, then produces signals from predicted probabilities. |
| `stat_arb` | Pair-trading strategy. Selects cointegrated pairs from the universe, estimates an OLS or rolling-OLS hedge ratio, and generates long/short spread signals when the z-score crosses configurable entry/exit bands. |
| `vn100_quantamental` | Walk-forward ML factor strategy purpose-built for the VN100 universe. Combines QSMOM momentum transforms, technical indicators, and VN fundamentals as features, then trains an XGBoost regressor on rolling windows to produce long-only signals. |

---

## Backtest Engines

Two simulation engines are available. Both accept identical `BacktestConfig` and return identical
`BacktestResult` schema.

| Registry key | Class | When to use |
|---|---|---|
| `vectorbt` (alias `fast`) | `VectorBTProEngine` | Vectorized target-percent simulation. Fastest for crypto and large universes. Reads DuckDB/Polars directly; no exchange calendar required. |
| `zipline` (alias `normal`) | `ZiplineReloadedEngine` | Calendar-aware bar simulation with realistic fill models. Use for US stocks (`nyse` calendar) or VN stocks (`hose` calendar). Requires the `zipline` extra and auto-ingests a bundle on each run. |

Choose `vectorbt` for rapid iteration and crypto universes. Choose `zipline` when you need
exchange-calendar alignment, realistic open-price fills, or mixed-asset universes with an
explicit `calendar:`.

---

## Current Limitations

- **No CLI.** There is no `qts run` command. The entry point is the Python async function
  `qts_flow(config_path)`.
- **Validation does not auto-compare Sharpe.** `workflow: validation` parses
  `promotion_gate.max_sharpe_degradation`, but the flow does not currently rerun a research
  backtest and compare the two Sharpe ratios. Treat `validation` as a stricter config profile,
  not an automated promotion step.
- **`dnse` is geo-restricted.** The DNSE OpenAPI endpoint (`openapi.dnse.com.vn`) only responds
  from Vietnamese IP addresses. Use `vnstock` as a global fallback.
- **`DNSEBroker` is not auto-imported.** The broker is implemented in
  `qts/execution/brokers/dnse.py` but is not exported by `qts.execution.brokers.__init__`, so
  it is not loaded by a plain `import qts`.
- **`COMMODITY` is a reserved enum value.** Symbols with `CMX:` prefix are silently skipped by
  `DataManager` until a commodity data source is registered.
- **No packaged CLI layer.** Strategy-specific runners are not part of the supported workflow
  surface; engines remain the only simulation entrypoints.

---

## Storage Layout

All persistent data lives under `~/.qts/` by default. Override the root with the `QTS_ROOT`
environment variable.

```text
~/.qts/
├── database/
│   └── qts.duckdb          ← primary store; six tables:
│                               stock_prices, crypto_prices, futures_prices,
│                               vn_stock_prices, vn_futures_prices, vn_warrant_prices
├── cache/
│   └── vn_fundamentals/    ← {ticker}_{annual|quarterly}.parquet
│                               TTL 24 h; force_refresh=True bypasses the cache
├── bundles/                ← Zipline bundles (one per asset type)
└── exports/
    ├── backtests/          ← trade_log CSV, portfolio_snapshots CSV, tearsheet PDF
    └── live/               ← live portfolio CSV snapshots
```

---

## See Also

- [installation.md](installation.md) — set up your environment
- [run_backtest.md](run_backtest.md) — run your first backtest
- [pull_data.md](pull_data.md) — fetch and inspect market data
- [creating_strategy.md](creating_strategy.md) — write a custom strategy
