# WORKFLOWS

## Overview

Three workflows — one Prefect flow, one CLI entry point. Only the YAML config changes.

```
Research  →  Validation  →  Live
  (fast)      (normal)     (execute)

qts run --config research.yaml
qts run --config validation.yaml
qts run --config live.yaml
```

---

## 1. Research — Signal Discovery

**Goal:** Fast iteration over features, parameters, universe.
**Engine:** `VectorBTProEngine` — full history vectorised.

```
1. DataManager.get_ohlcv(config.universe, config.start_date, config.end_date)
      Routes by config.data_sources → writes to config.storage (duckdb)

2. FeaturePipeline.fit_transform(df)
      Built from config.features:
        technical    → all asset types
        fundamental  → stocks only (no-op for crypto)
        onchain      → crypto only (no-op for stocks)
        ForwardReturns(periods=config.features.forward_returns.periods)

3. Strategy.generate_signals(df)
      Resolved from config.strategy.type via registry
      Returns SignalFrame[date, symbol, signal, weight]

4. VectorBTProEngine.run(strategy, data, config)
      Vectorises full signal history → batch simulation

5. BacktestResult → metrics.py
      Sharpe, Sortino, CAGR, max drawdown, win rate

6. (optional) Grid search / Optuna sweep over steps 3–5
```

**Output:** Best strategy config — parameters, features, universe.

---

## 2. Validation — Pre-Live Check

**Goal:** Confirm research results hold under realistic simulation.
**Engine:** `ZiplineEngine` — strict bar-by-bar, look-ahead prevented.

```
1. Load Zipline bundle (built from same DuckDB data — no re-download)

2. FeaturePipeline.fit_transform(df)
      Identical pipeline to research

3. Strategy.generate_signals(visible_data_only)
      Engine exposes only data up to current bar — no future leakage

4. ZiplineEngine.run(strategy, data, config)
      config.fill_model      → resolved from registry
      config.slippage_model  → resolved from registry
      config.commission      → model + rate applied per fill
      config.calendar        → nyse | hkex | crypto

5. BacktestResult vs research result
      Sharpe degradation > config.promotion_gate.max_sharpe_degradation → reject
      Metrics hold → promote to live
```

**Output:** Validated strategy config — safe to go live.

---

## 3. Live — Scheduled Execution

**Goal:** Periodic rebalance against live brokers.
**Orchestration:** `qts_flow` scheduled per `config.schedule`.

```
1. [Prefect · parallel] Data download for each asset type in config.asset_types
      download_ohlcv(config.universe, lookback)          → config.storage
      download_fundamentals(config.universe, lookback)   → config.storage
        (only if config.features.fundamental = true)

2. [Prefect] build_features(df, config.features)
      Same FeaturePipeline — built from config

3. [Prefect] run_backtest(strategy, data, config)
      config.backtest_engine = "fast" — produces final-bar target weights

4. [Prefect] sync_positions(config.brokers)
      broker.get_positions() per configured broker → current holdings
      PositionSync.compute_deltas(target_weights, current) → order list

5. [Prefect] execute_rebalance(orders, config.brokers)
      OrderRouter.execute(orders)
        routes each order to broker resolved from config.brokers by AssetType
          stock  → MoomooBroker
          crypto → BinanceBroker

6. Log fills → config.storage positions table
```

**Schedule:** `config.schedule` per asset type — no cron literals in `flow.py`.

---

## Config Examples

```yaml
# research.yaml
workflow: research
asset_types: [stock, crypto]
universe:
  stock:  [AAPL, GOOGL, MSFT]
  crypto: [BTC/USDT, ETH/USDT]
start_date: "2018-01-01"
end_date:   "2024-12-31"
initial_capital: 100000
data_sources:
  stock:  fmp
  crypto: binance
storage: duckdb
features:
  technical: true
  fundamental: true
  onchain: true
  forward_returns:
    periods: [1, 5, 20]
strategy:
  type: factor
  params: { n_factors: 5, lookback: 60 }
backtest_engine: fast
```

```yaml
# validation.yaml
workflow: validation
asset_types: [stock]
universe:
  stock: [AAPL, GOOGL, MSFT]
start_date: "2018-01-01"
end_date:   "2024-12-31"
initial_capital: 100000
data_sources:
  stock: fmp
storage: duckdb
features:
  technical: true
  fundamental: true
  onchain: false
  forward_returns:
    periods: [1, 5, 20]
strategy:
  type: factor
  params: { n_factors: 5, lookback: 60 }
backtest_engine: normal
fill_model: next_open
slippage_model: volatility_scaled
commission:
  model: percentage
  rate: 0.001
calendar: nyse
promotion_gate:
  max_sharpe_degradation: 0.30
```

```yaml
# live.yaml
workflow: live
asset_types: [stock, crypto]
universe:
  stock:  [AAPL, GOOGL, MSFT]
  crypto: [BTC/USDT, ETH/USDT]
data_sources:
  stock:  fmp
  crypto: binance
storage: duckdb
features:
  technical: true
  fundamental: true
  onchain: true
  forward_returns:
    periods: [1, 5, 20]
strategy:
  type: factor
  params: { n_factors: 5, lookback: 60 }
backtest_engine: fast
fill_model: next_open
slippage_model: volatility_scaled
commission:
  model: percentage
  rate: 0.001
brokers:
  stock:  moomoo
  crypto: binance
  binance_mode: demo       # demo (testnet) | live (production)
schedule:
  stock:  "0 16 * * 1-5"   # daily after NYSE close
  crypto: "0 */4 * * *"    # every 4 hours
```

---

## Promotion Gates

```
research.yaml ──► metrics pass threshold
      │
      ▼
validation.yaml ──► Sharpe degradation < 30%
      │
      ▼
live.yaml ──► scheduled per config.schedule
```

Config files are version-controlled. Promotion is a code review, not a runtime decision.
