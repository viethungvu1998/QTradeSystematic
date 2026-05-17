# WORKFLOWS

## Overview

This repository currently exposes two callable async flows and one deployment-registration script.

- `qts_flow(config_path)`
- `data_fetch_flow(config_path, asset_types, data_types)`
- `python qts/orchestration/serve.py`

There is no `qts run` CLI today.
There is also no supported strategy-specific workflow surface; engines remain the only simulation entrypoints.

## 1. Research

Use `workflow: research` in YAML.

Runtime path:

1. `qts_flow()` calls `Config.build(config_path)`
2. `build_data_manager()` assembles `DataManager` from `ResolvedConfig`
3. `download_ohlcv()` fetches `stock`, `vn_stock`, `vn_warrant`, and spot `crypto`
4. `download_futures_ohlcv()` fetches `vn_futures` and `crypto_futures` when present in config
5. `download_fundamentals()` runs only when the configured feature pipeline declares `requires_fundamentals()`
6. `ResolvedConfig.with_fundamentals()` binds those fundamentals into a copied feature pipeline
7. `FeaturePipeline.fit_transform()` preprocesses OHLCV, then applies configured features
8. `run_backtest()` executes the configured engine
9. The flow returns a `BacktestResult`

Important notes:

- `qts_flow` is `async`
- the argument is a config path string
- walk-forward signal generation is handled inside the engine path when pipeline and raw OHLCV are supplied
- strategy-specific runners are not part of the supported workflow surface

Example:

```python
import asyncio

from qts.orchestration.flow import qts_flow

result = asyncio.run(qts_flow("examples/research_zipline.yaml"))
print(result.metrics)
```

## 2. Validation

Use `workflow: validation` in YAML.

Current behavior:

- the loader requires additional keys such as `fill_model`, `slippage_model`, `commission`, `calendar`, and `promotion_gate`
- `qts_flow` still follows the same control path as research
- the selected backtest engine is whatever `backtest_engine` resolves to, typically `zipline`

What validation does not currently do:

- it does not rerun a separate research backtest automatically
- it does not compare research and validation Sharpe
- it does not enforce `promotion_gate.max_sharpe_degradation`

So `validation` is best understood as a stricter config profile plus a different engine choice, not an automated promotion step.

## 3. Live

Use `workflow: live` in YAML.

Runtime path:

1. `qts_flow()` resolves config and dependencies
2. OHLCV is downloaded for `stock`, `vn_stock`, `vn_warrant`, and spot `crypto`, plus futures data for `vn_futures` and `crypto_futures`
3. features are built from the combined panel
4. the configured backtest engine generates a `BacktestResult`
5. the latest signal row per symbol is converted into target weights
6. `PositionSync` computes delta orders from broker positions and account value
7. `build_order_router()` assembles the live broker map from `ResolvedConfig`
8. `OrderRouter` dispatches orders concurrently
9. the flow returns a dict with:
   - `result`
   - `orders`
   - `fills`
   - `schedule`

Current live-mode caveats:

- `MoomooBroker` is fixture-friendly unless a client is injected
- `BinanceBroker` supports real credentials through `from_env()`, but `Config.build()` currently instantiates brokers through their default constructors
- `DNSEBroker` is implemented, but `qts.execution.brokers.__init__` does not auto-import it yet
- `brokers.binance_mode` is parsed into config but not applied during object construction

## 4. Data Refresh

`data_fetch_flow()` is the data-only flow.

Signature:

```python
async def data_fetch_flow(config_path: str, asset_types: list[str], data_types: list[str]) -> None
```

Current behavior:

- resolves config with `Config.build()`
- builds a `DataManager` through `build_data_manager(..., include_bundle=False)`
- resolves symbols through the shared `requested_symbols()` helper
- supports `stock`, `vn_stock`, `vn_warrant`, `vn_futures`, `crypto`, and `crypto_futures`
- calls `manager.get(DataType(...), symbols, start=..., end=...)`

Example:

```python
import asyncio

from qts.orchestration.flows.data_fetch_flow import data_fetch_flow

asyncio.run(data_fetch_flow("examples/research_zipline.yaml", ["stock"], ["ohlcv"]))
```

## 5. Deployment Registration

`qts/orchestration/serve.py` registers six data-refresh deployments through the Prefect compatibility layer.

Current deployments:

| Deployment | Asset types | Data types | Cron |
|---|---|---|---|
| `stock-ohlcv-daily` | `["stock"]` | `["ohlcv"]` | `0 21 * * 1-5` |
| `vn-stock-ohlcv-daily` | `["vn_stock"]` | `["ohlcv"]` | `0 9 * * 1-5` |
| `crypto-ohlcv-daily` | `["crypto"]` | `["ohlcv"]` | `0 0 * * *` |
| `crypto-funding-8h` | `["crypto"]` | `["funding_rates"]` | `0 */8 * * *` |
| `stock-fundamentals-weekly` | `["stock"]` | `["fundamentals"]` | `0 8 * * 1` |
| `vn-stock-fundamentals-weekly` | `["vn_stock"]` | `["fundamentals"]` | `0 8 * * 1` |

Run it with:

```bash
python qts/orchestration/serve.py
```

If Prefect is not installed, the compatibility shim keeps the decorators and deployment objects importable for local development and tests.

## Config Shape Used by Flows

Common fields:

```yaml
workflow: research | validation | live
asset_types: [stock, vn_stock, vn_futures, vn_warrant, crypto, crypto_futures]
start_date: "YYYY-MM-DD"   # required for research and validation
end_date: "YYYY-MM-DD"     # required for research and validation
initial_capital: 100000    # required for research and validation
universe:
  stock: []
  vn_stock: []          # VN: prefix, e.g. VN:VNM
  vn_futures: []        # VNF: prefix; DNSE accepts dated inputs and normalizes them to rolling aliases like VNF:VN30F1M
  vn_warrant: []        # DNSE can take an underlying like VNM or VNW:VNM and expand it into all currently listed warrants
  crypto: []
  crypto_futures: []
data_sources:
  stock: fmp | yahoo
  vn_stock: vnstock     # KBS public API — use dnse only from Vietnamese IP
  vn_futures: vnstock_futures
  vn_warrant: vnstock   # with dnse, warrant-underlying requests expand to concrete codes such as VNW:CVNM2511
  crypto: binance
  crypto_futures: binance_futures
storage: duckdb
features:
  indicators: []
  technical: false
  fundamental: false
  onchain: false
  forward_returns:
    periods: []
strategy:
  type: factor | ml_factor | stat_arb
  params: {}
backtest_engine: vectorbt | zipline | fast | normal
train_window: 252
rebalance_frequency: monthly
```

Interpretation rules:

- all strategy-specific options, including stat-arb settings, live under `strategy.params`
- the existing feature booleans remain valid for YAML compatibility
- registry-driven feature entries can also reference plugins such as `vn_fundamental`
- cleanup work should normalize those booleans and storage resolution through registries without changing the public YAML shape in the same pass

Validation-only required keys:

```yaml
fill_model: immediate | next_open | vwap
slippage_model: fixed | volatility_scaled
commission:
  model: percentage | per_trade
  rate: 0.001
calendar: nyse | hkex | crypto
promotion_gate:
  max_sharpe_degradation: 0.30
```

Live-only required keys:

```yaml
fill_model: immediate | next_open | vwap
slippage_model: fixed | volatility_scaled
commission:
  model: percentage | per_trade
  rate: 0.001
brokers:
  stock: moomoo
  vn_stock: null
  vn_futures: null
  vn_warrant: null
  crypto: binance
schedule:
  stock: "0 16 * * 1-5"
  crypto: "0 */4 * * *"
```

## Suggested Mental Model

Treat the workflows as follows:

- `research`: normal backtest run
- `validation`: backtest run with stricter config and usually a different engine
- `live`: signal generation plus order routing
- `data_fetch_flow`: storage refresh only
- strategy-owned backtest helpers are not part of the long-term callable surface
