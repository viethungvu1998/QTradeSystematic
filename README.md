# QTradeSystematic

Config-driven research and execution framework for systematic trading across stocks, VN stocks, VN warrants, VN futures, spot crypto, and crypto futures.

## Current Status

This repository is a working framework, not a finished turnkey trading product.

- The typed config layer, registry, feature pipeline, backtest engines, and orchestration flows are implemented.
- The package is primarily exercised through tests and direct Python usage.
- Some adapters are production-oriented, while others are still fixture-friendly scaffolds.

The most important current-state details are:

- There is no CLI entry point yet.
- `qts_flow` is `async` and takes a config path string, not a prebuilt config object.
- `validation` is a config mode with extra required fields, but the runtime does not automatically compare research and validation results or enforce `promotion_gate`.
- `vn_warrant`, `vn_futures`, and `crypto_futures` are wired through the top-level flows.

## What Exists Today

### Core runtime

- Typed YAML loader: `qts.config.load_config`
- Runtime resolver: `qts.config.Config.build`
- Runtime assembly helpers: `qts.orchestration.runtime`
- Registry-driven plugins for data sources, features, strategies, engines, brokers, and simulation models
- Unified `AssetType` routing from symbol format:
  - `AAPL` -> `stock`
  - `VN:VNM` -> `vn_stock`
  - `VNW:VNM` -> `vn_warrant`
  - `VNF:VN30F2503` -> `vn_futures`
  - `BTC/USDT` -> `crypto`
  - `PERP:BTC/USDT` -> `crypto_futures`

### Data layer

- `DataManager` routes requests by `(AssetType, DataType)`
- Persistent storage:
  - DuckDB at `~/.qts/database/qts.duckdb`
  - Parquet cache at `~/.qts/cache/`
  - Bundle files at `~/.qts/bundles/`
- Registered data sources:
  - `fmp`
  - `yahoo`
  - `dnse`
  - `binance`
  - `binance_futures`

Current adapter status:

- `vnstock`, `vnstock_futures`, and `dnse` can build real clients via `from_env()`
- `binance` and `binance_futures` expose `from_env()` on the adapter, but `Config.build()` currently instantiates them via the default constructor
- `fmp` and `yahoo` are still fixture-friendly adapters that expect injected payloads
- `dnse` supports live VN equity data, rolling VN30 futures aliases, and warrant-underlying expansion when running from a Vietnamese IP
- `Config.build()` prefers `from_env()` only for `dnse`, `vnstock`, and `vnstock_futures`, with safe fallback to default constructors where applicable

### Research layer

- Feature preprocessing via `preprocess_ohlcv()`
- Built-in features:
  - `technical`
  - `fundamental`
  - `vn_fundamental`
  - `onchain`
  - `forward_returns`
- Indicator plugins:
  - `rsi`
  - `roc`
  - `macd`
  - `adx`
  - `atr`
  - `bollinger`
  - `hist_vol`
  - `obv`
  - `volume_ratio`
  - `zscore`
- Strategies:
  - `factor`
  - `ml_factor`
  - `stat_arb`
- Analysis utilities under:
  - `qts/research/portfolio_analysis/`
  - `qts/research/statistical_analysis/`

Strategy architecture direction:

- the public strategy seam is `BaseStrategy.generate_signals(data) -> SignalFrame`
- family-level shared modules are earned only when shared behavior is real
- `factor/` and `stat_arb/` are the current families that justify shared bases/utilities
- stat-arb is being converged onto the same universe-in, standard-signal-out contract as the other strategies

### Backtesting

- `vectorbt` / `fast` -> `VectorBTProEngine`
- `zipline` -> `ZiplineReloadedEngine`
- YAML alias `normal` currently normalizes to `zipline` at config-load time
- Walk-forward signal generation via `qts/research/backtest/_runner.py`

Supported direction:

- engines are the canonical simulation entrypoints
- strategy packages may keep diagnostics, but not alternate public backtest APIs
- no strategy-specific backtest runner remains in the supported surface

### Execution

- `MoomooBroker` is currently fixture-friendly unless a client is injected
- `BinanceBroker` supports real credentials through `from_env()`
- `DNSEBroker` is implemented in `qts/execution/brokers/dnse.py`, but `qts.execution.brokers.__init__` does not auto-import it yet
- `OrderRouter` executes concurrently across broker instances
- `PositionSync` converts target weights into rebalance orders

### Orchestration

- Main flow: `qts/orchestration/flow.py`
- Data-only flow: `qts/orchestration/flows/data_fetch_flow.py`
- Runtime composition helpers: `qts/orchestration/runtime.py`
- Prefect compatibility shim: `qts/orchestration/prefect_compat.py`
- Deployment registration: `qts/orchestration/serve.py`

## Installation

```bash
pip install -e .
pip install -e ".[data]"
pip install -e ".[research]"
pip install -e ".[zipline]"
pip install -e ".[execution]"
pip install -e ".[orchestration]"
pip install -e ".[all]"
```

Python requirement: `>=3.13`

## Configuration

Supported workflow values:

- `research`
- `validation`
- `live`

Minimal research example:

```yaml
workflow: research
asset_types: [stock, crypto]
universe:
  stock: [AAPL]
  crypto: [BTC/USDT]
start_date: "2024-01-01"
end_date: "2024-03-20"
initial_capital: 100000
data_sources:
  stock: fmp
  crypto: binance
storage: duckdb
features:
  indicators:
    - name: rsi
      params:
        periods: [14]
    - name: macd
      params: {}
  forward_returns:
    periods: [1, 5]
strategy:
  type: factor
  params:
    long_quantile: 0.7
    short_quantile: 0.3
backtest_engine: vectorbt
train_window: 252
rebalance_frequency: monthly
```

Notes:

- `features.technical: true` still works as a backward-compatible shortcut.
- `qts_flow()` builds runtime collaborators from `ResolvedConfig`, then binds fundamentals into the feature pipeline only when a feature explicitly requests them.
- all strategy-specific parameters, including stat-arb knobs, belong under `strategy.params`; there is no supported parallel stat-arb-only backtest config surface.
- `promotion_gate` is parsed for `validation` configs, but it is not enforced inside `qts_flow`.
- `brokers.binance_mode` is stored in config, but `Config.build()` does not yet use it to instantiate brokers automatically.
- `data_fetch_flow()` accepts `stock`, `vn_stock`, `vn_warrant`, `vn_futures`, `crypto`, and `crypto_futures` in `asset_types` and resolves symbols through the same routing helpers used by the main flow.
- example configs document schema and wiring, but real execution still requires working adapter clients or injected fixture payloads.

## Usage

Run the main flow from Python:

```python
import asyncio

from qts.orchestration.flow import qts_flow

result = asyncio.run(qts_flow("examples/research_zipline.yaml"))
print(result.metrics)
```

Run the data-only flow:

```python
import asyncio

from qts.orchestration.flows.data_fetch_flow import data_fetch_flow

asyncio.run(
    data_fetch_flow(
        "examples/research_zipline.yaml",
        ["stock"],
        ["ohlcv"],
    )
)
```

Run Prefect-style deployment registration:

```bash
python qts/orchestration/serve.py
```

## Examples

- `examples/research_zipline.yaml`
- `examples/btc_momentum.yaml`

## Testing

```bash
pytest
pytest -m paper
pytest --cov=qts
```

Test tiers:

- Unit: pure logic and validation
- Integration: DuckDB, Parquet, bundles, engines
- Paper: sandbox-style broker coverage where available

## Related Docs

- `ARCHITECTURE.md`
- `WORKFLOWS.md`
- `FEATURES.md`
- `AGENTS.md`

## License

Proprietary. All rights reserved.
