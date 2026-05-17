# FEATURES

Current capability inventory for the repository as it exists today.

## Config and Registry

- Typed YAML parsing into `BacktestConfig`
- Required-key validation by workflow
- Engine alias normalization:
  - `fast` -> `vectorbt`
  - `normal` -> `zipline`
- Registry-based resolution for:
  - data sources
  - features
  - strategies
  - engines
  - brokers
  - fill models
  - slippage models
  - commission models
  - calendars

Direction for cleanup:

- keep using the registry seam uniformly wherever it already exists
- storage and built-in features should resolve through registries, not hardcoded constructors
- preserve the existing YAML shape while normalizing the internals

## Asset Coverage

Implemented at the model and routing layer:

- `stock`
- `vn_stock` — equities on HOSE/HNX/UPCOM, prefix `VN:`
- `vn_futures` — VN30 index futures, prefix `VNF:`
- `vn_warrant` — covered warrants (chứng quyền), prefix `VNW:`
- `crypto`
- `crypto_futures`
- `commodity` as a reserved enum value

Implemented end to end through the top-level flows:

- `stock`
- `vn_stock`
- `vn_futures`
- `vn_warrant`
- `crypto`
- `crypto_futures`

## Data Sources

Registered source keys:

| Key | Supports | Notes |
|---|---|---|
| `fmp` | `ohlcv`, `fundamentals` | fixture-only |
| `yahoo` | `ohlcv` | fixture-only |
| `dnse` | `ohlcv`, `futures_ohlcv` | `from_env()` — geo-restricted to VN IP |
| `vnstock` | `ohlcv`, `fundamentals` | `from_env()` — KBS public API, global access |
| `vnstock_futures` | `futures_ohlcv` | `from_env()` — KBS public API, global access |
| `binance` | `ohlcv` | `from_env()` |
| `binance_futures` | `futures_ohlcv` | `from_env()` |

VN-specific notes:

- `vnstock` covers VN equities (`VN:`), VN indices (`VN:VN30` etc.), and covered warrants (`VNW:`).
- `vnstock_futures` covers VN30 index futures (`VNF:`); auto-converts old `VN30F2606` format to KRX `41I1G6000` for contracts from May 2025 onward.
- `dnse` serves as an alternative VN OHLCV source when running from a Vietnamese IP; uses DNSE OpenAPI auth via `DNSE_API_KEY` / `DNSE_API_SECRET`.
- `dnse` accepts VN stock symbols like `VN:VNM`, VN futures requests like `VNF:VN30F2503`, and VN warrant-underlying requests like `VNW:VNM`.
- `dnse` normalizes VN futures requests to DNSE's rolling aliases such as `VNF:VN30F1M`.
- `dnse` expands warrant-underlying requests into concrete warrant symbols and stores the concrete codes, for example `VNW:CVNM2511`.
- All sources normalize outputs into Polars frames with the canonical OHLCV schema.

## Storage

- DuckDB primary storage
- Parquet cache storage
- Local bundle adapter
- Zipline bundle adapter
- Persistent path helpers under `qts/utils/paths.py`

## Feature Engineering

Pipeline behavior:

- preprocessing occurs before any feature plugin
- features can declare `requires_fundamentals()` and receive bound data through `with_fundamentals()`
- feature plugins must append columns, not remove the existing OHLCV columns

Built-in high-level features:

- `technical`
- `fundamental`
- `onchain`
- `forward_returns`

Other registered feature plugins:

- `vn_fundamental`

Indicator plugins:

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

Research helper modules also exist for:

- portfolio tearsheets
- benchmark comparison
- factor IC analysis
- rolling correlation analysis
- feature importance

## Strategies

Registered strategy keys:

- `factor`
- `ml_factor`
- `stat_arb`

Supporting modules exist for:

- ML training wrappers
- portfolio construction functions
- pair selection and spread signal generation
- stat-arb universe screening

Structure rules:

- family `base.py` and `core.py` modules are appropriate only when a strategy family has real shared behavior
- `factor/` and `stat_arb/` are the current families that justify this structure
- do not add empty `cross_sectional/` family scaffolding before the first concrete strategy exists

## Backtesting

Shared backtest contracts:

- `BacktestConfig`
- `BacktestResult`
- `walk_forward_signals()`
- `run_backtest_frame()`

Registered engines:

- `vectorbt`
- `fast`
- `zipline`
- `normal`

Other backtest support:

- fill models: `immediate`, `next_open`, `vwap`
- slippage models: `fixed`, `volatility_scaled`
- commission models: `percentage`, `per_trade`
- calendars: `nyse`, `hkex`, `crypto`

Supported architecture:

- engines are the only supported simulation entrypoints
- strategy packages may keep diagnostics and research helpers, but not alternate public backtest APIs
- stat-arb now runs through the common engine path; no dedicated strategy-specific batch runner remains

## Execution

Registered broker keys:

- `moomoo`
- `binance`
- `dnse` in `qts/execution/brokers/dnse.py`

Execution utilities:

- `OrderRouter`
- `PositionSync`

Current status:

- `MoomooBroker` works as a fixture-friendly adapter unless a client is injected
- `BinanceBroker` can build a real client with `from_env()`
- `DNSEBroker` is implemented, but `qts.execution.brokers.__init__` does not auto-import it yet

## Orchestration

Implemented:

- `qts_flow(config_path)`
- `data_fetch_flow(config_path, asset_types, data_types)`
- shared runtime assembly in `qts/orchestration/runtime.py`
- Prefect compatibility shim for environments without Prefect
- deployment registration script in `qts/orchestration/serve.py`

Current gaps:

- no CLI wrapper
- no automatic validation promotion gate
- no automatic validation comparison between research and validation runs

## Tests

The repository includes coverage for:

- config parsing and component resolution
- data routing and storage
- feature engineering
- strategies
- backtest engines
- orchestration flows
- paper-style broker tests
