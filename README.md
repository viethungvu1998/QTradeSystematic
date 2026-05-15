# QTradeSystematic

A config-driven systematic trading platform for equities and crypto. Define your universe, strategy, and risk parameters in a single YAML file — the framework handles data ingestion, feature engineering, backtesting, and live execution.

---

## Features

- **Unified instrument model** — stocks and crypto share the same data contract
- **Multi-source data pipeline** — FMP and Yahoo Finance for equities; Binance REST + WebSocket for crypto; all normalised to a common OHLCV schema stored in DuckDB
- **Dual backtest engines** — vectorised batch (`VectorBTProEngine`) and bar-by-bar look-ahead-safe (`ZiplineEngine`)
- **Pluggable simulation** — mix-and-match fill models, slippage models, commissions, and trading calendars per run
- **ML and stat-arb strategies** — XGBoost factor model and pairs/cointegration strategy out of the box
- **Live execution** — Moomoo (Futu OpenD) for stocks, Binance for crypto; concurrent order dispatch via `OrderRouter`
- **Prefect orchestration** — single `qts_flow` entry point; all scheduling and routing resolved from config at deploy time
- **Plugin registry** — extend any layer (data source, storage, feature, strategy, engine, broker) with a single decorator

---

## Architecture

```
qts/
├── core/           # Instrument, Order, Portfolio, Event bus, Plugin registry
├── data/           # Sources (FMP, Yahoo, Binance) → DuckDB / Parquet / Zipline bundles
├── research/       # Features (technical, fundamental, onchain) + strategies + backtesting
├── execution/      # Live brokers (Moomoo, Binance) + OrderRouter + PositionSync
├── orchestration/  # Prefect flow + task library
└── config/         # YAML loader + registry resolver
```

### Data flow

```
FMP / Yahoo ──┐                          ┌──► Zipline bundle ──► ZiplineEngine
              ├──► DataManager ──► DuckDB─┤
Binance ───────┘                          └──► FeaturePipeline ──► Strategy.generate_signals()
                                                                           │
                                                              VectorBTProEngine
                                                                           │
                                                                    BacktestResult
                                                                           │
                                                               OrderRouter.execute()
                                                                │               │
                                                          MoomooBroker    BinanceBroker
```

---

## Requirements

- Python ≥ 3.13

---

## Installation

```bash
# Core dependencies only
pip install -e .

# Add data sources (httpx)
pip install -e ".[data]"

# Add research extras (XGBoost)
pip install -e ".[research]"

# Add live execution (futu-api, binance-connector)
pip install -e ".[execution]"

# Add orchestration (Prefect)
pip install -e ".[orchestration]"

# Everything including dev tools
pip install -e ".[all]"
```

---

## Configuration

All behaviour is controlled by a single YAML file. The `workflow` key selects one of three modes:

| `workflow` | What runs |
|---|---|
| `research` | Data → features → `VectorBTProEngine` → `BacktestResult` |
| `validation` | Same as research, then re-runs through `ZiplineEngine` and checks a Sharpe degradation gate |
| `live` | Data → features → `PositionSync` → `OrderRouter` → live/paper brokers |

### Full config schema

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
  model: percentage         # percentage | per_trade
  rate:  0.001
calendar: nyse              # nyse | hkex | crypto

# ── Brokers (live only) ────────────────────────────────────
brokers:
  stock:  moomoo
  crypto: binance
  binance_mode: demo        # demo (testnet) | live (production)

# ── Schedule (live only) ───────────────────────────────────
schedule:
  stock:  "0 16 * * 1-5"   # daily after NYSE close
  crypto: "0 */4 * * *"    # every 4 hours

# ── Promotion gate (validation only) ──────────────────────
promotion_gate:
  max_sharpe_degradation: 0.30
```

### Environment variables

Credentials are read from the environment at connection time — never stored in config files.

| Variable | Used by |
|---|---|
| `FMP_API_KEY` | FMP data source |
| `BINANCE_DEMO_TRADING_API_KEY` | Binance testnet data + broker |
| `BINANCE_DEMO_TRADING_SECRET_KEY` | Binance testnet broker |
| `BINANCE_TRADING_KEY` | Binance production broker |
| `BINANCE_TRADING_SECRET_KEY` | Binance production broker |

Copy `.env.example` to `.env` and fill in the values (`.env` is git-ignored).

---

## Usage

### Research backtest

```python
from qts.config import Config
from qts.orchestration.flow import qts_flow

config = Config.build("configs/research.yaml")
result = qts_flow(config)
print(result.sharpe, result.cagr, result.max_drawdown)
```

### Custom strategy

```python
from qts.core.registry import Registry
from qts.research.strategies.base import BaseStrategy, SignalFrame

@Registry.register_strategy("my_strategy")
class MyStrategy(BaseStrategy):
    def generate_signals(self, df) -> SignalFrame:
        ...
```

Then reference it in your YAML:

```yaml
strategy:
  type: my_strategy
  params: {}
```

### Custom broker

```python
from qts.core.registry import Registry
from qts.execution.base import BaseBroker

@Registry.register_broker("my_broker")
class MyBroker(BaseBroker):
    async def place_order(self, order): ...
    async def get_positions(self): ...
```

---

## Testing

```bash
# Unit and integration tests
pytest

# Include paper-trading tests (requires live broker credentials)
pytest -m paper

# With coverage
pytest --cov=qts
```

Test tiers:
- **Unit** — no external dependencies, run on every commit
- **Integration** — real DuckDB / Parquet / Zipline I/O, no network calls
- **Paper** (`@pytest.mark.paper`) — requires paper/testnet broker accounts

---

## Tech stack

| Layer | Library |
|---|---|
| Data storage | DuckDB, Apache Parquet (via PyArrow) |
| DataFrames | Polars |
| Backtest (fast) | VectorBT Pro |
| Backtest (normal) | Zipline |
| ML strategy | XGBoost |
| Stocks broker | futu-api (Moomoo / Futu OpenD) |
| Crypto broker | binance-connector |
| Orchestration | Prefect |
| HTTP client | httpx |
| Build | Hatchling |
| Linting | Ruff |

---

## License

Proprietary. All rights reserved.
