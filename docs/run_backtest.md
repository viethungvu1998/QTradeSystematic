# Running a Backtest

## Entry Points

QTS exposes two ways to run a backtest:

| Method | When to use |
|---|---|
| `asyncio.run(qts_flow(config_path))` | Primary entry point for all workflows — research, validation, live |
| Direct Python (construct `ResolvedConfig` manually) | Advanced use only; not part of the stable public surface |

There is **no `qts run` CLI** today. You call `qts_flow` from Python or from a shell one-liner.

---

## Complete Research Example

### Step 1 — Write the config

Copy the canonical research config as a starting point:

```bash
# Both macOS and Windows
cp examples/research_zipline.yaml my_backtest.yaml
```

The file already contains a valid research config for US stocks via FMP with Zipline engine:

```yaml
# my_backtest.yaml
workflow: research
asset_types: [stock]
universe:
  stock: [AAPL, MSFT, GOOGL, AMZN, META]
start_date: "2022-01-01"
end_date: "2024-12-31"
initial_capital: 100000
data_sources:
  stock: fmp
storage: duckdb
features:
  indicators:
    - name: rsi
      params: {periods: [14]}
    - name: macd
      params: {}
  forward_returns:
    periods: [21]
strategy:
  type: factor
  params:
    long_quantile: 0.2
    short_quantile: 0.2
backtest_engine: "zipline"
calendar: nyse
rebalance_frequency: monthly
fill_model: next_open
commission:
  model: percentage
  rate: 0.001
slippage_model: fixed
```

### Step 2 — Populate `.env`

For the `fmp` source in fixture mode you do not need API keys. If you switch to a live source
(e.g. `binance`), add the relevant keys to `.env` first. See
[installation.md](installation.md#environment-variables) for the full variable list.

### Step 3 — Run

```bash
# Both macOS and Windows
python -c "
import asyncio
from qts.orchestration.flow import qts_flow
result = asyncio.run(qts_flow('my_backtest.yaml'))
print(result.metrics)
"
```

Or from a script:

```python
# run_backtest.py
import asyncio
from qts.orchestration.flow import qts_flow

result = asyncio.run(qts_flow("my_backtest.yaml"))
print(result.metrics)
print(result.trade_log.shape)
print(result.equity_curve.tail(5))
```

```bash
# Both macOS and Windows
python run_backtest.py
```

### Step 4 — Interpret the output

```python
# result.metrics keys
{
    "sharpe":       1.35,   # annualised Sharpe ratio
    "sortino":      1.92,   # annualised Sortino ratio
    "cagr":         0.24,   # compound annual growth rate (24 %)
    "max_drawdown": -0.15,  # maximum drawdown (-15 %)
    "win_rate":     0.58,   # 58 % of closed trades were profitable
}
```

QTS also writes CSV exports to `~/.qts/exports/backtests/` automatically after every run.

---

## VectorBT Variant (Crypto)

Use `backtest_engine: vectorbt` (or the alias `fast`) for crypto universes. VectorBTProEngine
reads DuckDB/Polars directly and does not require an exchange calendar.

```yaml
# examples/btc_momentum.yaml
workflow: research
asset_types: [crypto]
universe:
  crypto: ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
start_date: "2022-01-01"
end_date: "2024-01-01"
initial_capital: 100000
data_sources:
  crypto: binance
storage: duckdb
features:
  indicators:
    - name: rsi
      params: {periods: [14]}
    - name: roc
      params: {periods: [1, 7, 30]}
strategy:
  type: factor
  params: {}
backtest_engine: vectorbt    # VectorBTProEngine — reads DuckDB/Polars directly
rebalance_frequency: weekly
train_window: 90
slippage_model: fixed
commission:
  model: percentage
  rate: 0.001
# No calendar — crypto trades 24/7
```

```bash
# Both macOS and Windows
python -c "
import asyncio
from qts.orchestration.flow import qts_flow
result = asyncio.run(qts_flow('examples/btc_momentum.yaml'))
print(result.metrics)
"
```

---

## `research` vs `validation` Workflow

Change the `workflow:` key to switch modes.

### `workflow: research`

Normal backtest run. The minimum required keys are `workflow`, `asset_types`, `universe`,
`start_date`, `end_date`, `initial_capital`, `data_sources`, `strategy`, and `backtest_engine`.
`qts_flow` returns a `BacktestResult`.

### `workflow: validation`

Stricter config profile intended for a second-pass evaluation with a different set of
simulation parameters. Validation requires additional keys:

```yaml
workflow: validation
# ... all research keys, plus:
fill_model: next_open          # required
slippage_model: volatility_scaled  # required
commission:
  model: percentage
  rate: 0.001
calendar: nyse                 # required
promotion_gate:
  max_sharpe_degradation: 0.30
```

**What validation does not currently do:**
- It does not automatically rerun the research backtest for comparison.
- It does not enforce `promotion_gate.max_sharpe_degradation`. The gate is parsed into config
  but not checked by the flow. Treat `workflow: validation` as a manually enforced quality gate
  for now.

The flow returns the same `BacktestResult` type as `research`.

---

## Adjusting Simulation Parameters

### Rebalance frequency

```yaml
rebalance_frequency: daily     # rebalance every bar
rebalance_frequency: weekly
rebalance_frequency: monthly   # default
rebalance_frequency: 21        # integer = every N bars
```

### Commission

```yaml
commission:
  model: percentage   # percentage of trade notional
  rate: 0.001         # 0.1 % per trade

commission:
  model: per_trade    # flat fee per trade
  rate: 1.00          # $1 per trade
```

### Slippage

```yaml
slippage_model: fixed              # fixed spread (default 5 bps)
slippage_model: volatility_scaled  # slippage proportional to recent ATR
```

### Fill model (Zipline only)

```yaml
fill_model: next_open   # fill at the next bar's open price (most realistic)
fill_model: immediate   # fill at the current bar's close
fill_model: vwap        # fill at an estimated VWAP
```

---

## Walk-Forward Backtesting

Set `test_start_date` to split the backtest into an in-sample training period and an
out-of-sample evaluation period. The engine trains the model on all bars before
`test_start_date` and reports separate `metrics_is` and `metrics_oos` in `BacktestResult`.

```yaml
workflow: research
start_date: "2019-01-01"
test_start_date: "2023-01-01"   # OOS period starts here
end_date: "2024-12-31"
```

```python
result = asyncio.run(qts_flow("my_backtest.yaml"))
print("In-sample:", result.metrics_is)
print("Out-of-sample:", result.metrics_oos)
```

Walk-forward signal generation across rolling windows is handled inside the engine path via
`walk_forward_signals()` when `pipeline` and `ohlcv` are supplied. This happens automatically
when `workflow` is `research` or `validation`.

---

## See Also

- [pull_data.md](pull_data.md) — make sure data is in DuckDB before running
- [logging.md](logging.md) — interpret and visualize `BacktestResult`
- [sweep.md](sweep.md) — automate many backtest runs to tune parameters
- [creating_strategy.md](creating_strategy.md) — write a custom strategy
