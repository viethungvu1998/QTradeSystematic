# Observability and Logging

---

## Part A — Structured Backtest Results

### `BacktestResult` Fields

`qts_flow()` and `run_backtest()` both return a `BacktestResult` dataclass. All fields are
immutable (`frozen=True`).

| Field | Type | Description |
|---|---|---|
| `engine_name` | `str` | Registry key of the engine that produced the result (`"vectorbt"`, `"zipline"`) |
| `metrics` | `dict[str, float]` | Aggregate performance metrics for the full period |
| `metrics_is` | `dict[str, float]` | In-sample metrics (populated only with a `test_start_date` split) |
| `metrics_oos` | `dict[str, float]` | Out-of-sample metrics (populated only with a `test_start_date` split) |
| `returns` | `pl.DataFrame` | Daily portfolio returns — schema: `[date: Date, portfolio_return: Float64]` |
| `equity_curve` | `pl.DataFrame` | Daily equity — schema: `[date: Date, equity: Float64]` |
| `signals` | `pl.DataFrame` | Full SignalFrame emitted by the strategy (all dates, all symbols) |
| `trade_log` | `pl.DataFrame` | One row per closed round-trip trade |
| `portfolio_snapshots` | `pl.DataFrame` | Timestamped portfolio state snapshots |

Standard keys present in `metrics`:

| Key | Description |
|---|---|
| `sharpe` | Annualised Sharpe ratio |
| `sortino` | Annualised Sortino ratio |
| `cagr` | Compound annual growth rate |
| `max_drawdown` | Maximum drawdown (negative number) |
| `win_rate` | Fraction of closed trades with positive P&L |

### `trade_log` Schema

Each row represents one closed round-trip trade.

| Column | Type | Description |
|---|---|---|
| `ticker` | `String` | Symbol |
| `entry_time` | `Datetime` | Bar timestamp when the position was opened |
| `exit_time` | `Datetime` | Bar timestamp when the position was closed |
| `start_price` | `Float64` | Entry price |
| `end_price` | `Float64` | Exit price |
| `quantity` | `Float64` | Shares / contracts |
| `profit_pct` | `Float64` | Round-trip P&L as a fraction |
| `fee` | `Float64` | Total commission paid |
| `side` | `String` | `"BUY"` or `"SELL"` |

### `portfolio_snapshots` Schema

| Column | Type | Description |
|---|---|---|
| `timestamp` | `Datetime` | Snapshot time |
| `tokens` | `List[Struct]` | List of `{token, quantity, avg_buy_price, current_price}` |
| `equity` | `Float64` | Total portfolio equity at this timestamp |

### Live-Mode Observability Types

For `workflow: live`, `qts/core/observability.py` provides frozen dataclasses that mirror the
`portfolio_snapshots` structure but are returned in the live result dict:

```python
from qts.core.observability import ClosedTrade, TokenSnapshot, PortfolioSnapshot

# ClosedTrade — one completed round-trip
trade.ticker        # str
trade.entry_time    # datetime
trade.exit_time     # datetime
trade.start_price   # Decimal
trade.end_price     # Decimal
trade.quantity      # Decimal
trade.profit_pct    # Decimal
trade.fee           # Decimal
trade.side          # OrderSide

# TokenSnapshot — one held position inside a PortfolioSnapshot
snap.token          # str
snap.quantity       # Decimal
snap.avg_buy_price  # Decimal
snap.current_price  # Decimal

# PortfolioSnapshot — point-in-time portfolio state
snap.timestamp      # datetime
snap.tokens         # tuple[TokenSnapshot, ...]
snap.equity         # Decimal
```

### Extracting and Printing Metrics

```python
import asyncio
from qts.orchestration.flow import qts_flow

result = asyncio.run(qts_flow("examples/btc_momentum.yaml"))

# Core metrics
print(result.metrics)
# {'sharpe': 1.23, 'sortino': 1.85, 'cagr': 0.31, 'max_drawdown': -0.18, 'win_rate': 0.54}

# Daily returns as a Pandas Series (for third-party tools)
returns_pd = result.returns.to_pandas().set_index("date")["portfolio_return"]

# Trade log
print(result.trade_log.head(5))

# Portfolio equity curve
print(result.equity_curve.tail(10))

# Signals (all dates)
print(result.signals.filter(result.signals["signal"] != 0).head(20))
```

After a backtest run, QTS automatically exports CSV files to `~/.qts/exports/backtests/`:
- `{engine}_{timestamp}_trade_log.csv`
- `{engine}_{timestamp}_snapshots.csv`
- `{engine}_{timestamp}_tearsheet.pdf` (when `reporting` extra is installed)

### Performance Tearsheets with pyfolio-reloaded

Install the reporting extra:

```bash
# Both macOS and Windows
pip install -e ".[reporting]"
```

Generate a full tearsheet from the result's daily returns:

```python
import pyfolio as pf
import asyncio
from qts.orchestration.flow import qts_flow

result = asyncio.run(qts_flow("examples/btc_momentum.yaml"))

returns_pd = (
    result.returns
    .to_pandas()
    .set_index("date")["portfolio_return"]
)
returns_pd.index = returns_pd.index.to_pydatetime()

pf.create_full_tear_sheet(returns_pd)
```

QTS also saves a tearsheet PDF automatically during `_export_backtest_observability` if
`pyfolio-reloaded` is installed. The file lands at
`~/.qts/exports/backtests/{engine}_{timestamp}_tearsheet.pdf`.

---

## Part B — Runtime Logging

### Standard Python Logging

QTS uses the Python standard library `logging` module throughout. No third-party logging
framework is required.

To see all debug output during a run:

```python
import logging
import asyncio
from qts.orchestration.flow import qts_flow

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

result = asyncio.run(qts_flow("examples/btc_momentum.yaml"))
```

To set log level via an environment variable (useful in `.env`):

```bash
# .env
LOG_LEVEL=DEBUG
```

```python
import logging
import os

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
```

For production or scheduled runs, redirect output to a file:

```python
logging.basicConfig(
    level=logging.INFO,
    filename="qts.log",
    filemode="a",
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
```

### Prefect Task Logging

When you run `qts/orchestration/serve.py` with Prefect installed, every `@flow` and `@task`
decorated function emits structured logs that appear in the Prefect UI. The `log_prints=True`
flag on `qts_flow` forwards all `print()` calls to Prefect's task logger.

Start the Prefect UI to view live task logs:

```bash
# Both macOS and Windows
prefect server start
```

Then in a separate terminal, register deployments:

```bash
# Both macOS and Windows
python qts/orchestration/serve.py
```

Each of the six data-refresh deployments (`stock-ohlcv-daily`, `vn-stock-ohlcv-daily`, etc.)
appears in the Prefect UI with its schedule, run history, and per-task log stream.

If Prefect is not installed, `qts/orchestration/prefect_compat.py` provides no-op shims so the
decorators remain importable and logging falls back to standard Python `logging`.

---

## See Also

- [run_backtest.md](run_backtest.md) — run a backtest and get a `BacktestResult`
- [sweep.md](sweep.md) — log sweep metrics to MLflow
- [pull_data.md](pull_data.md) — Prefect deployment schedules for data refresh
