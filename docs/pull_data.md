# Fetching and Storing Data

## How `DataManager` Works

`DataManager` is the single gateway for all market data. When you request OHLCV for a symbol it
checks storage tiers in this order:

1. **DuckDB** (`~/.qts/database/qts.duckdb`) — returns immediately if the rows exist.
2. **Parquet cache** (`~/.qts/cache/`) — used for fundamentals (TTL 24 h).
3. **Source fetch** — calls the registered `BaseDataSource` and persists the result back to
   DuckDB/Parquet so the next request is served from disk.

`Config.build()` assembles `DataManager` automatically from the `data_sources:` keys in your
YAML. You rarely call `DataManager` directly; use `data_fetch_flow` instead.

---

## Fetching Data with `data_fetch_flow`

`data_fetch_flow` is the data-only async flow. It resolves your config, builds a `DataManager`,
and downloads the requested asset types and data types into DuckDB.

```python
# Both macOS and Windows
import asyncio
from qts.orchestration.flows.data_fetch_flow import data_fetch_flow

# Fetch US stock OHLCV defined in the config universe
asyncio.run(data_fetch_flow("examples/research_zipline.yaml", ["stock"], ["ohlcv"]))

# Fetch VN stock OHLCV and fundamentals
asyncio.run(data_fetch_flow("my_vn_config.yaml", ["vn_stock"], ["ohlcv", "fundamentals"]))

# Fetch crypto spot and futures OHLCV
asyncio.run(data_fetch_flow("examples/btc_momentum.yaml", ["crypto", "crypto_futures"], ["ohlcv"]))
```

---

## Supported `asset_types` and `data_types` Combinations

| `asset_types` value | `data_types` values | Source key | Notes |
|---|---|---|---|
| `stock` | `ohlcv`, `fundamentals` | `fmp`, `yahoo` | `fmp` supports both; `yahoo` supports `ohlcv` only |
| `vn_stock` | `ohlcv`, `fundamentals` | `vnstock`, `dnse` | `dnse` requires VN IP |
| `vn_futures` | `ohlcv` | `vnstock_futures`, `dnse` | `futures_ohlcv` capability |
| `vn_warrant` | `ohlcv` | `vnstock`, `dnse` | `dnse` expands underlying to concrete codes |
| `crypto` | `ohlcv`, `funding_rates` | `binance` | `funding_rates` not yet wired through `data_fetch_flow` |
| `crypto_futures` | `ohlcv` | `binance_futures` | `futures_ohlcv` capability |

---

## Config Snippets by Asset Class

### US Stocks

```yaml
workflow: research
asset_types: [stock]
universe:
  stock: [AAPL, MSFT, GOOGL, AMZN, META]
data_sources:
  stock: fmp      # or: yahoo
```

### VN Equities (global access)

```yaml
workflow: research
asset_types: [vn_stock]
universe:
  vn_stock: [VN:VNM, VN:VIC, VN:HPG, VN:MBB]
data_sources:
  vn_stock: vnstock   # KBS public API — accessible from any IP
```

### VN Equities (from Vietnamese IP)

```yaml
workflow: research
asset_types: [vn_stock]
universe:
  vn_stock: [VN:VNM, VN:VIC]
data_sources:
  vn_stock: dnse   # requires DNSE_API_KEY / DNSE_API_SECRET and a Vietnamese IP
```

### VN30 Futures

```yaml
workflow: research
asset_types: [vn_futures]
universe:
  vn_futures: [VNF:VN30F2606]
data_sources:
  vn_futures: vnstock_futures   # auto-converts to KRX format for contracts from May 2025
```

### VN Covered Warrants

```yaml
workflow: research
asset_types: [vn_warrant]
universe:
  vn_warrant: [VNW:CVNM2511]
data_sources:
  vn_warrant: vnstock
```

When using `dnse` as the warrant source, you can specify an underlying instead of a concrete
warrant code. DNSE will expand `VNW:VNM` into all currently listed warrant symbols and store the
concrete codes (e.g. `VNW:CVNM2511`).

### Spot Crypto

```yaml
workflow: research
asset_types: [crypto]
universe:
  crypto: ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
data_sources:
  crypto: binance   # requires BINANCE_API_KEY / BINANCE_API_SECRET in .env
```

### Crypto Perpetual Futures

```yaml
workflow: research
asset_types: [crypto_futures]
universe:
  crypto_futures: ["PERP:ETH/USDT", "PERP:BTC/USDT"]
data_sources:
  crypto_futures: binance_futures
```

---

## VN-Specific Notes

### `vnstock` vs `dnse`

| | `vnstock` | `dnse` |
|---|---|---|
| Access | Global (any IP) | Vietnamese IP only |
| Auth | None — KBS public API | `DNSE_API_KEY` + `DNSE_API_SECRET` required |
| Coverage | Equities, warrants, fundamentals | Equities, futures, warrants |
| Default | Yes — use this from outside Vietnam | No — use only from a VN IP |

### VN30 Futures Symbol Format

Old format `VN30F2606` is auto-converted to KRX format `41I1G6000` for contracts expiring
**May 2025 or later**. You can use either format in your YAML — the source adapter normalizes
it transparently. DNSE normalizes futures requests to rolling aliases such as `VNF:VN30F1M`
regardless of whether you pass a dated symbol (`VNF:VN30F2503`).

### VN Warrant Expansion

The `dnse` source accepts a warrant underlying like `VNW:VNM` and automatically expands it
into all currently listed warrant symbols for fetch, cache, and storage. Concrete codes
(e.g. `VNW:CVNM2511`) are stored in DuckDB. `vnstock` requires you to specify the full
concrete code.

### VN Fundamentals Cache

Fundamentals fetched via `vnstock` are cached as Parquet files with a **24-hour TTL**:

```
~/.qts/cache/vn_fundamentals/{ticker}_{annual|quarterly}.parquet
```

To bypass the cache and force a fresh fetch:

```python
from qts.data.manager import DataManager
from qts.data.base import DataType

manager: DataManager = ...   # built by Config.build()
df = await manager.get_fundamentals("VN:VNM", force_refresh=True)
```

The fundamentals schema is tidy long format:
`symbol | report_type | period | fiscal_year | quarter | report_date | item_en | value`

Values from `KQKD` (income), `CDKT` (balance sheet), and `LCTT` (cash flow) are in thousands
of VND. `CSTC` (ratios) values are native units (%, ×, VND/share).

---

## Querying Stored Data Directly from DuckDB

After a fetch run, all OHLCV data lives in `~/.qts/database/qts.duckdb`. You can query it
directly with the `duckdb` Python client:

```python
import duckdb

con = duckdb.connect("~/.qts/database/qts.duckdb")

# List available tables
con.execute("SHOW TABLES").df()

# Spot crypto
df = con.execute("""
    SELECT *
    FROM crypto_prices
    WHERE symbol = 'BTC/USDT'
    ORDER BY date DESC
    LIMIT 10
""").df()
print(df)

# VN stock prices
df = con.execute("""
    SELECT date, symbol, close, volume
    FROM vn_stock_prices
    WHERE symbol = 'VN:VNM'
      AND date >= '2024-01-01'
    ORDER BY date
""").df()
print(df)

# VN30 futures
df = con.execute("""
    SELECT *
    FROM vn_futures_prices
    ORDER BY date DESC
    LIMIT 10
""").df()
print(df)

con.close()
```

All OHLCV tables share the same schema:
`[date: Date, symbol: Utf8, open: f64, high: f64, low: f64, close: f64, volume: f64]`

---

## Prefect Deployment Schedules

Running `python qts/orchestration/serve.py` registers six data-refresh deployments with Prefect.
Each deployment calls `data_fetch_flow` on a cron schedule.

| Deployment name | Asset types | Data types | Cron schedule | Human description |
|---|---|---|---|---|
| `stock-ohlcv-daily` | `["stock"]` | `["ohlcv"]` | `0 21 * * 1-5` | Weekdays at 21:00 UTC (after NYSE close) |
| `vn-stock-ohlcv-daily` | `["vn_stock"]` | `["ohlcv"]` | `0 9 * * 1-5` | Weekdays at 09:00 UTC (after HOSE close) |
| `crypto-ohlcv-daily` | `["crypto"]` | `["ohlcv"]` | `0 0 * * *` | Daily at midnight UTC |
| `crypto-funding-8h` | `["crypto"]` | `["funding_rates"]` | `0 */8 * * *` | Every 8 hours |
| `stock-fundamentals-weekly` | `["stock"]` | `["fundamentals"]` | `0 8 * * 1` | Mondays at 08:00 UTC |
| `vn-stock-fundamentals-weekly` | `["vn_stock"]` | `["fundamentals"]` | `0 8 * * 1` | Mondays at 08:00 UTC |

Register the deployments:

```bash
# Both macOS and Windows (Prefect must be installed: pip install -e ".[orchestration]")
python qts/orchestration/serve.py
```

If Prefect is not installed, the compatibility shim (`qts/orchestration/prefect_compat.py`)
keeps the decorators importable. The deployments are no-ops but the script still runs without
error.

---

## See Also

- [introduction.md](introduction.md) — data source registry table and symbol conventions
- [installation.md](installation.md) — install extras and set environment variables
- [run_backtest.md](run_backtest.md) — run a backtest once data is in DuckDB
- [logging.md](logging.md) — inspect the results after a run
