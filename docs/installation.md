# Installation

QTS uses [uv](https://docs.astral.sh/uv/) for environment and dependency management. `uv`
handles Python installation, virtual environments, and package installs in one tool — no
separate Python installer or `pip` invocation needed.

## Prerequisites

| Requirement | Notes |
|---|---|
| Git | For cloning the repo |
| vectorbt.pro wheel | Must be present at `lib/vectorbt.pro-main/` before installing the `research` extra |

Python 3.13 is installed by `uv` in the next step; you do not need to install it separately.

---

## Step 1 — Install uv

### macOS

```bash
# macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal, then verify:

```bash
# macOS
uv --version
```

### Windows

```powershell
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your terminal, then verify:

```powershell
# Windows (PowerShell)
uv --version
```

---

## Step 2 — Clone the Repo

```bash
# Both macOS and Windows
git clone <repo-url> QTradeSystematic
cd QTradeSystematic
```

---

## Step 3 — Install Everything

One command installs Python 3.13, creates `.venv`, and installs all extras:

```bash
# Both macOS and Windows — recommended for development
uv sync --all-extras
```

`uv sync` reads `pyproject.toml`, pins Python 3.13 automatically, and resolves all
dependencies. The `.venv` directory is created at the repo root.

> **vectorbt.pro wheel:** the `research` extra references the local wheel at
> `lib/vectorbt.pro-main/`. That directory must exist before running `uv sync --all-extras`.
> If it is missing the install will fail with a path-not-found error.

### Selective installs (optional)

If you want a smaller environment, install only the extras you need:

| Group | Installs | When you need it |
|---|---|---|
| `dev` | pytest, pytest-asyncio, pytest-cov, ruff | Always — tests and linting |
| `data` | httpx | Any live data source (Binance, DNSE) |
| `vn` | vnstock | VN equities (`VN:`), VN futures (`VNF:`), VN warrants (`VNW:`) |
| `research` | matplotlib, scikit-learn, xgboost, vectorbtpro wheel | VectorBTProEngine; ML strategies |
| `zipline` | zipline-reloaded | ZiplineReloadedEngine |
| `execution` | binance-connector, futu-api, python-dotenv | Live order routing |
| `orchestration` | prefect, prefect-redis, asyncpg, redis | Prefect deployments via `serve.py` |
| `tracking` | mlflow, psycopg2-binary | MLflow experiment tracking |
| `tuning` | optuna | Hyperparameter sweeping |
| `reporting` | pyfolio-reloaded | Performance tearsheets |
| `all` | everything above | Recommended for development |

```bash
# Both macOS and Windows — examples
uv sync --extra dev                        # tests + linting only
uv sync --extra dev --extra data           # + live data sources
uv sync --extra dev --extra data --extra vn   # + VN equities/futures
uv sync --extra dev --extra research       # + VectorBT + ML
```

---

## Step 4 — Environment Variables

Credentials must never be hardcoded. Create a `.env` file at the repo root (it is git-ignored):

```bash
# macOS
touch .env
```

```powershell
# Windows (PowerShell)
New-Item .env -ItemType File
```

Populate `.env` with the variables below. Only add the variables for the sources you actually
use.

```bash
# .env — never commit this file

# Binance spot and futures (binance / binance_futures sources)
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# DNSE — only needed when running from a Vietnamese IP
DNSE_API_KEY=your_dnse_key_here
DNSE_API_SECRET=your_dnse_secret_here

# Override default storage root (optional; defaults to ~/.qts)
QTS_ROOT=/path/to/custom/data/root
```

| Variable | Required by | Default | Notes |
|---|---|---|---|
| `BINANCE_API_KEY` | `binance`, `binance_futures` sources | — | Binance API key |
| `BINANCE_API_SECRET` | `binance`, `binance_futures` sources | — | Binance API secret |
| `DNSE_API_KEY` | `dnse` source | — | DNSE OpenAPI key; VN IP only |
| `DNSE_API_SECRET` | `dnse` source | — | DNSE OpenAPI secret; VN IP only |
| `QTS_ROOT` | all | `~/.qts` | Root directory for DuckDB, cache, and bundles |

`python-dotenv` loads `.env` automatically when adapters call `from_env()`. If a required
variable is missing and the source supports `from_env()`, `Config.build()` falls back to the
default constructor so fixture-based tests still work without credentials.

---

## Step 5 — Verify the Installation

```bash
# Both macOS and Windows
uv run pytest tests/ -x
```

`uv run` activates the managed venv automatically — you do not need to `source .venv/bin/activate`
first. A successful run prints:

```
========================= N passed in X.XXs =========================
```

To skip paper-tier tests (which need live Binance testnet or Futu OpenD credentials):

```bash
# Both macOS and Windows
uv run pytest tests/ -x -m "not paper"
```

---

## Running Any Command with uv

Prefix any command with `uv run` to run it inside the managed venv:

```bash
# Both macOS and Windows
uv run python -c "from qts.orchestration.flow import qts_flow; print('ok')"
uv run ruff check qts/
uv run python qts/orchestration/serve.py
```

Alternatively, activate the venv once per terminal session and then run commands directly:

```bash
# macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

---

## See Also

- [introduction.md](introduction.md) — asset classes, data sources, and storage layout
- [run_backtest.md](run_backtest.md) — run your first backtest after installing
- [pull_data.md](pull_data.md) — fetch market data into DuckDB
