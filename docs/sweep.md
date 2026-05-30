# Parameter Sweeping with Optuna

Parameter sweeping runs many backtest configurations in order to find the combination of
hyperparameters — lookback windows, quantile thresholds, rebalance frequencies, commission
rates — that produces the best out-of-sample performance. QTS integrates with
[Optuna](https://optuna.org/) for this workflow.

---

## Install the Tuning Extra

```bash
# Both macOS and Windows (venv must be active)
pip install -e ".[tuning]"
```

This adds `optuna>=4.2.0` to your environment. You do not need to reinstall the rest of the
package.

---

## Basic Sweep Script

The script below shows the full Optuna loop: an `objective` function that builds a temp YAML
from trial-suggested params, runs a backtest, and returns a scalar metric.

```python
# scripts/sweep_momentum.py
import asyncio
import tempfile
import textwrap

import optuna

from qts.orchestration.flow import qts_flow


def objective(trial: optuna.Trial) -> float:
    """Run one backtest configuration and return the Sharpe ratio."""

    # 1. Suggest hyperparameters
    roc_period = trial.suggest_int("roc_period", 5, 63)
    long_q = trial.suggest_float("long_quantile", 0.1, 0.3)
    short_q = trial.suggest_float("short_quantile", 0.1, 0.3)
    train_window = trial.suggest_int("train_window", 60, 252)

    # 2. Write a temporary YAML config
    config_yaml = textwrap.dedent(f"""
        workflow: research
        asset_types: [crypto]
        universe:
          crypto: ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        start_date: "2022-01-01"
        end_date: "2023-12-31"       # in-sample window only
        initial_capital: 100000
        data_sources:
          crypto: binance
        storage: duckdb
        features:
          indicators:
            - name: roc
              params:
                periods: [{roc_period}]
        strategy:
          type: factor
          params:
            long_quantile: {long_q}
            short_quantile: {short_q}
        backtest_engine: vectorbt
        train_window: {train_window}
        rebalance_frequency: weekly
        slippage_model: fixed
        commission:
          model: percentage
          rate: 0.001
    """)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(config_yaml)
        config_path = f.name

    # 3. Run the backtest and extract the metric
    try:
        result = asyncio.run(qts_flow(config_path))
        sharpe = result.metrics.get("sharpe", float("-inf"))
    except Exception:
        sharpe = float("-inf")   # penalise failed runs

    return sharpe


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Best params:", study.best_params)
    print("Best Sharpe:", study.best_value)
```

Run it:

```bash
# Both macOS and Windows
python scripts/sweep_momentum.py
```

---

## Look-Ahead Bias Warning

**Never sweep on the same window you will use to report final performance.**

If you optimize `roc_period` and `long_quantile` over the full 2022–2024 date range and then
report the Sharpe ratio from that same range, the reported number is inflated by
in-sample fitting. The correct approach:

1. Designate a **held-out validation window** (e.g. 2024-01-01 to 2024-12-31) and never
   include it in any objective function.
2. Run `study.optimize()` with `end_date` set to the last day *before* your validation window.
3. After `study.optimize()` finishes, set `start_date` / `end_date` to the validation window,
   use `study.best_params` verbatim, and run a single final backtest. That is the number you
   report.

---

## MLflow Integration

Install the tracking extra to log each trial to MLflow:

```bash
# Both macOS and Windows
pip install -e ".[tracking]"
```

Add `mlflow` calls inside `objective`:

```python
import mlflow

def objective(trial: optuna.Trial) -> float:
    roc_period = trial.suggest_int("roc_period", 5, 63)
    long_q = trial.suggest_float("long_quantile", 0.1, 0.3)

    # ... build config_yaml and run backtest as above ...

    with mlflow.start_run():
        mlflow.log_params({
            "roc_period": roc_period,
            "long_quantile": long_q,
        })
        mlflow.log_metrics({
            "sharpe": result.metrics.get("sharpe", 0.0),
            "cagr": result.metrics.get("cagr", 0.0),
            "max_drawdown": result.metrics.get("max_drawdown", 0.0),
        })

    return result.metrics.get("sharpe", float("-inf"))
```

Start the MLflow UI to inspect results:

```bash
# Both macOS and Windows
mlflow ui
# Opens at http://127.0.0.1:5000
```

---

## Tips

- **Parallelize trials.** Optuna supports multi-process and distributed backends. For local
  machines, pass `n_jobs=-1` to `study.optimize()` to use all CPU cores. Make sure each trial
  writes to a separate temp file.
- **Prune unpromising trials.** Use `optuna.pruners.MedianPruner()` when your objective
  supports intermediate values (e.g. yearly Sharpe during a long backtest).
- **Persist the study.** Use an SQLite backend so you can resume interrupted sweeps:
  ```python
  study = optuna.create_study(
      direction="maximize",
      storage="sqlite:///sweep.db",
      study_name="btc_momentum_v1",
      load_if_exists=True,
  )
  ```

---

## See Also

- [run_backtest.md](run_backtest.md) — understand `BacktestResult.metrics` keys
- [logging.md](logging.md) — extract and visualize backtest results
- [creating_strategy.md](creating_strategy.md) — add strategy params to the search space
