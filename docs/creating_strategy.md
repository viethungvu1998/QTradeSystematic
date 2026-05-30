# How to Write a Custom Strategy

## The `BaseStrategy` Contract

Every strategy in QTS must subclass `BaseStrategy` and implement one method:

```python
from abc import abstractmethod
import polars as pl
from qts.research.strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    @abstractmethod
    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """Produce a standard signal frame."""
```

`generate_signals` receives a Polars DataFrame that has already been preprocessed by the feature
pipeline (OHLCV columns plus any indicator columns you configured). It must return a
**SignalFrame**.

---

## The SignalFrame Schema

Every strategy must return exactly these four columns:

| Column | Type | Constraint | Meaning |
|---|---|---|---|
| `date` | `pl.Date` | — | Bar date for this signal |
| `symbol` | `pl.String` | — | Symbol the signal applies to |
| `signal` | `pl.Int32` | `∈ {-1, 0, 1}` | Direction: `1` = long, `-1` = short, `0` = flat |
| `weight` | `pl.Float64` | `∈ [0, 1]` | Fractional allocation weight |

The base class provides two helpers you should use:

```python
# Return an empty but correctly typed frame when there is nothing to signal
return self.empty_signal_frame()

# Validate your output before returning it (raises ValueError on violation)
return self.validate_signal_frame(frame)
```

---

## Step-by-Step: Create a New Strategy

### 1. Create the file

Place your strategy under `qts/research/strategies/`:

```bash
# Both macOS and Windows
touch qts/research/strategies/my_strategy.py
```

For a strategy family with multiple files, create a sub-package instead:

```bash
mkdir qts/research/strategies/my_strategy
touch qts/research/strategies/my_strategy/__init__.py
touch qts/research/strategies/my_strategy/model.py
```

### 2. Implement the strategy

The example below is a simple cross-sectional momentum strategy: rank symbols by 21-day
return-of-change, go long the top quantile, short the bottom quantile.

```python
# qts/research/strategies/my_strategy.py
from __future__ import annotations

import polars as pl

from qts.core.registry import Registry
from qts.research.strategies.base import BaseStrategy


@Registry.register_strategy("my_momentum")
class MyMomentumStrategy(BaseStrategy):
    """Simple cross-sectional momentum: long top-N, short bottom-N by ROC."""

    def __init__(self, long_quantile: float = 0.2, short_quantile: float = 0.2) -> None:
        self.long_quantile = long_quantile
        self.short_quantile = short_quantile

    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        if "roc_21" not in data.columns:
            return self.empty_signal_frame()

        ranked = (
            data.filter(pl.col("roc_21").is_not_null())
            .with_columns(
                pl.col("roc_21")
                .rank("ordinal")
                .over("date")
                .alias("rank"),
                pl.col("roc_21")
                .count()
                .over("date")
                .alias("n"),
            )
            .with_columns(
                (pl.col("rank") / pl.col("n")).alias("pct_rank")
            )
        )

        signals = ranked.select(
            pl.col("date"),
            pl.col("symbol"),
            pl.when(pl.col("pct_rank") >= 1.0 - self.long_quantile)
            .then(pl.lit(1))
            .when(pl.col("pct_rank") <= self.short_quantile)
            .then(pl.lit(-1))
            .otherwise(pl.lit(0))
            .cast(pl.Int32)
            .alias("signal"),
            pl.lit(1.0).cast(pl.Float64).alias("weight"),
        )

        return self.validate_signal_frame(signals)
```

### 3. Register the strategy

The `@Registry.register_strategy("my_momentum")` decorator in step 2 registers the class under
the key `"my_momentum"`. `Config.build()` calls `Registry.get_strategy("my_momentum")(**params)`
when it encounters `strategy.type: my_momentum` in your YAML.

Make sure your module is imported before `Config.build()` runs. The simplest way is to add an
import to `qts/research/strategies/__init__.py`:

```python
# qts/research/strategies/__init__.py
from . import my_strategy  # noqa: F401 — side-effect import registers the strategy
```

### 4. Reference it in YAML

```yaml
strategy:
  type: my_momentum
  params:
    long_quantile: 0.2
    short_quantile: 0.2
```

All keys under `params:` are passed as keyword arguments to `__init__`. An unregistered
`type` raises `RegistryError` at `Config.build()` time.

---

## Existing Strategy Families

### `factor`

A cross-sectional factor strategy that ranks symbols by numeric signals derived from the feature
pipeline and converts ranks to long/short weights. It supports three signal algorithms —
`cross_sectional_rank`, `factor_as_signal`, and `ic_weighted` — and three ML trainer backends
(`xgb_regressor`, `xgb_ranker`, `ic_composite`). You configure the algorithm, quantile
thresholds, and trainer key entirely through `strategy.params`. The family owns
`qts/research/strategies/factor/`.

### `ml_factor`

A classification strategy that trains an ML model (`xgb_classifier`, `xgb_regressor`, `linear`,
or `ic_composite`) to predict the sign of forward returns. At each rebalance, the model is
retrained on the rolling training window and generates signals from predicted probabilities or
predicted values. All model and training parameters live under `strategy.params`. The family
owns `qts/research/strategies/ml_factor/`.

### `stat_arb`

A pair-trading strategy that screens the universe for cointegrated pairs, estimates hedge ratios
via `ols` or `rolling_ols`, and generates entry/exit signals whenever the spread z-score crosses
configurable bands (`zscore_threshold` signal rule). The strategy runs through the same engine
path as all other strategies and returns a standard SignalFrame across all traded pair legs. The
family owns `qts/research/strategies/stat_arb/`.

### `vn100_quantamental`

A walk-forward ML factor strategy purpose-built for the VN100 universe. It combines QSMOM
momentum transforms, technical indicator features, and VN fundamental data (income statement,
balance sheet, ratios), trains an XGBoost regressor on rolling windows, and produces long-only
signals. This is the reference implementation for combining fundamentals with ML in a VN context.
The strategy lives in `qts/research/strategies/vn100_quantamental/`.

---

## Portfolio Construction

After `generate_signals` returns a SignalFrame, the engine optionally applies a portfolio
constructor to compute final allocation weights. You configure this with the
`portfolio_construction:` key in YAML.

| Registry key | Description |
|---|---|
| `equal_weight` | Divide capital equally among all non-zero signals |
| `exponential_weight` | Decay weights exponentially by signal recency or rank |
| `inverse_volatility` | Weight inversely proportional to recent realized volatility |
| `volatility_target` | Scale weights so the portfolio hits a target annualised volatility |
| `mean_variance` | Markowitz mean-variance optimal weights |
| `min_variance` | Minimum-variance portfolio (ignores expected returns) |
| `mean_variance_turnover` | Mean-variance with a turnover penalty |
| `risk_parity` | Equal risk contribution across positions |
| `hrp` | Hierarchical risk parity (tree clustering) |
| `kelly` | Kelly criterion fractional sizing |
| `cost_adjusted` | Adjusts weights to reduce unnecessary turnover given commissions |

Example YAML:

```yaml
portfolio_construction:
  name: hrp
  params: {}
```

```yaml
portfolio_construction:
  name: volatility_target
  params:
    target_vol: 0.15    # 15 % annualised
```

If `portfolio_construction:` is absent, the signal `weight` column from your strategy is used
directly.

Constraint helpers (imported directly, not registry-backed) can be applied in custom strategy
code after calling the constructor:

```python
from qts.research.portfolio_construction.constraints import (
    apply_weight_constraints,
    apply_volatility_cap,
)

weights = apply_weight_constraints(weights, min_weight=0.01, max_weight=0.20)
weights = apply_volatility_cap(weights, returns, cap=0.25)
```

---

## See Also

- [introduction.md](introduction.md) — full strategy and portfolio constructor registry
- [run_backtest.md](run_backtest.md) — wire your strategy into a YAML config and run it
- [sweep.md](sweep.md) — tune strategy parameters with Optuna
