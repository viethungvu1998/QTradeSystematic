# VN100 Quantamental ML Strategy

## Algorithm Contract

**Type**: ML factor (XGBoost regressor → long-only portfolio)
**Universe**: VN100 Vietnam equities (`configs/assets/vn100.yml`)
**Asset type**: `vn_stock`

## Invariants

1. The portfolio construction is **long-only** — `num_short_positions` MUST remain 0 unless the user explicitly enables shorting.
2. The target column MUST NOT be named `forward_return_*`, `future_*`, or `target_*` — use `fwd_ret_N` style naming when adding custom targets.
3. The `portfolio.name` MUST NOT be changed to `equal_weight` for this strategy; the weighted-score portfolio is load-bearing for alpha generation.
4. `min_trading_days >= 126` — do not drop this below 126 bars or the liquidity screen becomes meaningless.
5. Walk-forward training uses the **previous** window only — no future data must leak into `train_data`.

## Strategy Description

**Data pipeline**:
- Fetch OHLCV for VN100 symbols from VnStock
- Fetch quarterly fundamentals (TTM financials via DNSE)
- Compute QSMOM momentum signal (fast/slow EMA crossover + trailing returns)
- Compute technical features: RSI, MACD, ATR, HistVol, ROC
- Compute fundamental factor scores (Value, Quality, Growth, Momentum)
- Z-score cross-sectionally per date

**Model**:
- XGBoost regressor trained on cross-sectional z-scored features
- Target: forward N-day return z-scored within date
- Walk-forward: fit on T-window, predict on last bar of window

**Portfolio construction**:
- Rank symbols by predicted score
- Long top K by score (score > `long_threshold`)
- Equal-weight within the long book

## Baseline Configuration

| Parameter | Default | Allowed range |
|-----------|---------|---------------|
| `train_window` | 504 | 126–1008 |
| `rebalance_period` | 21 (monthly) | integer > 5; sweep range 6–63 |
| `num_long_positions` | 10 | 3–20 |
| `qsmom_fast` | 21 | 5–63 |
| `qsmom_slow` | 252 | 63–504 |
| `forward_period` | 21 | 5–63 |
| `model_max_depth` | 3 | 2–6 |
| `model_n_estimators` | 80 | 20–200 |

## Success Criteria

- Sharpe ratio ≥ 0.8 (annualised, 252 trading days)
- Max drawdown ≤ 25%
- CAGR > 0%
