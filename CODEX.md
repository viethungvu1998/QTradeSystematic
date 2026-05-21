# QTradeSystematic — Codex Context

This file is the short contributor-facing snapshot of the repository's actual current state.

For fuller detail, use:

- `README.md`
- `ARCHITECTURE.md`
- `WORKFLOWS.md`
- `FEATURES.md`
- `AGENTS.md`

## Current Reality

- No CLI entry point exists yet.
- `qts_flow` is async and accepts a config path string.
- `validation` is a config mode, not an automated promotion gate.
- `crypto_futures` is wired through both top-level orchestration flows.
- `Config.build()` resolves registry keys, but several adapters still need explicit real-client wiring for live use.

## Runtime Surface

### Main entry points

- `qts.orchestration.flow.qts_flow`
- `qts.orchestration.flows.data_fetch_flow`
- `qts.orchestration.serve`

<!-- AUTO-GENERATED from Registry decorators in qts/ -->

### Registered data source keys

- `fmp` — US stock OHLCV + fundamentals (FMP API)
- `yahoo` — US stock OHLCV (Yahoo Finance)
- `binance` — Spot crypto OHLCV
- `binance_futures` — Perpetual futures OHLCV + funding rates + open interest
- `dnse` — VN equity, warrants, and rolling VN30 futures (DNSE API)
- `vnstock` — VN equity and warrants (vnstock library)
- `vnstock_futures` — VN futures (vnstock library)

### Registered storage keys

- `duckdb` — Persistent DuckDB at `~/.qts/database/qts.duckdb`
- `parquet` — Parquet cache at `~/.qts/cache/`

### Registered engine keys

- `vectorbt` / `fast` — `VectorBTProEngine` (Polars/DuckDB, vectorbtpro)
- `zipline` / `normal` — `ZiplineReloadedEngine` (Zipline bundle, calendar-aware)

### Registered strategy keys

- `factor` — Cross-sectional factor model with ranking and IC-weighted signals
- `ml_factor` — ML-driven factor strategy (XGBoost, linear, IC composite)
- `stat_arb` — Pairs / statistical arbitrage with OLS/rolling-OLS spread models
- `vn100_quantamental` — VN100 quantamental strategy with KBS fundamental data

### Registered feature keys

- `technical` — Full OHLCV technical feature set
- `fundamental` — FMP-compatible fundamental ratios and factor scores
- `vn_fundamental` — KBS VN fundamental data mapped to FMP-like schema
- `onchain` — On-chain metrics (crypto)
- `forward_returns` — Forward return labels for supervised learning
- Indicators: `rsi`, `roc`, `macd`, `adx`, `atr`, `bollinger`, `hist_vol`, `obv`, `volume_ratio`, `zscore`

### Registered transform keys

- `qsmom` — QS momentum transform
- `price_preprocessor` — Price quality preprocessing
- `universe_screener` — Universe screening/filtering

### Registered signal algorithm keys

- `cross_sectional_rank` — Rank-based long/short signal generation
- `factor_as_signal` — Direct factor value as signal
- `ic_weighted` — IC-weighted composite signal

### Registered spread model keys (stat_arb)

- `ols` — OLS static spread
- `rolling_ols` — Rolling OLS spread

### Registered ML model keys

- `xgb_classifier` — XGBoost classifier
- `xgb_regressor` — XGBoost regressor
- `linear` — Linear regression
- `ic_composite` — IC-weighted composite model

### Registered broker keys

- `moomoo` — Moomoo/Futu OpenD broker
- `binance` — Binance spot/futures broker
- `dnse` — DNSE VN market broker

### Registered calendar keys

- `nyse` — NYSE trading calendar
- `hose` — Ho Chi Minh Stock Exchange
- `hkex` — Hong Kong Stock Exchange
- `crypto` — 24/7 crypto calendar

### Registered commission model keys

- `percentage` — Percentage of notional
- `per_trade` — Flat fee per trade

### Registered fill model keys

- `immediate` — Fill at current bar close
- `next_open` — Fill at next bar open
- `vwap` — VWAP fill approximation

### Registered slippage model keys

- `fixed` — Fixed basis-point slippage
- `volatility_scaled` — Volatility-adjusted slippage

<!-- END AUTO-GENERATED -->

## Notes for Contributors

- Update docs when changing runtime behavior, config keys, orchestration, or registry names.
- Prefer describing implemented behavior, not intended roadmap behavior.
- When in doubt, use the tests and the code paths in `qts/orchestration/flow.py` and `qts/config/builder.py` as the source of truth.
