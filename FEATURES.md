# FEATURES — QTradeSystematic

38 independently testable features across 6 phases.
Each has a single scope, explicit dependencies, and a concrete verify step.

**Test tiers:** **U** = Unit · **I** = Integration · **P** = Paper (`@pytest.mark.paper`)

---

## Phase 1 — Core

### F-01 · Instrument model
**Builds:** `core/instrument.py` — `AssetType(Stock | Crypto)` enum; `Instrument(symbol, asset_type, exchange, currency)`  
**Depends on:** —  
**Verify (U):** Construct one Stock and one Crypto instrument; assert each field; assert `AssetType` has exactly two members.

### F-02 · Order and Fill models
**Builds:** `core/order.py` — `OrderSide`, `OrderType`, `OrderStatus` enums; `Order`, `Fill` dataclasses; all monetary fields `Decimal`  
**Depends on:** F-01  
**Verify (U):** Construct every `OrderType`; assert `Fill.price` and `Fill.quantity` are `Decimal`; assert no `float` fields.

### F-03 · Portfolio models
**Builds:** `core/portfolio.py` — `Position`, `Portfolio` dataclasses; all monetary values `Decimal`  
**Depends on:** F-01, F-02  
**Verify (U):** Compute portfolio total value from known positions; assert result is `Decimal`; assert no `float` in any field.

### F-04 · Plugin Registry
**Builds:** `core/registry.py` — `@Registry.register_*` decorators; `Registry.get_*()` resolution for all extension points  
**Depends on:** —  
**Verify (U):** Register a stub under each extension point; resolve each by name; assert correct class returned. Assert an unregistered name raises `RegistryError` with the name in the message.

### F-05 · Async Event Bus
**Builds:** `core/events.py` — typed async event bus for ticks and order updates  
**Depends on:** —  
**Verify (U):** Emit a typed event; assert subscribed handler called with correct payload; assert unsubscribed handler not called.

---

## Phase 2 — Data

### F-06 · Data ABCs
**Builds:** `data/base.py` (`BaseDataSource`), `data/storage/base.py` (`BaseStorage`), `data/bundles/base.py` (`BaseBundleAdapter`) — full signatures per CODING_RULES.md  
**Depends on:** F-01  
**Verify (U):** Subclass each ABC with no implementations; assert every abstract method raises `NotImplementedError`.

### F-07 · Parquet storage
**Builds:** `data/storage/parquet.py` — `ParquetStorage`: `write / read / exists / list_keys`  
**Depends on:** F-06  
**Verify (I):** Write a Polars OHLCV DataFrame to a temp dir; read it back; assert schema `[date, symbol, open, high, low, close, volume]`.

### F-08 · DuckDB storage
**Builds:** `data/storage/duckdb.py` — `DuckDBStorage`: `write / read / append / query`  
**Depends on:** F-06  
**Verify (I):** Write OHLCV rows to an in-memory DuckDB instance; query back; assert schema; assert `append` does not produce duplicate rows.

### F-09 · FMP data source
**Builds:** `data/sources/fmp.py` — `FMPDataSource`: `get_ohlcv`, `get_fundamentals`  
**Depends on:** F-06, F-07  
**Verify (I):** Fetch OHLCV from a recorded FMP fixture; assert Polars output schema; assert `DataSourceError` raised on an invalid symbol.

### F-10 · Yahoo data source
**Builds:** `data/sources/yahoo.py` — `YahooDataSource`: `get_ohlcv`  
**Depends on:** F-06, F-07  
**Verify (I):** Fetch OHLCV for a stock symbol; assert output schema `[date, symbol, open, high, low, close, volume]`.

### F-11 · Binance data source
**Builds:** `data/sources/binance.py` — `BinanceDataSource`: `get_ohlcv`, `stream_ticks` backed by `binance-connector`  
**Depends on:** F-06, F-07  
**Credentials:** reads `BINANCE_DEMO_TRADING_API_KEY` + `BINANCE_DEMO_TRADING_SECRET_KEY` from env for testnet; `BINANCE_TRADING_KEY` + `BINANCE_TRADING_SECRET_KEY` for production  
**Verify (I):** Fetch OHLCV for `BTC/USDT` from Binance testnet using demo credentials; assert output schema `[date, symbol, open, high, low, close, volume]`. Assert `stream_ticks` yields `Tick` objects with `Decimal` price and volume.

### F-12 · DataManager — stock routing
**Builds:** `data/manager.py` — routes stock symbols → FMP/Yahoo by config; writes DuckDB; caches Parquet  
**Depends on:** F-07, F-08, F-09, F-10  
**Verify (I):** Call `get_ohlcv(["AAPL"])` with an in-memory DuckDB; assert rows written and schema correct; assert second call is served from cache without a network call.

### F-13 · Zipline bundle ingest
**Builds:** `data/bundles/local.py` (`LocalBundleAdapter`), `data/bundles/zipline_ingest.py` — ingest DuckDB stock tables → Zipline bundle on disk  
**Depends on:** F-08, F-12  
**Verify (I):** Ingest from in-memory DuckDB; load the resulting Zipline bundle; assert trading sessions match the input date range.

### F-14 · DataManager — crypto routing
**Builds:** `data/manager.py` (extended) — routes crypto symbols → Binance; skips Zipline ingest for crypto  
**Depends on:** F-11, F-12  
**Verify (I):** Call `get_ohlcv(["BTC/USDT"])`; assert DuckDB rows written; assert no Zipline bundle produced.

---

## Phase 3 — Research

### F-15 · BaseFeature ABC
**Builds:** `research/features/base.py` — `BaseFeature`: `fit_transform(df: pl.DataFrame) → pl.DataFrame`; contract: append columns, never drop  
**Depends on:** F-04  
**Verify (U):** `MockFeature` appends one column; assert every original column is present in output.

### F-16 · Technical features
**Builds:** `research/features/technical.py` — RSI, MACD, ATR, Bollinger; works for any OHLCV asset type  
**Depends on:** F-15  
**Verify (I):** Run on a 200-row OHLCV fixture; assert each output column present; assert no NaN values beyond each indicator's warmup window.

### F-17 · Fundamental features
**Builds:** `research/features/fundamentals.py` — P/E, EV/EBITDA; stocks only, no-op for crypto  
**Depends on:** F-15  
**Verify (I):** Run on stock OHLCV + fundamentals fixture; assert feature columns added. Run on a crypto OHLCV fixture; assert zero columns changed (no-op).

### F-18 · Onchain features
**Builds:** `research/features/onchain.py` — NVT, active addresses; crypto only, no-op for stocks  
**Depends on:** F-15  
**Verify (I):** Run on crypto fixture; assert onchain columns added. Run on stock fixture; assert zero columns changed (no-op).

### F-19 · Forward returns
**Builds:** `research/features/forward_returns.py` — target variable construction for configurable periods  
**Depends on:** F-15  
**Verify (I):** Compute on a known price series; assert 1-day forward return matches manually calculated value; assert the last `n` rows are NaN for each period `n` (no look-ahead).

### F-20 · BaseStrategy ABC + SignalFrame validation
**Builds:** `research/strategies/base.py` — `BaseStrategy`; validator enforces `signal ∈ {-1, 0, 1}` and `weight ∈ [0, 1]`  
**Depends on:** F-04, F-15  
**Verify (U):** `MockStrategy` returning a valid frame passes validation. `MockStrategy` returning `signal=2` or `weight=1.5` raises `ValueError`.

### F-21 · Factor strategy (XGBoost)
**Builds:** `research/strategies/factor/` — ML factor model; `generate_signals(df) → SignalFrame`  
**Depends on:** F-16, F-17, F-19, F-20  
**Verify (I):** Train and predict on a 1-year OHLCV fixture; assert `SignalFrame` schema; assert signals are in `{-1, 0, 1}`; assert at least one non-zero signal.

### F-22 · Stat-arb strategy
**Builds:** `research/strategies/stat_arb/` — pairs / cointegration strategy  
**Depends on:** F-16, F-20  
**Verify (I):** Run on a 2-symbol cointegrated fixture; assert `SignalFrame` schema; assert a spread crossing generates opposing signals on the two symbols.

### F-23 · Simulation models
**Builds:** `research/backtest/simulation/fills.py`, `slippage.py`, `commission.py`, `calendar.py`  
**Depends on:** F-02  
**Verify (U):**
- `NextOpenFill` fills at the next bar's open price
- `VolatilityScaledSlippage` produces larger slippage on a high-ATR bar than a low-ATR bar
- `PercentageCommission(rate=0.001)` charges the correct amount on a known trade
- `NYSE` calendar excludes a known public holiday; `Crypto` calendar includes weekends

### F-24 · Metrics
**Builds:** `research/backtest/metrics.py` — Sharpe, Sortino, CAGR, max drawdown, win rate  
**Depends on:** —  
**Verify (U):** Compute each metric on a synthetic returns series with a known answer; assert result matches within numerical tolerance.

### F-25 · BacktestConfig + BacktestResult
**Builds:** `research/backtest/base.py` — `BaseEngine` ABC; `BacktestConfig` (typed dataclass mirroring the YAML schema); `BacktestResult` (frozen dataclass)  
**Depends on:** F-04, F-23, F-24  
**Verify (U):** Construct `BacktestConfig` from each workflow fixture; assert every field is the correct Python type. Assert `BacktestResult` schema is identical regardless of which engine populates it.

### F-26 · VectorBTProEngine
**Builds:** `research/backtest/engines/vectorbtpro_engine.py` — reads Polars DataFrame from DuckDB; full-history vectorised batch  
**Depends on:** F-08, F-20, F-25  
**Verify (I):** Run on a 1-year OHLCV fixture with `MockStrategy`; assert `BacktestResult` produced; assert Sharpe, CAGR, and max drawdown are finite numeric values.

### F-27 · ZiplineEngine
**Builds:** `research/backtest/engines/zipline_engine.py` — reads Zipline bundle; bar-by-bar, strict look-ahead guard  
**Depends on:** F-13, F-20, F-25, F-26  
**Verify (I):** Run `VectorBTProEngine` and `ZiplineEngine` on the same fixture and `MockStrategy`; assert `BacktestResult` schema is identical; assert Sharpe values are within 30% of each other.

---

## Phase 4 — Config

### F-28 · YAML loader + BacktestConfig schema
**Builds:** `config/loader.py` — parse YAML into a typed `BacktestConfig`; validate required keys per `workflow` type  
**Depends on:** —

Full key schema:

| Key | Type | Required for |
|---|---|---|
| `workflow` | `research \| validation \| live` | all |
| `asset_types` | `list[stock \| crypto]` | all |
| `universe.stock`, `universe.crypto` | `list[str]` | all |
| `start_date`, `end_date` | `date` | research, validation |
| `initial_capital` | `Decimal` | research, validation |
| `data_sources.stock`, `data_sources.crypto` | registry key | all |
| `storage` | registry key | all |
| `features.technical`, `.fundamental`, `.onchain` | `bool` | all |
| `features.forward_returns.periods` | `list[int]` | all |
| `strategy.type` | registry key | all |
| `strategy.params` | `dict` | all |
| `backtest_engine` | `fast \| normal` | all |
| `fill_model` | registry key | validation, live |
| `slippage_model` | registry key | validation, live |
| `commission.model`, `commission.rate` | registry key, `Decimal` | validation, live |
| `calendar` | registry key | validation, live |
| `brokers.stock`, `brokers.crypto` | registry key | live |
| `schedule.stock`, `schedule.crypto` | cron string | live |
| `promotion_gate.max_sharpe_degradation` | `float` | validation |

**Verify (U):**
- Load `research.yaml`, `validation.yaml`, `live.yaml` fixtures; assert every field is the correct Python type.
- `live.yaml` with `brokers` key absent raises `ConfigError` naming the missing key.
- YAML with any unrecognised top-level key raises `ConfigError` naming the key.

### F-29 · Registry resolution
**Builds:** `config/builder.py` — `Config.build()` resolves every string registry key in `BacktestConfig` to a concrete instance  
**Depends on:** F-04, F-28  
**Verify (U):** Register mock plugins for all extension points; call `Config.build()` with a fully-populated config; assert each resolved field holds the correct concrete instance. Assert an unregistered key raises `RegistryError` with the key name in the message.

---

## Phase 5 — Execution

### F-30 · BaseBroker ABC
**Builds:** `execution/base.py` — `BaseBroker` with the full async contract from CODING_RULES.md  
**Depends on:** F-02, F-03, F-05  
**Verify (U):** `MockBroker(BaseBroker)` returns canned positions and fills; assert `place_order` returns a `Fill` with `Decimal` price and quantity.

### F-31 · MoomooBroker
**Builds:** `execution/brokers/moomoo.py` — Futu OpenD gateway via `futu-api`  
**Depends on:** F-30  
**Verify (U):** Mock Futu API responses; assert `place_order` constructs the correct Futu order object and maps the response to a `Fill`.  
**Verify (P):** Connect to Futu OpenD in paper mode; `place_order(AAPL, qty=1, LIMIT)`; assert `Fill` returned with correct symbol and `Decimal` fields.

### F-32 · BinanceBroker
**Builds:** `execution/brokers/binance.py` — Binance REST + user data stream via `binance-connector`  
**Depends on:** F-30  
**Credentials:** `binance_mode: demo` reads `BINANCE_DEMO_TRADING_API_KEY` + `BINANCE_DEMO_TRADING_SECRET_KEY` and uses `https://testnet.binance.vision`; `binance_mode: live` reads `BINANCE_TRADING_KEY` + `BINANCE_TRADING_SECRET_KEY` and uses the production endpoint  
**Verify (U):** Mock `binance-connector` responses; assert `place_order` constructs the correct request and maps the response to a `Fill` with `Decimal` fields.  
**Verify (P):** Set `binance_mode: demo`; connect to testnet; `place_order(BTC/USDT, qty=0.001, LIMIT)`; assert `Fill` returned with correct symbol and `Decimal` fields. Mark `@pytest.mark.paper`.

### F-33 · OrderRouter
**Builds:** `execution/router.py` — `OrderRouter`: `dict[AssetType, BaseBroker]`; dispatches orders concurrently by `AssetType`  
**Depends on:** F-01, F-02, F-30  
**Verify (U):** Inject `MockBroker` for Stock and Crypto; submit one stock order and one crypto order; assert each reaches only its designated broker.

### F-34 · PositionSync
**Builds:** `execution/sync.py` — `PositionSync`: target weights → delta `Order` list  
**Depends on:** F-03, F-33  
**Verify (U):** Given known current positions and target weights: assert correct buy/sell orders generated; assert no order emitted for symbols whose weight is unchanged; assert all monetary values are `Decimal`.

---

## Phase 6 — Orchestration

### F-35 · Task library
**Builds:** `orchestration/tasks/data_tasks.py`, `research_tasks.py`, `execution_tasks.py` — `download_ohlcv`, `download_fundamentals`, `build_features`, `run_backtest`, `sync_positions`, `execute_rebalance`  
**Depends on:** F-12, F-14, F-21, F-26, F-34  
**Rule:** Every task accepts a `BacktestConfig` — no literal asset types, broker names, or engine names inside task bodies.  
**Verify (U):** Each task tested in isolation with mocked domain dependencies and a `BacktestConfig` fixture; assert the correct domain function is called with values drawn from config; assert the task retries on `DataSourceError` and `BrokerError`.

### F-36 · `qts_flow` — research
**Builds:** `orchestration/flow.py` (`qts_flow`) reading a `research.yaml`; data → features → `VectorBTProEngine` → `BacktestResult`  
**Depends on:** F-29, F-35  
**Rule:** `flow.py` contains zero hardcoded values — all routing comes from `BacktestConfig`.  
**Verify (I):**
- Run with `asset_types: [stock]`; assert `BacktestResult` produced with all metrics; assert Zipline bundle written.
- Run with `asset_types: [crypto]`; assert `BacktestResult` produced; assert no Zipline bundle written.

### F-37 · `qts_flow` — validation
**Builds:** `orchestration/flow.py` reading a `validation.yaml`; Zipline bundle path + promotion gate  
**Depends on:** F-27, F-36  
**Verify (I):** Run `qts_flow` with `research.yaml` then `validation.yaml` on the same fixture; assert `ZiplineEngine` used when `backtest_engine: normal`; assert Sharpe degradation is compared against `promotion_gate.max_sharpe_degradation` and a rejection is raised when the threshold is exceeded.

### F-38 · `qts_flow` — live (paper)
**Builds:** `orchestration/flow.py` reading a `live.yaml`; `sync_positions` → `execute_rebalance` → paper broker; schedule registered from `config.schedule`  
**Depends on:** F-31, F-32, F-34, F-35, F-36  
**Rule:** Cron schedule is registered with Prefect from `config.schedule` at deploy time — not hardcoded in `flow.py`.  
**Verify (P):** Run `qts_flow` with `live.yaml` pointed at paper accounts (Futu paper + Binance testnet); assert `PositionSync` → `OrderRouter` → paper fill cycle completes; assert DuckDB positions table updated with fills. `@pytest.mark.paper`.

---

## Summary

| Phase | Features | Tiers |
|---|---|---|
| 1 — Core | F-01 → F-05 | U |
| 2 — Data | F-06 → F-14 | U, I |
| 3 — Research | F-15 → F-27 | U, I |
| 4 — Config | F-28 → F-29 | U |
| 5 — Execution | F-30 → F-34 | U, P |
| 6 — Orchestration | F-35 → F-38 | U, I, P |

**38 features total.** Each is independently buildable. No feature may be marked done until its verify step passes.
