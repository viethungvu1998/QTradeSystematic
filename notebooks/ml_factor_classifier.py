from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import duckdb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402, I001
import numpy as np
import pandas as pd
import polars as pl
import pyfolio as pf
import plotly.graph_objects as go
import xgboost as xgb
import yaml

_PLOTLY_TEMPLATE = go.layout.Template


class _PlotlyTemplateSkipInvalid(_PLOTLY_TEMPLATE):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("skip_invalid", True)
        super().__init__(*args, **kwargs)


go.layout.Template = _PlotlyTemplateSkipInvalid
try:
    import vectorbtpro as vbt
finally:
    go.layout.Template = _PLOTLY_TEMPLATE


OHLCV_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]


def find_repo_root() -> Path:
    repo_root = Path.cwd().resolve()
    if not (repo_root / "qts").exists():
        repo_root = repo_root.parent
    if not (repo_root / "qts").exists():
        raise RuntimeError("Run this script from the repository root or notebooks/ directory.")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def parse_args(repo_root: Path) -> argparse.Namespace:
    default_config = (
        repo_root / "configs" / "strategies" / "ml_factor" / "vn100_ml_classification2.yaml"
    )
    default_data_csv = repo_root / "data" /  "VNSTOCK.csv"
    default_database = Path.home() / ".qts" / "database" / "qts.duckdb"

    parser = argparse.ArgumentParser(
        description="Run the VN100 ML factor classifier notebook workflow end to end."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Strategy YAML with run/features/classification/signal/xgb params.",
    )
    parser.add_argument("--data-csv", type=Path, default=default_data_csv)
    parser.add_argument("--database", type=Path, default=default_database)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder. Defaults to output/ml_factor_classifier/<run_id>_<timestamp>.",
    )
    parser.add_argument(
        "--force-refresh-data",
        action="store_true",
        help="Rebuild the cached CSV from DuckDB before running.",
    )
    parser.add_argument(
        "--skip-tearsheet",
        action="store_true",
        help="Skip pyfolio tear sheet generation.",
    )
    return parser.parse_args()


def load_config(config_path: Path, repo_root: Path) -> None:
    global BENCHMARK
    global CLASS_PROB_COLS
    global CLASS_QUANTILES
    global CLASS_SCORES
    global CLASSIFICATION_THRESHOLD_SCOPE
    global CLASSIFICATION_WINDOW
    global CONFIG_EFFECTIVE
    global CONFIG_PATH
    global END_DATE
    global FALLBACK_MIN_POSITIONS
    global FEE_RATE
    global FORWARD_PERIOD
    global INIT_CASH
    global MAX_POSITIONS
    global MIN_FEATURE_COUNT
    global MIN_PREDICTED_CLASS
    global BALANCED_SAMPLE_WEIGHT
    global PREDICTOR_COLS
    global RANK_COL
    global REBALANCE_EXISTING_POSITIONS
    global REBALANCE_FREQUENCY
    global RUN_ID
    global SIGNAL_PROB_COLS
    global SIGNAL_PROB_SUM_COL
    global SIGNAL_THRESHOLD
    global START_DATE
    global STOP_LOSS
    global STOP_LOSS_BENCHMARK_FILTER
    global STOP_LOSS_MIN_HOLD
    global TARGET_COL
    global TAKE_PROFIT
    global TEST_INTERVALS
    global TRAIN_WINDOW
    global TRAILING_STOP
    global XGB_PARAMS
    global BENCHMARK_REGIME_ENABLED
    global BENCHMARK_REGIME_MIN_PERIODS
    global BENCHMARK_REGIME_RULE
    global BENCHMARK_REGIME_WINDOW

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else repo_root / path

    def read_yaml(path: Path) -> dict[str, Any]:
        path = resolve(path)
        if not path.exists():
            raise FileNotFoundError(f"YAML config not found: {path}")
        data = yaml.safe_load(path.read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a YAML mapping.")
        return data

    def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        out = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = merge(out[key], value)
            else:
                out[key] = value
        return out

    script_defaults = {
        "start_date": "2018-01-01",
        "end_date": "2026-05-31",
        "benchmark": "VN:VNINDEX",
        "initial_capital": 100_000_000,
        "commission": {"rate": 0.0016},
        "run": {"min_feature_count": 3},
        "validation": {
            "test_intervals": [{"start_date": "2021-01-01", "end_date": "2026-05-30"}],
        },
    }

    config_path = resolve(config_path)
    payload = read_yaml(config_path)
    if payload.get("default_params_path"):
        payload = merge(read_yaml(Path(payload["default_params_path"])), payload)
    payload = merge(script_defaults, payload)

    run = payload.get("run", {})
    features = payload.get("features", {})
    classification = payload.get("classification", {})
    signal = payload.get("signal", {})
    commission = payload.get("commission", {})
    exit_rules = payload.get("backtest", {}).get("exit_rules", {})
    backtest = payload.get("backtest", {})
    benchmark_regime = payload.get("benchmark_regime", {})
    portfolio_params = payload.get("portfolio_construction", {}).get("params", {})
    validation = payload.get("validation", {})

    CONFIG_EFFECTIVE = payload
    CONFIG_PATH = config_path
    RUN_ID = str(payload["run_id"])

    START_DATE = pd.Timestamp(payload["start_date"]).date()
    END_DATE = pd.Timestamp(payload["end_date"]).date()
    BENCHMARK = str(payload["benchmark"])
    INIT_CASH = float(payload["initial_capital"])
    FEE_RATE = float(commission["rate"])

    FORWARD_PERIOD = int(run["forward_period"])
    CLASSIFICATION_WINDOW = int(run["classification_window"])
    TRAIN_WINDOW = int(run["train_window"])
    REBALANCE_FREQUENCY = int(run["rebalance_frequency"])
    REBALANCE_EXISTING_POSITIONS = bool(backtest.get("rebalance_existing_positions", False))
    MAX_POSITIONS = int(run["max_positions"])
    MIN_FEATURE_COUNT = int(run["min_feature_count"])
    portfolio_positions = portfolio_params.get("num_long_positions")
    if portfolio_positions is not None:
        if isinstance(portfolio_positions, list):
            if len(portfolio_positions) != 1:
                raise ValueError(
                    "portfolio_construction.params.num_long_positions has multiple values; "
                    "use a strategy-specific run.max_positions for this single-run script."
                )
            portfolio_positions = portfolio_positions[0]
        MAX_POSITIONS = int(portfolio_positions)

    if min(FORWARD_PERIOD, CLASSIFICATION_WINDOW, TRAIN_WINDOW, MAX_POSITIONS) <= 0:
        raise ValueError("run periods and max_positions must be positive.")

    predictor_cols = features.get("predictor_cols") or features["baseline_columns"]
    PREDICTOR_COLS = [str(col) for col in predictor_cols]

    CLASS_QUANTILES = [float(value) for value in classification["class_quantiles"]]
    CLASS_SCORES = np.asarray(
        classification["class_scores"],
        dtype=float,
    )
    if len(CLASS_QUANTILES) != len(CLASS_SCORES) - 1:
        raise ValueError("classification.class_quantiles must have len(class_scores) - 1 entries.")
    CLASS_PROB_COLS = [f"class_{idx}_prob" for idx in range(len(CLASS_SCORES))]
    CLASSIFICATION_THRESHOLD_SCOPE = str(
        classification.get("threshold_scope", "classification_window")
    )
    if CLASSIFICATION_THRESHOLD_SCOPE not in {"classification_window", "train"}:
        raise ValueError(
            "classification.threshold_scope must be 'classification_window' or 'train'."
        )
    BALANCED_SAMPLE_WEIGHT = bool(classification.get("balanced_sample_weight", False))

    SIGNAL_PROB_COLS = [str(value) for value in signal["probability_cols"]]
    SIGNAL_PROB_SUM_COL = str(signal["probability_sum_col"])
    threshold = signal.get("min_probability_threshold")
    if threshold is None:
        threshold = signal["probability_threshold"]
    SIGNAL_THRESHOLD = None if threshold is None else float(threshold)
    FALLBACK_MIN_POSITIONS = int(signal["fallback_min_positions"])
    min_predicted_class = signal.get("min_predicted_class")
    MIN_PREDICTED_CLASS = None if min_predicted_class is None else int(min_predicted_class)
    if MIN_PREDICTED_CLASS is not None and not 0 <= MIN_PREDICTED_CLASS < len(CLASS_SCORES):
        raise ValueError("signal.min_predicted_class must be a valid class index.")
    RANK_COL = str(signal["rank_col"])
    STOP_LOSS = None if exit_rules.get("stop_loss") is None else float(exit_rules["stop_loss"])
    STOP_LOSS_MIN_HOLD = int(exit_rules.get("stop_loss_min_hold", 1))
    if STOP_LOSS_MIN_HOLD < 1:
        raise ValueError("backtest.exit_rules.stop_loss_min_hold must be >= 1.")
    stop_loss_benchmark_filter = exit_rules.get("stop_loss_benchmark_filter")
    STOP_LOSS_BENCHMARK_FILTER = (
        None if stop_loss_benchmark_filter is None else str(stop_loss_benchmark_filter)
    )
    if STOP_LOSS_BENCHMARK_FILTER not in {None, "below_sma"}:
        raise ValueError("backtest.exit_rules.stop_loss_benchmark_filter must be 'below_sma'.")
    TAKE_PROFIT = None if exit_rules.get("take_profit") is None else float(
        exit_rules["take_profit"]
    )
    TRAILING_STOP = None if exit_rules.get("trailing_stop") is None else float(
        exit_rules["trailing_stop"]
    )
    BENCHMARK_REGIME_ENABLED = bool(benchmark_regime.get("enabled", False))
    BENCHMARK_REGIME_RULE = str(benchmark_regime.get("rule", "above_sma"))
    BENCHMARK_REGIME_WINDOW = int(benchmark_regime.get("window", 42))
    BENCHMARK_REGIME_MIN_PERIODS = int(benchmark_regime.get("min_periods", 14))
    if BENCHMARK_REGIME_RULE != "above_sma":
        raise ValueError("benchmark_regime.rule must be 'above_sma'.")

    XGB_PARAMS = dict(payload["xgb"])

    TEST_INTERVALS = []
    for item in validation["test_intervals"]:
        if isinstance(item, dict):
            start = item.get("start_date") or item.get("start")
            end = item.get("end_date") or item.get("end")
        else:
            start, end = item
        TEST_INTERVALS.append((pd.Timestamp(start), pd.Timestamp(end)))
    TARGET_COL = f"Return_fwd_{FORWARD_PERIOD}"


def write_prediction_diagnostics(output_dir: Path, predictions: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    diagnostics = {
        "prediction_rows": len(predictions),
        "prediction_dates": predictions["date"].nunique(),
        "signal_threshold": SIGNAL_THRESHOLD,
        "min_predicted_class": MIN_PREDICTED_CLASS,
        "classification_threshold_scope": CLASSIFICATION_THRESHOLD_SCOPE,
        "balanced_sample_weight": BALANCED_SAMPLE_WEIGHT,
        "max_signal_probability": predictions[SIGNAL_PROB_SUM_COL].max(),
        "rows_above_signal_threshold": int(
            (predictions[SIGNAL_PROB_SUM_COL] > SIGNAL_THRESHOLD).sum()
        ),
    }
    if MIN_PREDICTED_CLASS is not None:
        predicted_class_filter = predictions["predicted_class"] >= MIN_PREDICTED_CLASS
        diagnostics["rows_at_min_predicted_class"] = int(predicted_class_filter.sum())
        diagnostics["rows_passing_entry_filters"] = int(
            (
                predicted_class_filter
                & (predictions[SIGNAL_PROB_SUM_COL] > SIGNAL_THRESHOLD)
            ).sum()
        )
    for class_idx in range(len(CLASS_SCORES)):
        diagnostics[f"predicted_class_{class_idx}_rows"] = int(
            (predictions["predicted_class"] == class_idx).sum()
        )

    pd.Series(diagnostics, name="prediction_diagnostics").to_csv(
        output_dir / "prediction_diagnostics.csv"
    )


def default_output_dir(repo_root: Path) -> Path:
    safe_run_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in RUN_ID)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    return repo_root / "output" / "ml_factor_classifier" / f"{safe_run_id}_{stamp}"


def load_symbols(repo_root: Path) -> list[str]:
    vn100_path = repo_root / "configs" / "assets" / "vn100.yml"
    vn100 = yaml.safe_load(vn100_path.read_text())
    return list(vn100["symbols"])


def query_prices(database_path: Path, symbols: list[str]) -> pl.DataFrame:
    if not database_path.exists():
        raise FileNotFoundError(f"DuckDB database not found: {database_path}")

    placeholders = ", ".join("?" for _ in symbols)
    con = duckdb.connect(str(database_path), read_only=True)
    try:
        return con.execute(
            f"""
            select date, symbol, open, high, low, close, volume
            from vn_stock_prices
            where symbol in ({placeholders}) and date between ? and ?
            order by symbol, date
            """,
            [*symbols, START_DATE, END_DATE],
        ).pl()
    finally:
        con.close()


def load_or_build_raw(
    repo_root: Path,
    data_csv_path: Path,
    database_path: Path,
    *,
    force_refresh: bool,
) -> pl.DataFrame:
    if force_refresh or not data_csv_path.exists():
        symbols = load_symbols(repo_root)
        raw = query_prices(database_path, [*symbols, BENCHMARK])
        if raw.is_empty():
            raise ValueError(f"No rows returned from {database_path}.")
        data_csv_path.parent.mkdir(parents=True, exist_ok=True)
        raw.select(OHLCV_COLUMNS).write_csv(data_csv_path)
        return raw.select(OHLCV_COLUMNS)

    raw = pl.read_csv(data_csv_path, try_parse_dates=True)
    raw = raw.select([col for col in OHLCV_COLUMNS if col in raw.columns])
    missing_cols = [col for col in OHLCV_COLUMNS if col not in raw.columns]
    if missing_cols:
        raise ValueError(f"{data_csv_path} is missing columns: {missing_cols}")

    if raw.filter(pl.col("symbol") == BENCHMARK).is_empty():
        benchmark_raw = query_prices(database_path, [BENCHMARK])
        if benchmark_raw.is_empty():
            raise ValueError(f"{BENCHMARK} is not present in {database_path}.")
        raw = (
            pl.concat([raw, benchmark_raw.select(OHLCV_COLUMNS)], how="vertical")
            .unique(subset=["symbol", "date"], keep="last")
            .sort(["symbol", "date"])
        )
        raw.write_csv(data_csv_path)

    return raw.select(OHLCV_COLUMNS)


def build_features(raw: pl.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    asset_raw = raw.filter(pl.col("symbol") != BENCHMARK)
    raw_prices = raw.select(["date", "symbol", "high", "low", "close"]).to_pandas()
    raw_prices["date"] = pd.to_datetime(raw_prices["date"])
    raw_prices = raw_prices.drop_duplicates(["symbol", "date"], keep="last").sort_values(
        ["symbol", "date"]
    )

    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    price_cols = ["open", "high", "low", "close"]

    featured = (
        asset_raw.unique(subset=["symbol", "date"], keep="last")
        .sort(["symbol", "date"])
        .with_columns(
            [
                pl.col(col).forward_fill().over("symbol").backward_fill().over("symbol").alias(col)
                for col in ohlcv_cols
            ]
        )
        .with_columns(
            [
                pl.when(pl.col(col) <= 0).then(pl.lit(1e-10)).otherwise(pl.col(col)).alias(col)
                for col in ohlcv_cols
            ]
        )
        .with_columns(
            pl.max_horizontal([pl.col(col) for col in price_cols]).alias("high"),
            pl.min_horizontal([pl.col(col) for col in price_cols]).alias("low"),
        )
        .with_columns(
            pl.col("open").clip(pl.col("low"), pl.col("high")).alias("open"),
            pl.col("close").clip(pl.col("low"), pl.col("high")).alias("close"),
        )
        .filter(pl.len().over("symbol") >= 252)
    )

    featured = featured.with_columns(
        (pl.col("close") / pl.col("close").shift(21).over("symbol") - 1).alias("_qsmom_roc_fast"),
        (pl.col("close") / pl.col("close").shift(252).over("symbol") - 1).alias("_qsmom_roc_slow"),
        (pl.col("close") / pl.col("close").shift(126).over("symbol") - 1).alias("_qsmom_returns"),
    ).with_columns(
        pl.when(pl.col("_qsmom_roc_fast") > 0)
        .then(pl.col("_qsmom_roc_slow") + pl.col("_qsmom_returns"))
        .otherwise(pl.lit(None))
        .alias("close_qsmom_21_252_126")
    )

    featured = featured.with_columns(
        pl.col("close")
        .ewm_mean(span=50, adjust=False, min_samples=50)
        .over("symbol")
        .alias("_ema_fast"),
        pl.col("close")
        .ewm_mean(span=200, adjust=False, min_samples=200)
        .over("symbol")
        .alias("_ema_slow"),
    ).with_columns((pl.col("_ema_fast") - pl.col("_ema_slow")).alias("_macd_line"))
    featured = featured.with_columns(
        pl.col("_macd_line")
        .ewm_mean(span=30, adjust=False, min_samples=30)
        .over("symbol")
        .alias("_macd_signal")
    ).with_columns(
        (pl.col("_macd_line") - pl.col("_macd_signal")).alias("close_macd_histogram_50_200_30")
    )

    rsi_delta = pl.col("close").diff().over("symbol")
    rsi_gain = pl.when(rsi_delta > 0).then(rsi_delta).otherwise(0.0)
    rsi_loss = pl.when(rsi_delta < 0).then(-rsi_delta).otherwise(0.0)
    featured = featured.with_columns(
        (pl.col("close") / pl.col("close").shift(63).over("symbol") - 1).alias("close_roc_0_63"),
        (pl.col("close") / pl.col("close").shift(252).over("symbol") - 1).alias("close_roc_0_252"),
        (
            100
            - (
                100
                / (
                    1
                    + rsi_gain.ewm_mean(alpha=1 / 63, adjust=False, min_samples=63).over("symbol")
                    / (
                        rsi_loss.ewm_mean(
                            alpha=1 / 63,
                            adjust=False,
                            min_samples=63,
                        ).over("symbol")
                        + 1e-8
                    )
                )
            )
        ).alias("close_rsi_63"),
    )

    featured = (
        featured.with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1).alias("_daily_return"),
            (pl.col("close") / pl.col("close").shift(252).over("symbol") - 1).alias(
                "_total_return"
            ),
        )
        .with_columns(
            (pl.col("_daily_return") > 0)
            .cast(pl.Float64)
            .rolling_sum(252, min_samples=252)
            .over("symbol")
            .alias("_positive_count"),
            (pl.col("_daily_return") < 0)
            .cast(pl.Float64)
            .rolling_sum(252, min_samples=252)
            .over("symbol")
            .alias("_negative_count"),
        )
        .with_columns(
            (
                pl.col("_total_return")
                * ((pl.col("_negative_count") - pl.col("_positive_count")) / 252)
            ).alias("close_fip_momentum_252")
        )
    )

    featured = featured.with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).over("symbol").log()).alias("_log_return")
    ).with_columns(
        (
            pl.col("_log_return").rolling_std(252, min_samples=252).over("symbol") * math.sqrt(252)
        ).alias("close_volatility_annualized_252"),
        (pl.col("close").shift(-FORWARD_PERIOD).over("symbol") / pl.col("close") - 1).alias(
            TARGET_COL
        ),
    )

    model_frame = featured.select(["date", "symbol", TARGET_COL, *PREDICTOR_COLS]).to_pandas()
    model_frame["date"] = pd.to_datetime(model_frame["date"])
    model_frame = (
        model_frame.replace([np.inf, -np.inf], np.nan)
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )
    model_frame["feature_count"] = model_frame[PREDICTOR_COLS].notna().sum(axis=1)

    dataset_summary = pd.Series(
        {
            "rows": len(model_frame),
            "dates": model_frame["date"].nunique(),
            "symbols": model_frame["symbol"].nunique(),
            "feature_ready_rows": int((model_frame["feature_count"] >= MIN_FEATURE_COUNT).sum()),
            "training_ready_rows": int(
                (
                    (model_frame["feature_count"] >= MIN_FEATURE_COUNT)
                    & model_frame[TARGET_COL].notna()
                ).sum()
            ),
        },
        name="training_data",
    )
    return model_frame, raw_prices, dataset_summary


def run_predictions(
    model_frame: pd.DataFrame,
    diagnostic_output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    all_dates = pd.DatetimeIndex(model_frame["date"].drop_duplicates()).sort_values()
    rebalance_dates = all_dates[::REBALANCE_FREQUENCY]
    rebalance_dates = pd.DatetimeIndex(
        [
            date
            for date in rebalance_dates
            if any(start <= date <= end for start, end in TEST_INTERVALS)
        ]
    )
    date_position = pd.Series(range(len(all_dates)), index=all_dates)

    prediction_rows = []
    for idx, rebalance_date in enumerate(rebalance_dates, start=1):
        if idx % 25 == 0 or idx == 1:
            print(f"training rebalance {idx}/{len(rebalance_dates)}: {rebalance_date.date()}")

        latest_trainable_idx = int(date_position.loc[rebalance_date]) - FORWARD_PERIOD
        trainable_dates = all_dates[: latest_trainable_idx + 1]
        trainable_dates = pd.DatetimeIndex(
            [
                date
                for date in trainable_dates
                if not any(start <= date <= end for start, end in TEST_INTERVALS)
            ]
        )
        if len(trainable_dates) < max(TRAIN_WINDOW, CLASSIFICATION_WINDOW):
            continue

        train_dates = trainable_dates[-TRAIN_WINDOW:]
        threshold_dates = trainable_dates[-CLASSIFICATION_WINDOW:]
        train = model_frame[model_frame["date"].isin(train_dates)].copy()
        train = train[(train[TARGET_COL].notna()) & (train["feature_count"] >= MIN_FEATURE_COUNT)]

        threshold_frame = train[train["date"].isin(threshold_dates)]
        predict = model_frame[
            (model_frame["date"] == rebalance_date)
            & (model_frame["feature_count"] >= MIN_FEATURE_COUNT)
        ].copy()
        if len(train) == 0 or len(threshold_frame) == 0 or len(predict) == 0:
            continue

        threshold_source = train if CLASSIFICATION_THRESHOLD_SCOPE == "train" else threshold_frame
        thresholds = np.quantile(threshold_source[TARGET_COL], CLASS_QUANTILES)
        train_labels = np.searchsorted(thresholds, train[TARGET_COL], side="left")
        classes = np.array(sorted(np.unique(train_labels)), dtype=int)
        if len(classes) < 2:
            continue

        model_labels = np.searchsorted(classes, train_labels)
        fit_kwargs = {}
        if BALANCED_SAMPLE_WEIGHT:
            class_counts = np.bincount(model_labels, minlength=len(classes))
            sample_weight = 1.0 / class_counts[model_labels]
            fit_kwargs["sample_weight"] = sample_weight / sample_weight.mean()

        model = xgb.XGBClassifier(**XGB_PARAMS, num_class=len(classes))
        model.fit(train[PREDICTOR_COLS], model_labels, **fit_kwargs)
        raw_probabilities = model.predict_proba(predict[PREDICTOR_COLS])

        probabilities = np.zeros((len(predict), len(CLASS_SCORES)))
        for source_idx, class_label in enumerate(classes):
            probabilities[:, class_label] = raw_probabilities[:, source_idx]
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

        target_values = predict[TARGET_COL].to_numpy(dtype=float)
        realized_class = np.full(len(predict), np.nan)
        finite_target = np.isfinite(target_values)
        realized_class[finite_target] = np.searchsorted(
            thresholds,
            target_values[finite_target],
            side="left",
        )

        out = (
            predict[["date", "symbol", TARGET_COL]]
            .rename(columns={TARGET_COL: "realized_forward_return"})
            .copy()
        )
        out["predicted_class"] = probabilities.argmax(axis=1)
        out["realized_class"] = realized_class
        out["score"] = probabilities @ CLASS_SCORES
        for col_idx, col_name in enumerate(CLASS_PROB_COLS):
            out[col_name] = probabilities[:, col_idx]
        out[SIGNAL_PROB_SUM_COL] = out[SIGNAL_PROB_COLS].sum(axis=1)
        out["score_prob"] = out["score"] * out[SIGNAL_PROB_SUM_COL]
        prediction_rows.append(out)

    if not prediction_rows:
        raise RuntimeError("No prediction rows generated.")

    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions = predictions.sort_values(
        ["date", RANK_COL, "symbol"],
        ascending=[True, False, True],
    )

    def select_cross_section(cross_section: pd.DataFrame) -> pd.DataFrame:
        selected = cross_section[cross_section[SIGNAL_PROB_SUM_COL] > SIGNAL_THRESHOLD].copy()
        if MIN_PREDICTED_CLASS is not None:
            selected = selected[selected["predicted_class"] >= MIN_PREDICTED_CLASS].copy()
        if len(selected) < FALLBACK_MIN_POSITIONS:
            fallback = (
                cross_section[~cross_section["symbol"].isin(selected["symbol"])]
                .sort_values([RANK_COL, "symbol"], ascending=[False, True])
                .head(FALLBACK_MIN_POSITIONS - len(selected))
            )
            selected = pd.concat([selected, fallback], ignore_index=True)
        selected = selected.sort_values([RANK_COL, "symbol"], ascending=[False, True]).head(
            MAX_POSITIONS
        )
        selected["target_weight"] = 1 / MAX_POSITIONS
        return selected

    selected_predictions = pd.concat(
        [
            select_cross_section(cross_section)
            for _, cross_section in predictions.groupby("date", sort=True)
        ],
        ignore_index=True,
    )
    if selected_predictions.empty:
        if diagnostic_output_dir is not None:
            write_prediction_diagnostics(diagnostic_output_dir, predictions)
        raise RuntimeError(
            "No predictions passed signal filters: "
            f"{SIGNAL_PROB_SUM_COL}>{SIGNAL_THRESHOLD}, "
            f"min_predicted_class={MIN_PREDICTED_CLASS}, "
            f"fallback_min_positions={FALLBACK_MIN_POSITIONS}."
        )

    scored_predictions = predictions[predictions["realized_forward_return"].notna()].copy()
    selected_eval = pd.concat(
        [
            select_cross_section(cross_section)
            for _, cross_section in scored_predictions.groupby("date", sort=True)
        ],
        ignore_index=True,
    )
    class_eval = scored_predictions[scored_predictions["realized_class"].notna()]
    run_eval = pd.Series(
        {
            "prediction_rows": len(scored_predictions),
            "prediction_dates": scored_predictions["date"].nunique(),
            "selected_rows": len(selected_eval),
            "selected_dates": selected_eval["date"].nunique(),
            "top_mean_forward_return": selected_eval["realized_forward_return"].mean(),
            "class_accuracy": (
                class_eval["predicted_class"] == class_eval["realized_class"].astype(int)
            ).mean(),
        },
        name="run_eval",
    )
    return predictions, selected_predictions, selected_eval, run_eval


def build_orders(
    raw_prices: pd.DataFrame,
    selected_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    close = raw_prices.pivot(index="date", columns="symbol", values="close").sort_index()
    close = close.sort_index(axis=1)
    high = raw_prices.pivot(index="date", columns="symbol", values="high").sort_index()
    high = high.reindex(index=close.index, columns=close.columns)
    low = raw_prices.pivot(index="date", columns="symbol", values="low").sort_index()
    low = low.reindex(index=close.index, columns=close.columns)
    target_weights = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    next_session = dict(zip(close.index[:-1], close.index[1:], strict=False))
    previous_symbols: set[str] = set()
    regime_allowed = benchmark_regime_allowed(close)

    for prediction_date, selected in selected_predictions.groupby("date", sort=True):
        execution_date = next_session.get(pd.Timestamp(prediction_date))
        if execution_date is None:
            continue

        current_symbols = (
            set(selected["symbol"])
            if regime_allowed is None or bool(regime_allowed.get(pd.Timestamp(prediction_date), False))
            else set()
        )
        target_weights.loc[execution_date, sorted(previous_symbols - current_symbols)] = 0.0
        entry_symbols = current_symbols if REBALANCE_EXISTING_POSITIONS else current_symbols - previous_symbols
        target_weights.loc[execution_date, sorted(entry_symbols)] = 1 / MAX_POSITIONS
        previous_symbols = current_symbols

    if STOP_LOSS is not None or TAKE_PROFIT is not None or TRAILING_STOP is not None:
        target_weights = apply_exit_rules(close, target_weights, high, low)

    used_symbols = target_weights.notna().any(axis=0)
    close_bt = close.loc[:, used_symbols]
    weights_bt = target_weights.loc[:, used_symbols]

    orders_summary = pd.Series(
        {
            "backtest_symbols": int(used_symbols.sum()),
            "order_dates": int(weights_bt.notna().any(axis=1).sum()),
            "first_order_date": weights_bt.dropna(how="all").index.min().date(),
            "last_order_date": weights_bt.dropna(how="all").index.max().date(),
        },
        name="orders",
    )

    orders = (
        weights_bt.stack(future_stack=True)
        .dropna()
        .rename("target_weight")
        .reset_index()
        .rename(
            columns={"date": "execution_date", "level_0": "execution_date", "level_1": "symbol"}
        )
    )
    if list(orders.columns[:2]) != ["execution_date", "symbol"]:
        orders.columns = ["execution_date", "symbol", "target_weight"]

    order_realized = selected_predictions.copy()
    order_realized["execution_date"] = order_realized["date"].map(next_session)
    order_realized = order_realized.rename(columns={"date": "prediction_date"})
    order_realized = order_realized.dropna(subset=["execution_date"])
    order_realized = order_realized[
        ["prediction_date", "execution_date", "symbol", "target_weight"]
        + [
            "realized_forward_return",
            "predicted_class",
            "realized_class",
            "score",
            "score_prob",
            *CLASS_PROB_COLS,
            SIGNAL_PROB_SUM_COL,
        ]
    ]

    return close_bt, weights_bt, orders, orders_summary, order_realized


def benchmark_regime_allowed(close: pd.DataFrame) -> pd.Series | None:
    if not BENCHMARK_REGIME_ENABLED:
        return None
    return benchmark_above_sma(close)


def benchmark_above_sma(close: pd.DataFrame) -> pd.Series:
    if BENCHMARK not in close.columns:
        raise ValueError(f"{BENCHMARK} is required for benchmark_regime.")
    benchmark_close = close[BENCHMARK]
    benchmark_sma = benchmark_close.rolling(
        BENCHMARK_REGIME_WINDOW,
        min_periods=BENCHMARK_REGIME_MIN_PERIODS,
    ).mean()
    return (benchmark_close > benchmark_sma).fillna(False)


def apply_exit_rules(
    close: pd.DataFrame,
    target_weights: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
) -> pd.DataFrame:
    adjusted = target_weights.copy()
    active_entries: dict[str, tuple[float, int, float]] = {}
    dates = list(close.index)
    stop_loss_allowed = None
    if STOP_LOSS_BENCHMARK_FILTER == "below_sma":
        stop_loss_allowed = ~benchmark_above_sma(close)

    for idx, date in enumerate(dates):
        orders = adjusted.loc[date].dropna()
        for symbol, target_weight in orders.items():
            if target_weight > 0:
                price = close.at[date, symbol]
                if pd.notna(price) and symbol not in active_entries:
                    active_entries[str(symbol)] = (float(price), idx, float(price))
            else:
                active_entries.pop(str(symbol), None)

        if idx + 1 >= len(dates):
            continue

        exit_symbols: list[str] = []
        entry_updates: dict[str, tuple[float, int, float]] = {}
        for symbol, (entry_price, entry_idx, peak_price) in active_entries.items():
            if idx <= entry_idx:
                continue
            close_price = close.at[date, symbol]
            if pd.isna(close_price) or entry_price <= 0:
                continue
            stop_price = close_price
            take_price = close_price
            current_peak = max(peak_price, float(take_price))
            holding_days = idx - entry_idx
            benchmark_filter_passed = (
                stop_loss_allowed is None or bool(stop_loss_allowed.get(date, False))
            )
            stop_hit = (
                STOP_LOSS is not None
                and holding_days >= STOP_LOSS_MIN_HOLD
                and benchmark_filter_passed
                and float(stop_price) / entry_price - 1 <= -STOP_LOSS
            )
            take_hit = TAKE_PROFIT is not None and float(take_price) / entry_price - 1 >= TAKE_PROFIT
            trail_hit = (
                TRAILING_STOP is not None
                and float(stop_price) / current_peak - 1 <= -TRAILING_STOP
            )
            if stop_hit or take_hit or trail_hit:
                adjusted.loc[dates[idx + 1], symbol] = 0.0
                exit_symbols.append(symbol)
            else:
                entry_updates[symbol] = (entry_price, entry_idx, current_peak)

        for symbol in exit_symbols:
            active_entries.pop(symbol, None)
        active_entries.update(entry_updates)

    return adjusted


def order_win_rate(order_realized: pd.DataFrame) -> float:
    if order_realized.empty:
        return float("nan")
    return float((order_realized["realized_forward_return"] > 0).sum() / len(order_realized))


def run_portfolio(
    close_bt: pd.DataFrame, weights_bt: pd.DataFrame, order_realized: pd.DataFrame
) -> tuple[vbt.Portfolio, pd.Series, pd.Series, pd.Series, pd.Series]:
    portfolio = vbt.Portfolio.from_orders(
        close=close_bt,
        size=weights_bt,
        size_type="TargetPercent",
        fees=FEE_RATE,
        cash_sharing=True,
        group_by=True,
        init_cash=INIT_CASH,
        freq=pd.tseries.frequencies.to_offset(close_bt.index[1] - close_bt.index[0]),
    )

    equity = portfolio.get_value(group_by=True).rename("equity")
    returns = portfolio.get_returns(group_by=True).astype(float).rename("portfolio_return")

    returns_list = returns.to_list()
    equity_list = equity.to_list()
    returns_std = np.std(returns_list, ddof=1)
    downside = [value for value in returns_list if value < 0]
    downside_std = np.std(downside, ddof=1)
    running_peak = np.maximum.accumulate(equity_list)
    qts_metrics = pd.Series(
        {
            "sharpe": np.mean(returns_list) / returns_std * np.sqrt(252),
            "sortino": np.mean(returns_list) / downside_std * np.sqrt(252),
            "cagr": (equity_list[-1] / equity_list[0]) ** (252 / (len(equity_list) - 1)) - 1,
            "max_drawdown": abs(np.min((np.array(equity_list) - running_peak) / running_peak)),
            "win_rate": order_win_rate(order_realized),
        },
        name="qts_metrics",
    )

    portfolio_stats = portfolio.stats()
    if not isinstance(portfolio_stats, pd.Series):
        portfolio_stats = portfolio_stats.squeeze()
    portfolio_stats = portfolio_stats.rename("portfolio_stats")
    return portfolio, equity, returns, qts_metrics, portfolio_stats


def benchmark_returns_from_raw(raw: pl.DataFrame) -> pd.Series:
    benchmark = (
        raw.filter(pl.col("symbol") == BENCHMARK)
        .select(["date", "close"])
        .unique(subset=["date"], keep="last")
        .sort("date")
        .to_pandas()
    )
    if benchmark.empty:
        raise ValueError(f"{BENCHMARK} is not present in raw.")

    test_start = pd.Timestamp(TEST_INTERVALS[0][0], tz="UTC")
    test_end = pd.Timestamp(TEST_INTERVALS[-1][1], tz="UTC")
    benchmark["date"] = pd.to_datetime(benchmark["date"], utc=True).dt.normalize()
    return (
        benchmark.set_index("date")["close"]
        .pct_change()
        .loc[test_start:test_end]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .rename(BENCHMARK)
    )


def align_pyfolio_returns(
    returns: pd.Series, benchmark_returns: pd.Series
) -> tuple[pd.Series, pd.Series]:
    pyfolio_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    pyfolio_returns.index = pd.to_datetime(pyfolio_returns.index, utc=True).normalize()
    test_start = pd.Timestamp(TEST_INTERVALS[0][0], tz="UTC")
    test_end = pd.Timestamp(TEST_INTERVALS[-1][1], tz="UTC")
    pyfolio_returns = pyfolio_returns.loc[test_start:test_end]
    pyfolio_returns.name = "ml_factor_classifier"

    aligned_returns = pd.concat([pyfolio_returns, benchmark_returns], axis=1, join="inner").dropna()
    if aligned_returns.empty:
        raise ValueError(
            f"No {BENCHMARK} returns overlap portfolio returns between "
            f"{test_start.date()} and {test_end.date()}."
        )
    return aligned_returns["ml_factor_classifier"], aligned_returns[BENCHMARK]


def save_images(
    output_dir: Path,
    equity: pd.Series,
    returns: pd.Series,
    pyfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    skip_tearsheet: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    equity.plot(ax=ax, title="VN100 ML classifier equity")
    ax.set_xlabel("date")
    ax.set_ylabel("equity")
    fig.tight_layout()
    fig.savefig(output_dir / "equity_curve.png", dpi=150)
    plt.close(fig)

    drawdown = (equity / equity.cummax() - 1).rename("drawdown")
    fig, ax = plt.subplots(figsize=(12, 4))
    drawdown.plot(ax=ax, title="VN100 ML classifier drawdown")
    ax.set_xlabel("date")
    ax.set_ylabel("drawdown")
    fig.tight_layout()
    fig.savefig(output_dir / "drawdown.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    cumulative = (1 + returns).cumprod() - 1
    cumulative.plot(ax=ax, label="portfolio")
    benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
    benchmark_cumulative.plot(ax=ax, label=BENCHMARK)
    ax.set_title("Cumulative returns")
    ax.set_xlabel("date")
    ax.set_ylabel("return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_returns.png", dpi=150)
    plt.close(fig)

    if skip_tearsheet:
        return

    plt.close("all")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tear_sheet_fig = pf.create_returns_tear_sheet(
            pyfolio_returns,
            benchmark_rets=benchmark_returns,
            return_fig=True,
        )
    if tear_sheet_fig is None:
        tear_sheet_fig = plt.gcf()
    tear_sheet_fig.savefig(output_dir / "tearsheet.png", dpi=150, bbox_inches="tight")
    tear_sheet_fig.savefig(output_dir / "tearsheet.pdf", bbox_inches="tight")
    plt.close(tear_sheet_fig)


def save_outputs(
    output_dir: Path,
    raw: pl.DataFrame,
    model_frame: pd.DataFrame,
    dataset_summary: pd.Series,
    predictions: pd.DataFrame,
    selected_predictions: pd.DataFrame,
    selected_eval: pd.DataFrame,
    run_eval: pd.Series,
    weights_bt: pd.DataFrame,
    orders: pd.DataFrame,
    orders_summary: pd.Series,
    order_realized: pd.DataFrame,
    portfolio: vbt.Portfolio,
    equity: pd.Series,
    returns: pd.Series,
    qts_metrics: pd.Series,
    portfolio_stats: pd.Series,
    pyfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    pyfolio_stats: pd.Series,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    params = pd.Series(
        {
            "run_id": RUN_ID,
            "config_path": CONFIG_PATH,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "benchmark": BENCHMARK,
            "init_cash": INIT_CASH,
            "fee_rate": FEE_RATE,
            "forward_period": FORWARD_PERIOD,
            "train_window": TRAIN_WINDOW,
            "classification_window": CLASSIFICATION_WINDOW,
            "classification_threshold_scope": CLASSIFICATION_THRESHOLD_SCOPE,
            "balanced_sample_weight": BALANCED_SAMPLE_WEIGHT,
            "rebalance_frequency": REBALANCE_FREQUENCY,
            "rebalance_existing_positions": REBALANCE_EXISTING_POSITIONS,
            "max_positions": MAX_POSITIONS,
            "signal_threshold": SIGNAL_THRESHOLD,
            "min_predicted_class": MIN_PREDICTED_CLASS,
            "stop_loss": STOP_LOSS,
            "stop_loss_min_hold": STOP_LOSS_MIN_HOLD,
            "stop_loss_benchmark_filter": STOP_LOSS_BENCHMARK_FILTER,
            "take_profit": TAKE_PROFIT,
            "trailing_stop": TRAILING_STOP,
            "benchmark_regime_enabled": BENCHMARK_REGIME_ENABLED,
            "benchmark_regime_rule": BENCHMARK_REGIME_RULE,
            "benchmark_regime_window": BENCHMARK_REGIME_WINDOW,
            "benchmark_regime_min_periods": BENCHMARK_REGIME_MIN_PERIODS,
            "raw_rows": raw.height,
            "raw_symbols": raw.select(pl.col("symbol").n_unique()).item(),
            "asset_symbols": raw.filter(pl.col("symbol") != BENCHMARK)
            .select(pl.col("symbol").n_unique())
            .item(),
        },
        name="params",
    )
    for filename, series in {
        "params.csv": params,
        "dataset_summary.csv": dataset_summary,
        "run_eval.csv": run_eval,
        "orders_summary.csv": orders_summary,
        "qts_metrics.csv": qts_metrics,
        "portfolio_stats.csv": portfolio_stats,
        "pyfolio_stats.csv": pyfolio_stats,
    }.items():
        series.rename("value").to_frame().to_csv(output_dir / filename)

    (output_dir / "effective_config.yaml").write_text(
        yaml.safe_dump(CONFIG_EFFECTIVE, sort_keys=False)
    )

    for filename, frame in {
        "model_frame.csv": model_frame,
        "predictions.csv": predictions,
        "selected_predictions.csv": selected_predictions,
        "selected_realized_results.csv": selected_eval,
        "orders.csv": orders,
        "orders_realized.csv": order_realized,
    }.items():
        frame.to_csv(output_dir / filename, index=False)
    weights_bt.to_csv(output_dir / "target_weights.csv", index_label="date")

    portfolio_frame = pd.concat([equity, returns], axis=1)
    portfolio_frame["drawdown"] = equity / equity.cummax() - 1
    portfolio_frame.index.name = "date"
    portfolio_frame.to_csv(output_dir / "portfolio.csv")

    aligned_frame = pd.concat([pyfolio_returns, benchmark_returns], axis=1)
    aligned_frame.index.name = "date"
    aligned_frame.to_csv(output_dir / "pyfolio_aligned_returns.csv")

    try:
        readable_orders = portfolio.orders.records_readable
        readable_orders.to_csv(output_dir / "vectorbt_orders.csv", index=False)
    except Exception as exc:  # pragma: no cover - best-effort export
        (output_dir / "vectorbt_orders_error.txt").write_text(str(exc))


def write_manifest(output_dir: Path) -> None:
    manifest = {
        "output_dir": str(output_dir),
        "files": sorted({path.name for path in output_dir.iterdir()} | {"manifest.json"}),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def main() -> None:
    repo_root = find_repo_root()
    args = parse_args(repo_root)
    load_config(args.config, repo_root)

    if args.output_dir is None:
        args.output_dir = default_output_dir(repo_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"output_dir={args.output_dir}")
    print(f"config={CONFIG_PATH}")
    print(
        "params:",
        f"run_id={RUN_ID}",
        f"fwd={FORWARD_PERIOD}",
        f"cw={CLASSIFICATION_WINDOW}",
        f"threshold_scope={CLASSIFICATION_THRESHOLD_SCOPE}",
        f"balanced_sample_weight={BALANCED_SAMPLE_WEIGHT}",
        f"tw={TRAIN_WINDOW}",
        f"rb={REBALANCE_FREQUENCY}",
        f"rebalance_existing_positions={REBALANCE_EXISTING_POSITIONS}",
        f"max_pos={MAX_POSITIONS}",
        f"threshold={SIGNAL_THRESHOLD}",
        f"min_predicted_class={MIN_PREDICTED_CLASS}",
        f"stop_loss={STOP_LOSS}",
        f"stop_loss_min_hold={STOP_LOSS_MIN_HOLD}",
        f"stop_loss_benchmark_filter={STOP_LOSS_BENCHMARK_FILTER}",
        f"take_profit={TAKE_PROFIT}",
        f"trailing_stop={TRAILING_STOP}",
        f"benchmark_regime_enabled={BENCHMARK_REGIME_ENABLED}",
    )
    raw = load_or_build_raw(
        repo_root,
        args.data_csv,
        args.database,
        force_refresh=args.force_refresh_data,
    )
    print(
        "loaded raw:",
        raw.height,
        "rows,",
        raw.select(pl.col("symbol").n_unique()).item(),
        "symbols",
    )

    model_frame, raw_prices, dataset_summary = build_features(raw)
    print(dataset_summary.to_string())

    predictions, selected_predictions, selected_eval, run_eval = run_predictions(
        model_frame,
        args.output_dir,
    )
    print(run_eval.to_string())

    close_bt, weights_bt, orders, orders_summary, order_realized = build_orders(
        raw_prices,
        selected_predictions,
    )
    print(orders_summary.to_string())

    portfolio, equity, returns, qts_metrics, portfolio_stats = run_portfolio(
        close_bt,
        weights_bt,
        order_realized,
    )
    benchmark_returns = benchmark_returns_from_raw(raw)
    pyfolio_returns, benchmark_returns = align_pyfolio_returns(returns, benchmark_returns)
    pyfolio_stats = pf.timeseries.perf_stats(
        pyfolio_returns,
        factor_returns=benchmark_returns,
    ).rename("pyfolio_stats")

    save_outputs(
        args.output_dir,
        raw,
        model_frame,
        dataset_summary,
        predictions,
        selected_predictions,
        selected_eval,
        run_eval,
        weights_bt,
        orders,
        orders_summary,
        order_realized,
        portfolio,
        equity,
        returns,
        qts_metrics,
        portfolio_stats,
        pyfolio_returns,
        benchmark_returns,
        pyfolio_stats,
    )
    save_images(
        args.output_dir,
        equity,
        returns,
        pyfolio_returns,
        benchmark_returns,
        skip_tearsheet=args.skip_tearsheet,
    )
    write_manifest(args.output_dir)
    print(f"wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
