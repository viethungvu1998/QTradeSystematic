# QTradeSystematic - VN ML classification end-to-end flow
# Data/config -> feature engineering -> XGB five-class classifier -> capped top-k backtest.
# ruff: noqa: E402

from __future__ import annotations

import asyncio
import atexit
import importlib
import json
import math
import os
import sys
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Any

repo_root = Path.cwd().resolve()
while not (repo_root / "qts").exists() and repo_root != repo_root.parent:
    repo_root = repo_root.parent

if not (repo_root / "qts").exists():
    raise FileNotFoundError("Could not find the repository root containing the qts package.")

repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
import yaml

import qts as qts_lib

importlib.reload(qts_lib)

from qts.config.builder import Config
from qts.orchestration.runtime import build_data_manager
from qts.research.backtest.base import BacktestConfig
from qts.research.backtest.metrics import build_metrics
from qts.research.backtest.vanilla import (
    export_vanilla_backtest_artifacts,
    run_vanilla_backtest_from_signals,
    save_vanilla_comparison_report_pdf,
)
from qts.research.features.forward_returns import ForwardReturns
from qts.research.features.indicators.momentum import MovingAverageFeature, ROCFeature, RSIFeature
from qts.research.features.indicators.statistical import ZScoreFeature
from qts.research.features.indicators.trend import ADXFeature, MACDFeature
from qts.research.features.indicators.volatility import HistVolFeature
from qts.research.features.indicators.volume import VolumeRatioFeature
from qts.research.features.preprocessor import preprocess_ohlcv
from qts.research.features.transforms.momentum import qsmom_transform
from qts.research.features.transforms.price import (
    average_price_transform,
    log_price_transform,
    log_volume_transform,
)
from qts.research.strategies.base import BaseStrategy
from qts.utils.classification import (
    QuantileClassThresholds,
    align_class_probabilities,
    class_scores_from_probabilities,
)
from qts.utils.date_intervals import (
    DateInterval,
    date_in_intervals,
    filter_dates_in_intervals,
    format_date_intervals,
    select_training_dates,
    test_intervals_from_config,
)
from qts.utils.export import (
    export_backtest_report_pdf,
    export_portfolio_snapshots,
    export_trade_log,
)
from qts.utils.paths import backtest_exports_dir, tearsheet_dir

CONFIG_PATH = repo_root / "configs" / "strategies" / "ml_factor" / "vn100_ml_example.yaml"
DEFAULT_PARAMS_PATH = Path(
    os.getenv(
        "QTS_CLASSIFICATION_DEFAULTS_PATH",
        str(
            repo_root
            / "configs"
            / "notebooks"
            / "stock_ml_e2e_classification"
            / "default_params.yaml"
        ),
    )
)
RUNTIME_LOG_DIR = repo_root / ".qts_notebooke_runtime"
ABLATION_RESULTS_PATH = RUNTIME_LOG_DIR / "ablation_results.tsv"
ABLATION_RANKING_PATH = RUNTIME_LOG_DIR / "ablation_ranking.md"
EXPECTED_BENCHMARKS = {"VNINDEX", "VN:VNINDEX"}


class TeeStream:
    """Mirror writes to the original stream and a runtime log file."""

    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.streams[0], name)


def enable_runtime_logging() -> Path:
    RUNTIME_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUNTIME_LOG_DIR / f"stock_ml_e2e_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)

    def close_runtime_log() -> None:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.flush()
        log_file.close()

    atexit.register(close_runtime_log)
    return log_path


def load_yaml_defaults(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing default params YAML: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Default params YAML must contain a mapping: {path}")
    return payload


DEFAULT_PARAMS = load_yaml_defaults(DEFAULT_PARAMS_PATH)


def _default_value(path: tuple[str, ...], fallback: Any) -> Any:
    current: Any = DEFAULT_PARAMS
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return fallback
        current = current[key]
    return current


def _default_path(path: tuple[str, ...], fallback: Path) -> Path:
    raw = _default_value(path, str(fallback))
    resolved = Path(str(raw))
    return resolved if resolved.is_absolute() else repo_root / resolved


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


def _env_optional_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    if value.strip().lower() in {"none", "null"}:
        return None
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value in (None, "") else float(value)


def _env_optional_float(name: str, default: float | None = None) -> float | None:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    if value.strip().lower() in {"none", "null", "off", "false"}:
        return None
    return float(value)


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value in (None, "") else value


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


MODEL_WEIGHTS_DIR = Path(
    os.getenv(
        "QTS_MODEL_WEIGHTS_DIR",
        str(
            _default_path(
                ("paths", "model_weights_dir"),
                RUNTIME_LOG_DIR / "model_weights" / "stock_ml_e2e_classification",
            )
        ),
    )
)
EFFECTIVE_YAML_DIR = Path(
    os.getenv(
        "QTS_EFFECTIVE_YAML_DIR",
        str(
            _default_path(
                ("paths", "effective_yaml_dir"),
                RUNTIME_LOG_DIR / "yaml" / "stock_ml_e2e_classification",
            )
        ),
    )
)
REPORT_PDF_DIR = Path(
    os.getenv(
        "QTS_REPORT_PDF_DIR",
        str(_default_path(("paths", "report_pdf_dir"), repo_root / "output" / "pdf")),
    )
)

REFERENCE_FORWARD_PERIOD = _env_int(
    "QTS_FORWARD_PERIOD",
    int(_default_value(("run", "forward_period"), 21)),
)
CLASSIFICATION_WINDOW = _env_int(
    "QTS_CLASSIFICATION_WINDOW",
    int(_default_value(("run", "classification_window"), 120)),
)
REFERENCE_TRAIN_WINDOW = _env_int(
    "QTS_TRAIN_WINDOW",
    int(_default_value(("run", "train_window"), 504)),
)
MAX_POSITIONS = _env_int(
    "QTS_MAX_POSITIONS",
    int(_default_value(("run", "max_positions"), 4)),
)
XGB_N_ESTIMATORS = _env_int(
    "QTS_XGB_N_ESTIMATORS",
    int(_default_value(("xgb", "n_estimators"), 5)),
)
XGB_LEARNING_RATE = _env_float(
    "QTS_XGB_LEARNING_RATE",
    float(_default_value(("xgb", "learning_rate"), 0.14)),
)
XGB_MAX_DEPTH = _env_optional_int(
    "QTS_XGB_MAX_DEPTH",
    int(_default_value(("xgb", "max_depth"), 3)),
)
XGB_N_JOBS = _env_int("QTS_XGB_N_JOBS", int(_default_value(("xgb", "n_jobs"), -1)))
XGB_RANDOM_STATE = _env_int(
    "QTS_XGB_RANDOM_STATE",
    int(_default_value(("xgb", "random_state"), 123)),
)
XGB_OBJECTIVE = _env_str(
    "QTS_XGB_OBJECTIVE",
    str(_default_value(("xgb", "objective"), "multi:softprob")),
)
XGB_EVAL_METRIC = _env_str(
    "QTS_XGB_EVAL_METRIC",
    str(_default_value(("xgb", "eval_metric"), "mlogloss")),
)
XGB_TREE_METHOD = _env_str(
    "QTS_XGB_TREE_METHOD",
    str(_default_value(("xgb", "tree_method"), "hist")),
)

TARGET_COL = f"Return_fwd_{REFERENCE_FORWARD_PERIOD}"
CLASS_COUNT = 5
CLASS_QUANTILES = tuple(
    float(value)
    for value in _default_value(("classification", "class_quantiles"), [0.15, 0.45, 0.6, 0.85])
)
CLASS_SCORES = np.array(
    _default_value(("classification", "class_scores"), [-2.0, -1.0, 0.0, 1.0, 2.0]),
    dtype=float,
)
CLASS_PROB_COLS = [f"class_{label}_prob" for label in range(CLASS_COUNT)]
SIGNAL_PROBABILITY_COLS = [
    str(value)
    for value in _default_value(("signal", "probability_cols"), ["class_3_prob", "class_4_prob"])
]
SIGNAL_PROBABILITY_SUM_COL = str(
    _default_value(("signal", "probability_sum_col"), "class_3_4_prob")
)
_DEFAULT_SIGNAL_PROBABILITY_THRESHOLD = _default_value(
    ("signal", "probability_threshold"),
    None,
)
SIGNAL_PROBABILITY_THRESHOLD = _env_optional_float(
    "QTS_SIGNAL_PROBABILITY_THRESHOLD",
    None
    if _DEFAULT_SIGNAL_PROBABILITY_THRESHOLD is None
    else float(_DEFAULT_SIGNAL_PROBABILITY_THRESHOLD),
)
SIGNAL_MIN_PROBABILITY_THRESHOLD = _env_optional_float(
    "QTS_SIGNAL_MIN_PROBABILITY_THRESHOLD",
    None
    if _default_value(("signal", "min_probability_threshold"), None) is None
    else float(_default_value(("signal", "min_probability_threshold"), None)),
)
SIGNAL_FALLBACK_MIN_POSITIONS = _env_int(
    "QTS_SIGNAL_FALLBACK_MIN_POSITIONS",
    int(_default_value(("signal", "fallback_min_positions"), 0)),
)
SIGNAL_RANK_COL = str(_default_value(("signal", "rank_col"), "class_4_prob"))
BENCHMARK_REGIME_ENABLED = _env_bool(
    "QTS_BENCHMARK_REGIME_ENABLED",
    bool(_default_value(("benchmark_regime", "enabled"), False)),
)
BENCHMARK_REGIME_RULE = _env_str(
    "QTS_BENCHMARK_REGIME_RULE",
    str(_default_value(("benchmark_regime", "rule"), "none")),
).strip().lower()
BENCHMARK_REGIME_WINDOW = _env_optional_int(
    "QTS_BENCHMARK_REGIME_WINDOW",
    _default_value(("benchmark_regime", "window"), None),
)
BENCHMARK_REGIME_MIN_PERIODS = _env_optional_int(
    "QTS_BENCHMARK_REGIME_MIN_PERIODS",
    _default_value(("benchmark_regime", "min_periods"), None),
)
baseline_features_cols = [
    str(value)
    for value in _default_value(
        ("features", "baseline_columns"),
        [
            "close_macd_histogram_50_200_30",
            "close_qsmom_21_252_126",
            "close_roc_0_63",
            "close_roc_0_252",
            "close_rsi_63",
            "close_fip_momentum_252",
            "close_volatility_annualized_252",
        ],
    )
]
BASELINE_FEATURE_COLS = baseline_features_cols
FEATURE_GROUP_COLUMNS = {
    "baseline": BASELINE_FEATURE_COLS,
    "log": [
        "log_price",
        "log_volume",
    ],
    "hist_vol": [
        "hist_vol_21",
        "hist_vol_42",
    ],
    "zscore": [
        "zscore_10",
        "zscore_21",
        "zscore_42",
    ],
    "volume_ratio": [
        "vol_ratio_5",
        "vol_ratio_12",
    ],
    "adx": [
        "adx_14",
        "adx_21",
    ],
    "roc_short": [
        "roc_14",
        "roc_21",
    ],
    "rsi_short": [
        "rsi_14",
        "rsi_21",
    ],
    "ma_short": [
        "ma_5",
        "ma_10",
        "ma_20",
    ],
}
FEATURE_GROUP_ORDER = [
    "baseline",
    "log",
    "hist_vol",
    "zscore",
    "volume_ratio",
    "adx",
    "roc_short",
    "rsi_short",
    "ma_short",
]


def _env_feature_groups() -> list[str]:
    default_groups = _default_value(("features", "selected_groups"), ["baseline"])
    default_raw = ",".join(str(item) for item in default_groups)
    raw = os.getenv("QTS_FEATURE_GROUPS", default_raw)
    groups = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not groups:
        groups = ["baseline"]
    unknown = [group for group in groups if group not in FEATURE_GROUP_COLUMNS]
    if unknown:
        allowed = ", ".join(FEATURE_GROUP_ORDER)
        raise ValueError(f"Unknown QTS_FEATURE_GROUPS values {unknown}; allowed: {allowed}")
    if "baseline" not in groups:
        groups.insert(0, "baseline")
    return list(dict.fromkeys(groups))


SELECTED_FEATURE_GROUPS = _env_feature_groups()
FEATURE_COLS = [
    column
    for group in SELECTED_FEATURE_GROUPS
    for column in FEATURE_GROUP_COLUMNS[group]
]


def experiment_id_from_feature_groups(groups: list[str]) -> str:
    additions = [group for group in groups if group != "baseline"]
    return "baseline" if not additions else "+".join(additions)


EXPERIMENT_ID = experiment_id_from_feature_groups(SELECTED_FEATURE_GROUPS)


class XGBClassifierPredictor:
    """Small XGBoost wrapper for five-class rolling-quantile classification."""

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        learning_rate: float = 0.1,
        max_depth: int | None = None,
        subsample: float | None = None,
        colsample_bytree: float | None = None,
        random_state: int = XGB_RANDOM_STATE,
        n_jobs: int = -1,
        objective: str = XGB_OBJECTIVE,
        eval_metric: str = XGB_EVAL_METRIC,
        tree_method: str = XGB_TREE_METHOD,
    ) -> None:
        self.params = {
            "objective": objective,
            "eval_metric": eval_metric,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "tree_method": tree_method,
        }
        if max_depth is not None:
            self.params["max_depth"] = max_depth
        if subsample is not None:
            self.params["subsample"] = subsample
        if colsample_bytree is not None:
            self.params["colsample_bytree"] = colsample_bytree
        self.model: xgb.XGBClassifier | None = None
        self.classes_: np.ndarray | None = None
        self.constant_probabilities_: np.ndarray | None = None
        self.thresholds_: QuantileClassThresholds | None = None

    def fit(
        self,
        train_frame: pd.DataFrame,
        predictor_cols: list[str],
        target_col: str,
        *,
        threshold_frame: pd.DataFrame | None = None,
    ) -> bool:
        train = train_frame.sort_values(["date", "symbol"]).reset_index(drop=True).copy()
        target = pd.to_numeric(train[target_col], errors="coerce")
        train = train.loc[np.isfinite(target)].copy()
        if train.empty or train["date"].nunique() < 2:
            self.model = None
            self.classes_ = None
            self.constant_probabilities_ = None
            self.thresholds_ = None
            return False

        threshold_source = train if threshold_frame is None else threshold_frame
        threshold_target = pd.to_numeric(threshold_source[target_col], errors="coerce")
        threshold_source = threshold_source.loc[np.isfinite(threshold_target)].copy()
        if threshold_source.empty:
            self.model = None
            self.classes_ = None
            self.constant_probabilities_ = None
            self.thresholds_ = None
            return False

        self.thresholds_ = QuantileClassThresholds.fit(
            threshold_source[target_col],
            target_col=target_col,
            percentiles=CLASS_QUANTILES,
        )
        labels = self.thresholds_.transform(train[target_col])
        labelled = labels >= 0
        train = train.loc[labelled].copy()
        labels = labels[labelled]
        if train.empty:
            self.model = None
            self.classes_ = None
            self.constant_probabilities_ = None
            self.thresholds_ = None
            return False

        self.classes_ = np.array(sorted(np.unique(labels)), dtype=int)
        self.constant_probabilities_ = None
        if self.classes_.size == 1:
            probabilities = np.zeros(CLASS_COUNT, dtype=float)
            probabilities[int(self.classes_[0])] = 1.0
            self.model = None
            self.constant_probabilities_ = probabilities
            return True

        encoded_labels = np.searchsorted(self.classes_, labels)
        params = dict(self.params)
        params["num_class"] = int(self.classes_.size)
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(train[predictor_cols], encoded_labels)
        return True

    def predict_probabilities(
        self,
        predict_frame: pd.DataFrame,
        predictor_cols: list[str],
    ) -> np.ndarray:
        if self.constant_probabilities_ is not None:
            return np.tile(self.constant_probabilities_, (len(predict_frame), 1))
        if self.model is None:
            raise RuntimeError("XGBClassifierPredictor.fit must succeed before predict.")
        probabilities = np.asarray(
            self.model.predict_proba(predict_frame[predictor_cols]),
            dtype=float,
        )
        if self.classes_ is None:
            raise RuntimeError("XGBClassifierPredictor.fit did not record class labels.")
        return align_class_probabilities(
            probabilities,
            self.classes_,
            num_classes=CLASS_COUNT,
        )

    def predict_scores(
        self,
        predict_frame: pd.DataFrame,
        predictor_cols: list[str],
    ) -> np.ndarray:
        probabilities = self.predict_probabilities(predict_frame, predictor_cols)
        return class_scores_from_probabilities(probabilities, CLASS_SCORES)

    def save_artifacts(
        self,
        artifact_dir: Path,
        stem: str,
        *,
        metadata: dict[str, Any],
    ) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            **metadata,
            "xgb_params": self.params,
            "classes": self.classes_.tolist() if self.classes_ is not None else [],
            "class_quantiles": list(CLASS_QUANTILES),
            "class_scores": CLASS_SCORES.tolist(),
            "thresholds": list(self.thresholds_.thresholds) if self.thresholds_ else [],
            "constant_probabilities": self.constant_probabilities_.tolist()
            if self.constant_probabilities_ is not None
            else None,
            "model_path": None,
        }
        if self.model is not None:
            model_path = artifact_dir / f"{stem}.json"
            self.model.save_model(str(model_path))
            payload["model_path"] = str(model_path)
        metadata_path = artifact_dir / f"{stem}.metadata.json"
        metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_rebalance_dates(
    frame: pd.DataFrame,
    rebalance_frequency: int | str,
) -> list[pd.Timestamp]:
    from qts.research.backtest._runner import _rebalance_dates

    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"]).drop_duplicates()).sort_values()
    date_values = [item.date() for item in dates]
    return [pd.Timestamp(item) for item in _rebalance_dates(date_values, rebalance_frequency)]


def add_fip_momentum(
    df: pl.DataFrame,
    *,
    window: int = 252,
    close_column: str = "close",
    output_column: str = "close_fip_momentum_252",
) -> pl.DataFrame:
    result = (
        df.sort(["symbol", "date"])
        .with_columns(
            ((pl.col(close_column) / pl.col(close_column).shift(1).over("symbol")) - 1).alias(
                "_daily_return"
            ),
            ((pl.col(close_column) / pl.col(close_column).shift(window).over("symbol")) - 1).alias(
                "_total_return"
            ),
        )
        .with_columns(
            (pl.col("_daily_return") > 0)
            .cast(pl.Float64)
            .rolling_sum(window, min_samples=window)
            .over("symbol")
            .alias("_positive_count"),
            (pl.col("_daily_return") < 0)
            .cast(pl.Float64)
            .rolling_sum(window, min_samples=window)
            .over("symbol")
            .alias("_negative_count"),
        )
        .with_columns(
            (
                pl.col("_total_return")
                * ((pl.col("_negative_count") - pl.col("_positive_count")) / float(window))
            ).alias(output_column)
        )
        .drop(["_daily_return", "_total_return", "_positive_count", "_negative_count"])
    )
    return result


def add_annualized_volatility(
    df: pl.DataFrame,
    *,
    window: int = 252,
    close_column: str = "close",
    output_column: str = "close_volatility_annualized_252",
) -> pl.DataFrame:
    result = (
        df.sort(["symbol", "date"])
        .with_columns(
            (pl.col(close_column).log() - pl.col(close_column).shift(1).over("symbol").log()).alias(
                "_log_return"
            )
        )
        .with_columns(
            (
                pl.col("_log_return").rolling_std(window, min_samples=window).over("symbol")
                * math.sqrt(252)
            ).alias(output_column)
        )
        .drop("_log_return")
    )
    return result


def build_reference_feature_frame(raw: pl.DataFrame, *, forward_period: int) -> pl.DataFrame:
    result = preprocess_ohlcv(raw)
    result = average_price_transform(result, output_column="avg_price")
    result = log_volume_transform(result, volume_column="volume", output_column="log_volume")
    result = log_price_transform(result, price_column="avg_price", output_column="log_price")
    result = qsmom_transform(
        result,
        roc_fast_period=21,
        roc_slow_period=252,
        returns_period=126,
        close_column="close",
    )
    for feature in [
        MACDFeature(fast=50, slow=200, signal=30),
        ROCFeature(periods=[63, 252]),
        RSIFeature(periods=[63]),
        RSIFeature(periods=[14, 21], price_column="log_price"),
        ROCFeature(periods=[14, 21], price_column="log_price"),
        MovingAverageFeature(periods=[5, 10, 20], price_column="log_price"),
        ADXFeature(periods=[14, 21]),
        VolumeRatioFeature(windows=[5, 12], volume_column="log_volume"),
        HistVolFeature(windows=[21, 42], price_column="log_price"),
        ZScoreFeature(windows=[10, 21, 42]),
    ]:
        result = feature.fit_transform(result)

    result = add_fip_momentum(result)
    result = add_annualized_volatility(result)
    result = ForwardReturns(periods=[forward_period]).fit_transform(result)
    return result.with_columns(
        pl.col("macd_hist").alias("close_macd_histogram_50_200_30"),
        pl.col("roc_63").alias("close_roc_0_63"),
        pl.col("roc_252").alias("close_roc_0_252"),
        pl.col("rsi_63").alias("close_rsi_63"),
        pl.col(f"forward_return_{forward_period}").alias(f"Return_fwd_{forward_period}"),
    )


def apply_feature_coverage(frame: pd.DataFrame, predictor_cols: list[str]) -> pd.DataFrame:
    min_non_null = max(3, len(predictor_cols) // 3)
    coverage = frame[predictor_cols].notna().sum(axis=1) >= min_non_null
    return frame.loc[coverage].copy()


def walk_forward_classification(
    frame: pd.DataFrame,
    *,
    predictor: XGBClassifierPredictor,
    predictor_cols: list[str],
    target_col: str,
    train_window: int,
    classification_window: int,
    rebalance_frequency: int | str,
    target_horizon: int,
    prediction_start_date: pd.Timestamp | None = None,
    prediction_end_date: pd.Timestamp | None = None,
    test_intervals: list[DateInterval] | None = None,
    model_artifact_dir: Path | None = None,
) -> pd.DataFrame:
    data = frame.sort_values(["date", "symbol"]).copy()
    all_dates = list(pd.DatetimeIndex(pd.to_datetime(data["date"]).drop_duplicates()).sort_values())
    rebalance_dates = build_rebalance_dates(data, rebalance_frequency)
    active_intervals = list(test_intervals or [])

    if active_intervals:
        rebalance_dates = filter_dates_in_intervals(rebalance_dates, active_intervals)
    if prediction_start_date is not None:
        rebalance_dates = [d for d in rebalance_dates if d >= prediction_start_date]
    if prediction_end_date is not None:
        rebalance_dates = [d for d in rebalance_dates if d <= prediction_end_date]

    rows: list[pd.DataFrame] = []
    for rebalance_date in rebalance_dates:
        idx = all_dates.index(rebalance_date)
        latest_trainable_idx = idx - target_horizon
        if latest_trainable_idx < 0:
            continue

        trainable_dates = all_dates[: latest_trainable_idx + 1]
        if len(trainable_dates) < max(train_window, classification_window):
            continue

        if active_intervals:
            train_dates_list = select_training_dates(
                trainable_dates,
                before=rebalance_date,
                train_window=train_window,
                test_intervals=active_intervals,
            )
            threshold_dates_list = select_training_dates(
                trainable_dates,
                before=rebalance_date,
                train_window=classification_window,
                test_intervals=active_intervals,
            )
        else:
            train_dates_list = trainable_dates[-train_window:]
            threshold_dates_list = trainable_dates[-classification_window:]
        if (
            len(train_dates_list) < train_window
            or len(threshold_dates_list) < classification_window
        ):
            continue

        train_dates = set(train_dates_list)
        threshold_dates = set(threshold_dates_list)
        train_slice = data.loc[data["date"].isin(train_dates)].copy()
        train_slice = train_slice.loc[train_slice[target_col].notna()].copy()
        train_slice = apply_feature_coverage(train_slice, predictor_cols)
        threshold_slice = train_slice.loc[train_slice["date"].isin(threshold_dates)].copy()

        predict_slice = data.loc[data["date"] == rebalance_date].copy()
        predict_slice = apply_feature_coverage(predict_slice, predictor_cols)

        if train_slice.empty or threshold_slice.empty or predict_slice.empty:
            continue
        if not predictor.fit(
            train_slice,
            predictor_cols,
            target_col,
            threshold_frame=threshold_slice,
        ):
            continue
        if model_artifact_dir is not None:
            train_dates_index = pd.DatetimeIndex(
                pd.to_datetime(train_slice["date"]).drop_duplicates()
            ).sort_values()
            threshold_dates_index = pd.DatetimeIndex(
                pd.to_datetime(threshold_slice["date"]).drop_duplicates()
            ).sort_values()
            predictor.save_artifacts(
                model_artifact_dir,
                f"rebalance_{rebalance_date:%Y%m%d}",
                metadata={
                    "rebalance_date": rebalance_date.date().isoformat(),
                    "target_col": target_col,
                    "predictor_cols": predictor_cols,
                    "train_window": train_window,
                    "classification_window": classification_window,
                    "target_horizon": target_horizon,
                    "train_rows": int(len(train_slice)),
                    "threshold_rows": int(len(threshold_slice)),
                    "train_start": train_dates_index.min().date().isoformat(),
                    "train_end": train_dates_index.max().date().isoformat(),
                    "threshold_start": threshold_dates_index.min().date().isoformat(),
                    "threshold_end": threshold_dates_index.max().date().isoformat(),
                },
            )

        probabilities = predictor.predict_probabilities(predict_slice, predictor_cols)
        scores = predictor.predict_scores(predict_slice, predictor_cols)
        predicted_class = np.argmax(probabilities, axis=1).astype(int)
        realized_class = np.full(len(predict_slice), np.nan, dtype=float)
        if target_col in predict_slice and predictor.thresholds_ is not None:
            labels = predictor.thresholds_.transform(predict_slice[target_col])
            realized_class = np.where(labels >= 0, labels.astype(float), np.nan)

        out = pd.DataFrame(
            {
                "date": predict_slice["date"].to_numpy(),
                "symbol": predict_slice["symbol"].to_numpy(),
                "predicted_class": predicted_class,
                "realized_class": realized_class,
                "score": scores.astype(float),
                "realized_forward_return": predict_slice[target_col].to_numpy()
                if target_col in predict_slice
                else np.nan,
            }
        )
        for index, column in enumerate(CLASS_PROB_COLS):
            out[column] = probabilities[:, index].astype(float)
        out[SIGNAL_PROBABILITY_SUM_COL] = out[SIGNAL_PROBABILITY_COLS].sum(axis=1)
        rows.append(out)

    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "predicted_class",
                "realized_class",
                "score",
                "realized_forward_return",
                *CLASS_PROB_COLS,
                SIGNAL_PROBABILITY_SUM_COL,
            ]
        )

    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["date", SIGNAL_RANK_COL], ascending=[True, False])
        .reset_index(drop=True)
    )


def _with_signal_probability_sum(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if SIGNAL_PROBABILITY_SUM_COL not in result.columns:
        result[SIGNAL_PROBABILITY_SUM_COL] = result[SIGNAL_PROBABILITY_COLS].sum(axis=1)
    return result


def select_topk_signal_candidates(frame: pd.DataFrame, top_n: int) -> pd.DataFrame:
    candidates = _with_signal_probability_sum(frame)
    probability_threshold = (
        SIGNAL_MIN_PROBABILITY_THRESHOLD
        if SIGNAL_MIN_PROBABILITY_THRESHOLD is not None
        else SIGNAL_PROBABILITY_THRESHOLD
    )
    if probability_threshold is not None:
        filtered = candidates.loc[
            candidates[SIGNAL_PROBABILITY_SUM_COL] > probability_threshold
        ].copy()
        fallback_target = min(top_n, SIGNAL_FALLBACK_MIN_POSITIONS)
        if len(filtered) < fallback_target:
            fallback = (
                candidates.loc[~candidates["symbol"].isin(filtered["symbol"])]
                .sort_values([SIGNAL_RANK_COL, "symbol"], ascending=[False, True])
                .head(fallback_target - len(filtered))
            )
            filtered = pd.concat([filtered, fallback], ignore_index=True)
        candidates = filtered
    if candidates.empty:
        return candidates
    return (
        candidates.sort_values([SIGNAL_RANK_COL, "symbol"], ascending=[False, True])
        .head(top_n)
        .copy()
    )


def summarize_topk_predictions(predictions: pd.DataFrame, top_n: int) -> dict[str, float]:
    scored = predictions.loc[predictions["realized_forward_return"].notna()].copy()
    if scored.empty:
        return {
            "prediction_rows": 0.0,
            "prediction_dates": 0.0,
            "selected_rows": 0.0,
            "selected_dates": 0.0,
            "top_mean_forward_return": np.nan,
            "class_accuracy": np.nan,
        }

    selected_rows = [
        select_topk_signal_candidates(group, top_n)
        for _, group in scored.groupby("date", sort=True)
    ]
    selected = (
        pd.concat(selected_rows, ignore_index=True)
        if selected_rows
        else pd.DataFrame(columns=scored.columns)
    )
    labelled = scored.loc[scored["realized_class"].notna()].copy()
    class_accuracy = (
        float(
            (
                labelled["predicted_class"].astype(int)
                == labelled["realized_class"].astype(int)
            ).mean()
        )
        if not labelled.empty
        else np.nan
    )
    return {
        "prediction_rows": float(len(scored)),
        "prediction_dates": float(scored["date"].nunique()),
        "selected_rows": float(len(selected)),
        "selected_dates": float(selected["date"].nunique()) if not selected.empty else 0.0,
        "top_mean_forward_return": float(selected["realized_forward_return"].mean())
        if not selected.empty
        else np.nan,
        "class_accuracy": class_accuracy,
    }


class XGBClassificationTopKStrategy(BaseStrategy):
    """Emit long-only capped slot-weight signals from precomputed classifier scores."""

    def __init__(
        self,
        predictions: pd.DataFrame,
        sessions: list[date],
        *,
        top_n: int = MAX_POSITIONS,
    ) -> None:
        self.predictions = predictions.copy()
        self.sessions = sorted(pd.to_datetime(pd.Index(sessions)).date)
        self.top_n = int(top_n)

    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        if self.predictions.empty:
            return self.empty_signal_frame()

        frame = self.predictions.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        next_session_map = {
            current: next_session
            for current, next_session in zip(self.sessions[:-1], self.sessions[1:], strict=False)
        }

        rows: list[dict[str, object]] = []
        previous_symbols: set[str] = set()
        for prediction_date in sorted(frame["date"].drop_duplicates()):
            execution_date = next_session_map.get(prediction_date)
            if execution_date is None:
                continue

            selected = select_topk_signal_candidates(
                frame.loc[frame["date"] == prediction_date],
                self.top_n,
            )
            current_symbols = set(selected["symbol"].tolist())
            if not selected.empty:
                weight = 1.0 / float(self.top_n)
                for record in selected.itertuples(index=False):
                    rows.append(
                        {
                            "date": execution_date,
                            "symbol": record.symbol,
                            "signal": 1,
                            "weight": weight,
                        }
                    )

            for symbol in sorted(previous_symbols - current_symbols):
                rows.append(
                    {
                        "date": execution_date,
                        "symbol": symbol,
                        "signal": 0,
                        "weight": 0.0,
                    }
                )
            previous_symbols = current_symbols

        if not rows:
            return self.empty_signal_frame()
        signals = pl.from_pandas(pd.DataFrame(rows)).sort(["date", "symbol"])
        return self.validate_signal_frame(signals)


def config_symbols(config: BacktestConfig) -> list[str]:
    return list(
        dict.fromkeys(
            [
                *config.universe.stock,
                *config.universe.vn_stock,
                *config.universe.vn_warrant,
                *config.universe.vn_futures,
                *config.universe.crypto,
                *config.universe.crypto_futures,
            ]
        )
    )


def validate_vn_classification_config(config: BacktestConfig) -> None:
    if "vn_stock" not in config.asset_types:
        raise ValueError(f"Expected VN stock config, got asset_types={config.asset_types!r}")
    if config.benchmark not in EXPECTED_BENCHMARKS:
        raise ValueError(f"Expected VNINDEX benchmark, got {config.benchmark!r}")


def resolve_rebalance_frequency(config: BacktestConfig) -> int | str:
    override = os.getenv("QTS_REBALANCE_FREQUENCY")
    if override:
        try:
            return int(override)
        except ValueError:
            return override
    default_rebalance = _default_value(("run", "rebalance_frequency"), None)
    if default_rebalance is not None:
        try:
            return int(default_rebalance)
        except (TypeError, ValueError):
            return str(default_rebalance)
    return config.rebalance_frequency


async def load_benchmark_returns(
    data_manager: Any,
    config: BacktestConfig,
) -> pd.Series | None:
    if not config.benchmark:
        return None
    try:
        benchmark = await data_manager.get_ohlcv(
            symbols=[config.benchmark],
            start=config.start_date,
            end=config.end_date,
        )
    except Exception:
        warnings.warn(
            f"Could not fetch benchmark {config.benchmark}; continuing without it.",
            stacklevel=2,
        )
        return None

    if benchmark.is_empty():
        return None
    frame = benchmark.select(["date", "close"]).sort("date").to_pandas()
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    returns = frame.set_index("date")["close"].pct_change().dropna()
    returns.name = config.benchmark
    return returns


async def load_benchmark_close(
    data_manager: Any,
    config: BacktestConfig,
) -> pd.DataFrame | None:
    if not config.benchmark:
        return None
    try:
        benchmark = await data_manager.get_ohlcv(
            symbols=[config.benchmark],
            start=config.start_date,
            end=config.end_date,
        )
    except Exception:
        warnings.warn(
            f"Could not fetch benchmark {config.benchmark}; regime filter disabled.",
            stacklevel=2,
        )
        return None

    if benchmark.is_empty():
        return None
    frame = benchmark.select(["date", "close"]).sort("date").to_pandas()
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.tz_convert(None)
    return frame.drop_duplicates("date").reset_index(drop=True)


def apply_benchmark_regime_filter(
    predictions: pd.DataFrame,
    benchmark_close: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary: dict[str, Any] = {
        "enabled": BENCHMARK_REGIME_ENABLED,
        "rule": BENCHMARK_REGIME_RULE if BENCHMARK_REGIME_ENABLED else "disabled",
        "window": BENCHMARK_REGIME_WINDOW,
        "min_periods": BENCHMARK_REGIME_MIN_PERIODS,
        "prediction_dates_before": (
            int(predictions["date"].nunique()) if not predictions.empty else 0
        ),
        "prediction_dates_after": (
            int(predictions["date"].nunique()) if not predictions.empty else 0
        ),
        "prediction_rows_before": int(len(predictions)),
        "prediction_rows_after": int(len(predictions)),
    }
    if (
        not BENCHMARK_REGIME_ENABLED
        or predictions.empty
        or benchmark_close is None
        or benchmark_close.empty
    ):
        return predictions, summary
    if BENCHMARK_REGIME_RULE != "above_sma":
        raise ValueError(f"Unsupported benchmark_regime.rule={BENCHMARK_REGIME_RULE!r}")
    if BENCHMARK_REGIME_WINDOW is None or BENCHMARK_REGIME_WINDOW <= 0:
        raise ValueError("benchmark_regime.window must be a positive integer")

    window = int(BENCHMARK_REGIME_WINDOW)
    min_periods = (
        int(BENCHMARK_REGIME_MIN_PERIODS)
        if BENCHMARK_REGIME_MIN_PERIODS is not None
        else max(3, window // 3)
    )
    summary["min_periods"] = min_periods

    benchmark = benchmark_close.copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    benchmark = benchmark.drop_duplicates("date").sort_values("date").set_index("date")
    benchmark["sma"] = benchmark["close"].rolling(window, min_periods=min_periods).mean()

    prediction_dates = pd.DatetimeIndex(
        pd.to_datetime(predictions["date"]).drop_duplicates()
    ).sort_values()
    regime = benchmark.reindex(prediction_dates, method="ffill")
    allowed_dates = prediction_dates[(regime["close"] > regime["sma"]).fillna(False).to_numpy()]
    allowed_date_values = set(pd.to_datetime(allowed_dates).date)

    filtered = predictions.copy()
    filtered["date"] = pd.to_datetime(filtered["date"])
    filtered = filtered.loc[filtered["date"].dt.date.isin(allowed_date_values)].copy()
    summary.update(
        {
            "rule": "above_sma",
            "window": window,
            "prediction_dates_after": int(filtered["date"].nunique()),
            "prediction_rows_after": int(len(filtered)),
        }
    )
    return filtered.reset_index(drop=True), summary


def prepare_dataset(
    featured: pl.DataFrame,
    *,
    target_col: str,
    predictor_cols: list[str],
) -> pd.DataFrame:
    featured_pd = featured.to_pandas().copy()
    featured_pd["date"] = pd.to_datetime(featured_pd["date"])
    featured_pd = featured_pd.sort_values(["date", "symbol"]).reset_index(drop=True)

    missing_features = [col for col in predictor_cols if col not in featured_pd.columns]
    if missing_features:
        raise KeyError(f"Missing expected feature columns: {missing_features}")
    if target_col not in featured_pd.columns:
        raise KeyError(f"Missing expected target column: {target_col}")

    dataset_cols = ["date", "symbol", target_col, *predictor_cols]
    dataset = featured_pd[dataset_cols].copy()
    dataset[predictor_cols] = dataset[predictor_cols].replace([np.inf, -np.inf], np.nan)
    return dataset.reset_index(drop=True)


def build_effective_params(
    *,
    run_id: str,
    rebalance_frequency: int | str,
    model_weights_path: Path,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "default_params_path": str(DEFAULT_PARAMS_PATH),
        "model_weights_path": str(model_weights_path),
        "report_pdf_dir": str(REPORT_PDF_DIR),
        "run": {
            "forward_period": REFERENCE_FORWARD_PERIOD,
            "classification_window": CLASSIFICATION_WINDOW,
            "train_window": REFERENCE_TRAIN_WINDOW,
            "rebalance_frequency": rebalance_frequency,
            "max_positions": MAX_POSITIONS,
        },
        "features": {
            "selected_groups": SELECTED_FEATURE_GROUPS,
            "predictor_cols": FEATURE_COLS,
        },
        "classification": {
            "class_quantiles": list(CLASS_QUANTILES),
            "class_scores": CLASS_SCORES.tolist(),
        },
        "signal": {
            "probability_cols": SIGNAL_PROBABILITY_COLS,
            "probability_sum_col": SIGNAL_PROBABILITY_SUM_COL,
            "probability_threshold": SIGNAL_PROBABILITY_THRESHOLD,
            "min_probability_threshold": SIGNAL_MIN_PROBABILITY_THRESHOLD,
            "fallback_min_positions": SIGNAL_FALLBACK_MIN_POSITIONS,
            "rank_col": SIGNAL_RANK_COL,
        },
        "xgb": {
            "n_estimators": XGB_N_ESTIMATORS,
            "learning_rate": XGB_LEARNING_RATE,
            "max_depth": XGB_MAX_DEPTH,
            "n_jobs": XGB_N_JOBS,
            "random_state": XGB_RANDOM_STATE,
            "objective": XGB_OBJECTIVE,
            "eval_metric": XGB_EVAL_METRIC,
            "tree_method": XGB_TREE_METHOD,
        },
        "benchmark_regime": {
            "enabled": BENCHMARK_REGIME_ENABLED,
            "rule": BENCHMARK_REGIME_RULE,
            "window": BENCHMARK_REGIME_WINDOW,
            "min_periods": BENCHMARK_REGIME_MIN_PERIODS,
        },
    }


def save_effective_params(run_id: str, payload: dict[str, Any]) -> Path:
    EFFECTIVE_YAML_DIR.mkdir(parents=True, exist_ok=True)
    path = EFFECTIVE_YAML_DIR / f"{run_id}_effective_params.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def export_outputs(
    *,
    run_id: str,
    predictions: pd.DataFrame,
    result,
    benchmark_rets: pd.Series | None,
    test_intervals: list[DateInterval],
) -> Path | None:
    out_dir = backtest_exports_dir()
    predictions.to_csv(out_dir / f"{run_id}_classifier_predictions.csv", index=False)
    result.signals.write_csv(out_dir / f"{run_id}_signals.csv")
    result.returns.write_csv(out_dir / f"{run_id}_returns.csv")
    result.equity_curve.write_csv(out_dir / f"{run_id}_equity_curve.csv")
    export_trade_log(result, out_dir / f"{run_id}_trade_log.csv")
    export_portfolio_snapshots(result, out_dir / f"{run_id}_portfolio_snapshots.csv")
    return export_backtest_report_pdf(
        result,
        run_id=run_id,
        output_dir=REPORT_PDF_DIR,
        benchmark_rets=benchmark_rets,
        tearsheet_out_dir=tearsheet_dir(),
        filename=report_pdf_filename(run_id),
        test_intervals=test_intervals,
    )


def report_pdf_filename(run_id: str) -> str:
    parts = run_id.rsplit("_", 2)
    if len(parts) == 3:
        return f"stock_ml_e2e_classification_report_{parts[1]}_{parts[2]}.pdf"
    return f"{run_id}_report.pdf"


def vanilla_report_pdf_filename(run_id: str) -> str:
    parts = run_id.rsplit("_", 2)
    if len(parts) == 3:
        return (
            "stock_ml_e2e_classification_vanilla_comparison_report_"
            f"{parts[1]}_{parts[2]}.pdf"
        )
    return f"{run_id}_vanilla_comparison_report.pdf"


def _format_report_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_report_value(value) for value in row) + " |")
    return "\n".join(lines)


def _held_portfolio_snapshots(signals: pl.DataFrame) -> pd.DataFrame:
    if signals.is_empty():
        return pd.DataFrame(columns=["date", "positions", "holdings"])

    signal_pd = signals.sort(["date", "symbol"]).to_pandas()
    signal_pd["date"] = pd.to_datetime(signal_pd["date"])
    current: dict[str, float] = {}
    rows: list[dict[str, object]] = []

    for signal_date, group in signal_pd.groupby("date", sort=True):
        for record in group.itertuples(index=False):
            symbol = str(record.symbol)
            signal = int(record.signal)
            weight = float(record.weight)
            if signal == 0 or weight == 0.0:
                current.pop(symbol, None)
            else:
                current[symbol] = signal * weight

        long_positions = sorted(
            (symbol, weight) for symbol, weight in current.items() if weight > 0
        )
        holdings = ", ".join(
            f"{symbol.replace('VN:', '')} {weight:.0%}" for symbol, weight in long_positions
        )
        rows.append(
            {
                "date": pd.Timestamp(signal_date).date(),
                "positions": len(long_positions),
                "holdings": holdings,
            }
        )

    return pd.DataFrame(rows)


def _filter_active_frame(frame: pd.DataFrame, intervals: list[DateInterval]) -> pd.DataFrame:
    if not intervals or frame.empty:
        return frame.copy()
    active = frame.copy()
    active["date"] = pd.to_datetime(active["date"])
    mask = active["date"].map(lambda value: date_in_intervals(value, intervals))
    return active.loc[mask].copy()


def _active_metrics(result: Any, active_intervals: list[DateInterval]) -> dict[str, float]:
    if not active_intervals:
        return {}

    returns = result.returns.to_pandas().copy()
    equity = result.equity_curve.to_pandas().copy()
    if returns.empty or equity.empty:
        return {}

    returns["date"] = pd.to_datetime(returns["date"])
    equity["date"] = pd.to_datetime(equity["date"])
    active_returns = returns.loc[
        returns["date"].map(lambda value: date_in_intervals(value, active_intervals)),
        "portfolio_return",
    ].tolist()
    active_equity = equity.loc[
        equity["date"].map(lambda value: date_in_intervals(value, active_intervals)),
        "equity",
    ].tolist()
    if not active_returns or len(active_equity) < 2:
        return {}
    return build_metrics(active_returns, active_equity)


def _most_frequent_holdings(snapshots: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for holdings in snapshots.get("holdings", pd.Series(dtype=str)).fillna(""):
        for item in str(holdings).split(", "):
            if not item:
                continue
            symbol = item.split()[0]
            counts[symbol] = counts.get(symbol, 0) + 1
    return pd.DataFrame(
        sorted(counts.items(), key=lambda item: item[1], reverse=True)[:top_n],
        columns=["symbol", "rebalance_count"],
    )


def build_runtime_report(
    *,
    run_id: str,
    result: Any,
    config: BacktestConfig,
    rebalance_frequency: int | str,
    eval_summary: dict[str, float],
    model_weights_path: Path,
    effective_params_path: Path,
    regime_summary: dict[str, Any],
) -> str:
    active_intervals = test_intervals_from_config(config)
    active_label = format_date_intervals(active_intervals)
    snapshots = _held_portfolio_snapshots(result.signals)
    active_snapshots = _filter_active_frame(snapshots, active_intervals)

    recent = active_snapshots.tail(20)
    frequent = _most_frequent_holdings(active_snapshots)
    active_stats = _active_metrics(result, active_intervals)

    lines = [
        f"# stock_ml_e2e runtime report: {run_id}",
        "",
        "## Strategy",
        _markdown_table(
            ["Item", "Value"],
            [
                ["benchmark", config.benchmark],
                ["task", "classification"],
                ["default_params_path", DEFAULT_PARAMS_PATH],
                ["effective_params_path", effective_params_path],
                ["model_weights_path", model_weights_path],
                ["classes", CLASS_COUNT],
                ["class_quantiles", ",".join(str(value) for value in CLASS_QUANTILES)],
                ["classification_window", CLASSIFICATION_WINDOW],
                ["forward_period", REFERENCE_FORWARD_PERIOD],
                ["train_window", REFERENCE_TRAIN_WINDOW],
                ["test_intervals", active_label],
                ["rebalance", rebalance_frequency],
                ["top_positions", MAX_POSITIONS],
                ["signal_probability_cols", "+".join(SIGNAL_PROBABILITY_COLS)],
                ["signal_probability_threshold", SIGNAL_PROBABILITY_THRESHOLD],
                ["signal_min_probability_threshold", SIGNAL_MIN_PROBABILITY_THRESHOLD],
                ["signal_fallback_min_positions", SIGNAL_FALLBACK_MIN_POSITIONS],
                ["signal_rank_col", SIGNAL_RANK_COL],
                ["experiment_id", EXPERIMENT_ID],
                ["feature_groups", ",".join(SELECTED_FEATURE_GROUPS)],
                ["predictor_count", len(FEATURE_COLS)],
                ["xgb_n_estimators", XGB_N_ESTIMATORS],
                ["xgb_learning_rate", XGB_LEARNING_RATE],
                ["xgb_max_depth", XGB_MAX_DEPTH],
                ["benchmark_regime_enabled", regime_summary.get("enabled")],
                ["benchmark_regime_rule", regime_summary.get("rule")],
                ["benchmark_regime_window", regime_summary.get("window")],
                ["benchmark_regime_min_periods", regime_summary.get("min_periods")],
                [
                    "benchmark_regime_prediction_dates",
                    (
                        f"{regime_summary.get('prediction_dates_after')} / "
                        f"{regime_summary.get('prediction_dates_before')}"
                    ),
                ],
            ],
        ),
        "",
        "## Predictor columns",
        _markdown_table(["Predictor"], [[column] for column in FEATURE_COLS]),
        "",
        "## Full-period stats",
        _markdown_table(
            ["Metric", "Value"],
            [[key, value] for key, value in result.metrics.items()],
        ),
        "",
        f"## Active-period stats for {active_label}",
        _markdown_table(["Metric", "Value"], [[key, value] for key, value in active_stats.items()]),
        "",
        "## Classifier evaluation",
        _markdown_table(["Metric", "Value"], [[key, value] for key, value in eval_summary.items()]),
        "",
        "## Active-period portfolio snapshots",
        _markdown_table(
            ["Date", "Positions", "Holdings after rebalance"],
            recent[["date", "positions", "holdings"]].values.tolist(),
        ),
        "",
        "## Most frequent active-period holdings",
        _markdown_table(
            ["Symbol", "Rebalance count"],
            frequent.values.tolist(),
        ),
    ]
    return "\n".join(lines)


def save_runtime_report(run_id: str, report: str) -> Path:
    RUNTIME_LOG_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RUNTIME_LOG_DIR / f"{run_id}_report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def _metric_value(metrics: dict[str, float], key: str) -> float:
    value = metrics.get(key, np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _read_ablation_results() -> pd.DataFrame:
    if not ABLATION_RESULTS_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(ABLATION_RESULTS_PATH, sep="\t")
    except pd.errors.ParserError:
        return pd.read_csv(
            ABLATION_RESULTS_PATH,
            sep="\t",
            engine="python",
            on_bad_lines="skip",
        )


def append_ablation_result(
    *,
    run_id: str,
    result: Any,
    config: BacktestConfig,
    rebalance_frequency: int | str,
    eval_summary: dict[str, float],
    runtime_log_path: Path,
    runtime_report_path: Path,
    model_weights_path: Path,
    effective_params_path: Path,
    regime_summary: dict[str, Any],
) -> Path:
    RUNTIME_LOG_DIR.mkdir(parents=True, exist_ok=True)
    active_intervals = test_intervals_from_config(config)
    active_stats = _active_metrics(result, active_intervals)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "experiment_id": EXPERIMENT_ID,
        "feature_groups": ",".join(SELECTED_FEATURE_GROUPS),
        "predictor_count": len(FEATURE_COLS),
        "predictors": ",".join(FEATURE_COLS),
        "benchmark": config.benchmark,
        "task": "classification",
        "default_params_path": str(DEFAULT_PARAMS_PATH),
        "effective_params_path": str(effective_params_path),
        "model_weights_path": str(model_weights_path),
        "class_count": CLASS_COUNT,
        "class_quantiles": ",".join(str(value) for value in CLASS_QUANTILES),
        "classification_window": CLASSIFICATION_WINDOW,
        "forward_period": REFERENCE_FORWARD_PERIOD,
        "train_window": REFERENCE_TRAIN_WINDOW,
        "test_intervals": format_date_intervals(active_intervals),
        "rebalance": rebalance_frequency,
        "top_positions": MAX_POSITIONS,
        "signal_probability_cols": "+".join(SIGNAL_PROBABILITY_COLS),
        "signal_probability_threshold": SIGNAL_PROBABILITY_THRESHOLD,
        "signal_min_probability_threshold": SIGNAL_MIN_PROBABILITY_THRESHOLD,
        "signal_fallback_min_positions": SIGNAL_FALLBACK_MIN_POSITIONS,
        "signal_rank_col": SIGNAL_RANK_COL,
        "xgb_n_estimators": XGB_N_ESTIMATORS,
        "xgb_learning_rate": XGB_LEARNING_RATE,
        "xgb_max_depth": XGB_MAX_DEPTH,
        "full_sharpe": _metric_value(result.metrics, "sharpe"),
        "full_sortino": _metric_value(result.metrics, "sortino"),
        "full_cagr": _metric_value(result.metrics, "cagr"),
        "full_max_drawdown": _metric_value(result.metrics, "max_drawdown"),
        "full_win_rate": _metric_value(result.metrics, "win_rate"),
        "active_sharpe": _metric_value(active_stats, "sharpe"),
        "active_sortino": _metric_value(active_stats, "sortino"),
        "active_cagr": _metric_value(active_stats, "cagr"),
        "active_max_drawdown": _metric_value(active_stats, "max_drawdown"),
        "active_win_rate": _metric_value(active_stats, "win_rate"),
        "prediction_rows": _metric_value(eval_summary, "prediction_rows"),
        "prediction_dates": _metric_value(eval_summary, "prediction_dates"),
        "selected_rows": _metric_value(eval_summary, "selected_rows"),
        "selected_dates": _metric_value(eval_summary, "selected_dates"),
        "top_mean_forward_return": _metric_value(eval_summary, "top_mean_forward_return"),
        "class_accuracy": _metric_value(eval_summary, "class_accuracy"),
        "benchmark_regime_enabled": regime_summary.get("enabled"),
        "benchmark_regime_rule": regime_summary.get("rule"),
        "benchmark_regime_window": regime_summary.get("window"),
        "benchmark_regime_min_periods": regime_summary.get("min_periods"),
        "regime_prediction_dates_before": regime_summary.get("prediction_dates_before"),
        "regime_prediction_dates_after": regime_summary.get("prediction_dates_after"),
        "regime_prediction_rows_before": regime_summary.get("prediction_rows_before"),
        "regime_prediction_rows_after": regime_summary.get("prediction_rows_after"),
        "runtime_log_path": str(runtime_log_path),
        "runtime_report_path": str(runtime_report_path),
    }
    existing = _read_ablation_results()
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True, sort=False)
    updated.to_csv(
        ABLATION_RESULTS_PATH,
        sep="\t",
        index=False,
    )
    return ABLATION_RESULTS_PATH


def _latest_ablation_rows() -> pd.DataFrame:
    rows = _read_ablation_results()
    if rows.empty:
        return rows
    rows["timestamp_sort"] = pd.to_datetime(rows["timestamp"], errors="coerce")
    rows = (
        rows.sort_values(["experiment_id", "timestamp_sort", "run_id"])
        .groupby("experiment_id", as_index=False)
        .tail(1)
        .copy()
    )
    return rows.drop(columns=["timestamp_sort"])


def _ranking_decision(row: pd.Series, baseline: pd.Series | None) -> str:
    full_mdd = float(row["full_max_drawdown"])
    active_mdd = float(row["active_max_drawdown"])
    if full_mdd > 0.35 or active_mdd > 0.35:
        return "drop: drawdown > 0.35"
    if row["experiment_id"] == "baseline":
        return "baseline"
    if baseline is None:
        return "pending: no baseline"

    beats_sharpe = (
        float(row["full_sharpe"]) > float(baseline["full_sharpe"])
        or float(row["active_sharpe"]) > float(baseline["active_sharpe"])
    )
    drawdown_ok = (
        full_mdd <= float(baseline["full_max_drawdown"]) + 0.03
        and active_mdd <= float(baseline["active_max_drawdown"]) + 0.03
    )
    return "keep" if beats_sharpe and drawdown_ok else "drop"


def _rank_ablation_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows

    numeric_cols = [
        "full_sharpe",
        "full_max_drawdown",
        "active_sharpe",
        "active_max_drawdown",
        "top_mean_forward_return",
    ]
    ranked = rows.copy()
    for col in numeric_cols:
        ranked[col] = pd.to_numeric(ranked[col], errors="coerce")

    baseline_rows = ranked.loc[ranked["experiment_id"] == "baseline"]
    baseline = baseline_rows.iloc[0] if not baseline_rows.empty else None
    ranked["decision"] = ranked.apply(_ranking_decision, axis=1, baseline=baseline)
    ranked["disqualified"] = (
        (ranked["full_max_drawdown"] > 0.35)
        | (ranked["active_max_drawdown"] > 0.35)
    )
    ranked = ranked.sort_values(
        [
            "disqualified",
            "active_sharpe",
            "full_sharpe",
            "active_max_drawdown",
            "full_max_drawdown",
            "top_mean_forward_return",
        ],
        ascending=[True, False, False, True, True, False],
    ).reset_index(drop=True)
    ranked.insert(0, "final_rank", range(1, len(ranked) + 1))
    return ranked


def write_ablation_ranking() -> Path:
    rows = _latest_ablation_rows()
    ranked = _rank_ablation_rows(rows)
    if ranked.empty:
        ABLATION_RANKING_PATH.write_text("# Ablation ranking\n\nNo runs yet.\n", encoding="utf-8")
        return ABLATION_RANKING_PATH

    table_rows = [
        [
            row.final_rank,
            row.experiment_id,
            row.feature_groups,
            row.full_sharpe,
            row.full_max_drawdown,
            row.active_sharpe,
            row.active_max_drawdown,
            row.top_mean_forward_return,
            row.decision,
            row.run_id,
        ]
        for row in ranked.itertuples(index=False)
    ]
    lines = [
        "# Ablation ranking",
        "",
        "Latest run per experiment from `ablation_results.tsv`.",
        "",
        _markdown_table(
            [
                "Final Rank",
                "Experiment",
                "Feature Groups",
                "Full Sharpe",
                "Full MDD",
                "Active Sharpe",
                "Active MDD",
                "Top Mean Fwd Return",
                "Decision",
                "Run ID",
            ],
            table_rows,
        ),
        "",
        "Decision rule: keep only if a group beats baseline on at least one Sharpe metric "
        "without increasing full or active max drawdown by more than 0.03; drawdown above "
        "0.35 is disqualified.",
    ]
    ABLATION_RANKING_PATH.write_text("\n".join(lines), encoding="utf-8")
    return ABLATION_RANKING_PATH


if __name__ == "__main__":
    runtime_log_path = enable_runtime_logging()
    print(f"Runtime log     : {runtime_log_path}")
    print(f"Imports OK | repo_root={repo_root}")
    print(f"Using config: {CONFIG_PATH}")

    resolved = Config.build(str(CONFIG_PATH))
    config: BacktestConfig = resolved.raw
    validate_vn_classification_config(config)
    rebalance_frequency = resolve_rebalance_frequency(config)
    safe_rebalance = str(rebalance_frequency).replace("/", "_")
    safe_experiment = EXPERIMENT_ID.replace("+", "_")
    run_id = (
        f"vn_reference_xgb_classifier_{safe_experiment}_top{MAX_POSITIONS}_"
        f"fwd{REFERENCE_FORWARD_PERIOD}_rb{safe_rebalance}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    model_weights_path = MODEL_WEIGHTS_DIR / run_id
    model_weights_path.mkdir(parents=True, exist_ok=True)
    effective_params_path = save_effective_params(
        run_id,
        build_effective_params(
            run_id=run_id,
            rebalance_frequency=rebalance_frequency,
            model_weights_path=model_weights_path,
        ),
    )
    print(f"Defaults YAML   : {DEFAULT_PARAMS_PATH}")
    print(f"Effective YAML  : {effective_params_path}")
    print(f"Model weights   : {model_weights_path}")

    symbols = config_symbols(config)
    data_manager = build_data_manager(resolved)
    raw: pl.DataFrame = asyncio.run(
        data_manager.get_ohlcv(
            symbols=symbols,
            start=config.start_date,
            end=config.end_date,
        )
    )
    benchmark_rets = asyncio.run(load_benchmark_returns(data_manager, config))
    benchmark_close = asyncio.run(load_benchmark_close(data_manager, config))

    featured: pl.DataFrame = build_reference_feature_frame(
        raw,
        forward_period=REFERENCE_FORWARD_PERIOD,
    )
    dataset_pd = prepare_dataset(
        featured,
        target_col=TARGET_COL,
        predictor_cols=FEATURE_COLS,
    )

    test_intervals = test_intervals_from_config(config)
    predictions = walk_forward_classification(
        frame=dataset_pd,
        predictor=XGBClassifierPredictor(
            n_estimators=XGB_N_ESTIMATORS,
            learning_rate=XGB_LEARNING_RATE,
            max_depth=XGB_MAX_DEPTH,
            n_jobs=XGB_N_JOBS,
        ),
        predictor_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        train_window=REFERENCE_TRAIN_WINDOW,
        classification_window=CLASSIFICATION_WINDOW,
        rebalance_frequency=rebalance_frequency,
        target_horizon=REFERENCE_FORWARD_PERIOD,
        test_intervals=test_intervals,
        model_artifact_dir=model_weights_path,
    )
    strategy_predictions, regime_summary = apply_benchmark_regime_filter(
        predictions,
        benchmark_close,
    )
    eval_summary = summarize_topk_predictions(strategy_predictions, MAX_POSITIONS)

    classifier_strategy = XGBClassificationTopKStrategy(
        predictions=strategy_predictions,
        sessions=raw["date"].unique().to_list(),
        top_n=MAX_POSITIONS,
    )
    entry_signals = classifier_strategy.generate_signals(raw)

    print(f"Benchmark       : {config.benchmark}")
    print(f"Task            : classification ({CLASS_COUNT} classes)")
    print(f"Class quantiles : {CLASS_QUANTILES}")
    print(f"Class window    : {CLASSIFICATION_WINDOW}")
    print(f"Forward period  : {REFERENCE_FORWARD_PERIOD}")
    print(f"Train window    : {REFERENCE_TRAIN_WINDOW}")
    print(f"Test intervals  : {format_date_intervals(test_intervals)}")
    print(f"Rebalance       : {rebalance_frequency}")
    print(f"Experiment ID   : {EXPERIMENT_ID}")
    print(f"Feature groups  : {','.join(SELECTED_FEATURE_GROUPS)}")
    print(f"Predictor count : {len(FEATURE_COLS)}")
    print(
        "Signal filter   : "
        f"{'+'.join(SIGNAL_PROBABILITY_COLS)} > {SIGNAL_PROBABILITY_THRESHOLD}, "
        f"min_threshold={SIGNAL_MIN_PROBABILITY_THRESHOLD}, "
        f"fallback_min_positions={SIGNAL_FALLBACK_MIN_POSITIONS}, "
        f"rank={SIGNAL_RANK_COL}"
    )
    print(
        "Benchmark regime: "
        f"enabled={regime_summary.get('enabled')}, "
        f"rule={regime_summary.get('rule')}, "
        f"window={regime_summary.get('window')}, "
        f"dates={regime_summary.get('prediction_dates_after')}/"
        f"{regime_summary.get('prediction_dates_before')}"
    )
    print(
        "XGB params      : "
        f"n_estimators={XGB_N_ESTIMATORS}, "
        f"learning_rate={XGB_LEARNING_RATE}, "
        f"max_depth={XGB_MAX_DEPTH}, "
        f"n_jobs={XGB_N_JOBS}"
    )
    print(f"Top positions   : {MAX_POSITIONS}")
    print(f"Prediction rows : {len(predictions):,}")
    print(f"Signal rows     : {entry_signals.height:,}")
    print(f"Signal dates    : {entry_signals['date'].n_unique() if entry_signals.height else 0}")
    print(f"Eval summary    : {eval_summary}")

    engine = resolved.engine
    print(f"Engine: {engine.__class__.__name__}")
    result = engine.run(
        strategy=classifier_strategy,
        data=raw,
        config=config,
    )

    report_pdf_path = export_outputs(
        run_id=run_id,
        predictions=strategy_predictions,
        result=result,
        benchmark_rets=benchmark_rets,
        test_intervals=test_intervals,
    )
    vanilla_artifacts = run_vanilla_backtest_from_signals(
        data=raw,
        signals=result.signals,
        config=config,
    )
    vanilla_export_paths = export_vanilla_backtest_artifacts(
        vanilla_artifacts,
        out_dir=backtest_exports_dir(),
        run_id=run_id,
    )
    vanilla_report_pdf_path = save_vanilla_comparison_report_pdf(
        vectorbt_result=result,
        vanilla_artifacts=vanilla_artifacts,
        run_id=run_id,
        output_dir=REPORT_PDF_DIR,
        vectorbt_report_path=report_pdf_path,
        exported_paths=vanilla_export_paths,
        filename=vanilla_report_pdf_filename(run_id),
        test_intervals=test_intervals,
    )
    runtime_report = build_runtime_report(
        run_id=run_id,
        result=result,
        config=config,
        rebalance_frequency=rebalance_frequency,
        eval_summary=eval_summary,
        model_weights_path=model_weights_path,
        effective_params_path=effective_params_path,
        regime_summary=regime_summary,
    )
    runtime_report_path = save_runtime_report(run_id, runtime_report)
    ablation_results_path = append_ablation_result(
        run_id=run_id,
        result=result,
        config=config,
        rebalance_frequency=rebalance_frequency,
        eval_summary=eval_summary,
        runtime_log_path=runtime_log_path,
        runtime_report_path=runtime_report_path,
        model_weights_path=model_weights_path,
        effective_params_path=effective_params_path,
        regime_summary=regime_summary,
    )
    ablation_ranking_path = write_ablation_ranking()
    print(f"Metrics         : {result.metrics}")
    print(f"Exports         : {backtest_exports_dir()}")
    print(f"Effective YAML  : {effective_params_path}")
    print(f"Model weights   : {model_weights_path}")
    print(f"Runtime report  : {runtime_report_path}")
    print(f"PDF report      : {report_pdf_path}")
    print(f"Vanilla PDF     : {vanilla_report_pdf_path}")
    print(f"Vanilla portfolio CSV: {vanilla_export_paths['portfolio_log']}")
    print(f"Vanilla entry CSV    : {vanilla_export_paths['entry_log']}")
    print(f"Ablation results: {ablation_results_path}")
    print(f"Ablation ranking: {ablation_ranking_path}")
    print("\n" + runtime_report)
