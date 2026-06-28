"""Train two VN ML classifiers, blend held target weights, and report all runs.

The data/backtest configuration comes from ``vn100_ml_example.yaml``. The two
classifier YAMLs define model-specific choices such as horizon, classifier
window, rebalance frequency, signal filter, top-k, and XGBoost params.
"""
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import asyncio
import importlib
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
import yaml
from matplotlib.backends.backend_pdf import PdfPages

repo_root = Path.cwd().resolve()
while not (repo_root / "qts").exists() and repo_root != repo_root.parent:
    repo_root = repo_root.parent

if not (repo_root / "qts").exists():
    raise FileNotFoundError("Could not find the repository root containing the qts package.")

repo_root_str = str(repo_root)
notebook_dir_str = str(repo_root / "notebooks")
for path in (repo_root_str, notebook_dir_str):
    if path not in sys.path:
        sys.path.insert(0, path)

warnings.filterwarnings("ignore")

import qts as qts_lib

importlib.reload(qts_lib)

from qts.config.builder import Config
from qts.orchestration.runtime import build_data_manager
from qts.research.backtest.base import BacktestConfig, BacktestResult
from qts.research.backtest.engines._targets import build_target_schedule
from qts.research.backtest.tearsheet import save_tearsheet, shade_test_intervals_on_axis
from qts.research.backtest.vanilla import (
    export_vanilla_backtest_artifacts,
    run_vanilla_backtest_from_signals,
    save_vanilla_comparison_report_pdf,
)
from qts.research.strategies.base import BaseStrategy
from qts.utils.classification import (
    QuantileClassThresholds,
    align_class_probabilities,
    class_scores_from_probabilities,
)
from qts.utils.date_intervals import (
    DateInterval,
    filter_dates_in_intervals,
    format_date_intervals,
    select_training_dates,
    test_intervals_from_config,
)
from qts.utils.export import export_portfolio_snapshots, export_trade_log
from qts.utils.model_cache import (
    frame_fingerprint,
    read_pickle_model_cache,
    write_pickle_model_cache,
)
from qts.utils.paths import backtest_exports_dir, tearsheet_dir

from stock_ml_e2e_classification import (
    FEATURE_GROUP_COLUMNS,
    apply_feature_coverage,
    build_rebalance_dates,
    build_reference_feature_frame,
    config_symbols,
    load_benchmark_returns,
    prepare_dataset,
    validate_vn_classification_config,
)


BASE_CONFIG_PATH = repo_root / "configs" / "strategies" / "ml_factor" / "vn100_ml_example.yaml"
DEFAULT_MODEL_CONFIGS = [
    repo_root / "configs" / "strategies" / "ml_factor" / "vn100_ml_classification.yaml",
    repo_root / "configs" / "strategies" / "ml_factor" / "vn100_ml_classification2.yaml",
]
DEFAULT_PARAMS_PATH = (
    repo_root / "configs" / "notebooks" / "stock_ml_e2e_classification" / "default_params.yaml"
)
RUNTIME_DIR = repo_root / ".qts_notebooke_runtime"
DEFAULT_REPORT_DIR = repo_root / "output" / "pdf"
DEFAULT_BASELINE_COLUMNS = [
    "close_macd_histogram_50_200_30",
    "close_qsmom_21_252_126",
    "close_roc_0_63",
    "close_roc_0_252",
    "close_rsi_63",
    "close_fip_momentum_252",
    "close_volatility_annualized_252",
]
DEFAULT_CLASS_QUANTILES = [0.15, 0.45, 0.6, 0.85]
DEFAULT_CLASS_SCORES = [-2.0, -1.0, 0.0, 1.0, 2.0]


@dataclass(frozen=True, slots=True)
class SignalSpec:
    """Model-specific conversion from classifier probabilities to top-k signals."""

    probability_cols: list[str]
    probability_sum_col: str
    probability_threshold: float | None
    min_probability_threshold: float | None
    fallback_min_positions: int
    rank_col: str


@dataclass(frozen=True, slots=True)
class ClassificationSpec:
    """Resolved model-specific classifier settings."""

    name: str
    path: Path
    raw: dict[str, Any]
    defaults: dict[str, Any]
    forward_period: int
    classification_window: int
    train_window: int
    rebalance_frequency: int | str
    max_positions: int
    predictor_cols: list[str]
    class_quantiles: tuple[float, ...]
    class_scores: np.ndarray
    signal: SignalSpec
    xgb_params: dict[str, Any]
    benchmark_regime: dict[str, Any]

    @property
    def class_count(self) -> int:
        return int(len(self.class_scores))

    @property
    def class_probability_cols(self) -> list[str]:
        return [f"class_{label}_prob" for label in range(self.class_count)]


@dataclass(frozen=True, slots=True)
class OutputDirs:
    yaml_dir: Path
    model_weights_dir: Path
    target_weights_dir: Path
    report_dir: Path


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """Outputs from one separate strategy or the blended ensemble."""

    label: str
    run_id: str
    result: BacktestResult
    predictions: pd.DataFrame | None
    signals: pl.DataFrame
    target_weights: pd.DataFrame
    model_weights_path: Path | None
    eval_summary: dict[str, float]
    output_paths: dict[str, Path | None]
    model_cache_summary: dict[str, Any] = field(default_factory=dict)


class PrecomputedSignalStrategy(BaseStrategy):
    """Strategy adapter that returns a prepared standard signal frame."""

    def __init__(self, signals: pl.DataFrame) -> None:
        self.signals = self.validate_signal_frame(signals) if not signals.is_empty() else signals

    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        return self.signals


class ConfigurableXGBClassifierPredictor:
    """Small XGBoost wrapper for per-spec rolling-quantile classification."""

    def __init__(
        self,
        *,
        class_quantiles: tuple[float, ...],
        class_scores: np.ndarray,
        n_estimators: int = 200,
        learning_rate: float = 0.1,
        max_depth: int | None = None,
        subsample: float | None = None,
        colsample_bytree: float | None = None,
        random_state: int = 123,
        n_jobs: int = -1,
        objective: str = "multi:softprob",
        eval_metric: str = "mlogloss",
        tree_method: str = "hist",
    ) -> None:
        self.class_quantiles = class_quantiles
        self.class_scores = class_scores
        self.class_count = len(class_scores)
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
            self._clear_model()
            return False

        threshold_source = train if threshold_frame is None else threshold_frame
        threshold_target = pd.to_numeric(threshold_source[target_col], errors="coerce")
        threshold_source = threshold_source.loc[np.isfinite(threshold_target)].copy()
        if threshold_source.empty:
            self._clear_model()
            return False

        self.thresholds_ = QuantileClassThresholds.fit(
            threshold_source[target_col],
            target_col=target_col,
            percentiles=self.class_quantiles,
        )
        labels = self.thresholds_.transform(train[target_col])
        labelled = labels >= 0
        train = train.loc[labelled].copy()
        labels = labels[labelled]
        if train.empty:
            self._clear_model()
            return False

        self.classes_ = np.array(sorted(np.unique(labels)), dtype=int)
        self.constant_probabilities_ = None
        if self.classes_.size == 1:
            probabilities = np.zeros(self.class_count, dtype=float)
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
            raise RuntimeError("ConfigurableXGBClassifierPredictor.fit must succeed first.")
        probabilities = np.asarray(
            self.model.predict_proba(predict_frame[predictor_cols]),
            dtype=float,
        )
        if self.classes_ is None:
            raise RuntimeError("ConfigurableXGBClassifierPredictor.fit did not record classes.")
        return align_class_probabilities(
            probabilities,
            self.classes_,
            num_classes=self.class_count,
        )

    def predict_scores(
        self,
        predict_frame: pd.DataFrame,
        predictor_cols: list[str],
    ) -> np.ndarray:
        return class_scores_from_probabilities(
            self.predict_probabilities(predict_frame, predictor_cols),
            self.class_scores,
        )

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
            "class_quantiles": list(self.class_quantiles),
            "class_scores": self.class_scores.tolist(),
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
        pickle_path = write_pickle_model_cache(
            artifact_dir,
            stem,
            payload={
                "model": self.model,
                "params": self.params,
                "classes": self.classes_,
                "constant_probabilities": self.constant_probabilities_,
                "thresholds": self.thresholds_,
            },
            metadata={
                **payload,
                "model_kind": "xgb_classifier",
            },
        )
        payload["pickle_path"] = str(pickle_path)

    def load_artifacts(
        self,
        artifact_dir: Path,
        stem: str,
        *,
        expected_metadata: dict[str, Any],
    ) -> bool:
        cached = read_pickle_model_cache(
            artifact_dir,
            stem,
            expected_metadata={
                **expected_metadata,
                "xgb_params": self.params,
                "class_quantiles": list(self.class_quantiles),
                "class_scores": self.class_scores.tolist(),
                "model_kind": "xgb_classifier",
            },
        )
        if cached is None:
            return False
        payload, _ = cached
        params = payload.get("params")
        if isinstance(params, dict):
            self.params = params
        self.model = payload.get("model")
        self.classes_ = payload.get("classes")
        self.constant_probabilities_ = payload.get("constant_probabilities")
        self.thresholds_ = payload.get("thresholds")
        if self.model is None and self.constant_probabilities_ is None:
            self._clear_model()
            return False
        return True

    def _clear_model(self) -> None:
        self.model = None
        self.classes_ = None
        self.constant_probabilities_ = None
        self.thresholds_ = None


class ClassificationTopKStrategy(BaseStrategy):
    """Emit long-only slot-weight signals from precomputed classifier probabilities."""

    def __init__(
        self,
        predictions: pd.DataFrame,
        sessions: list[object],
        *,
        spec: ClassificationSpec,
    ) -> None:
        self.predictions = predictions.copy()
        self.sessions = sorted(pd.to_datetime(pd.Index(sessions)).date)
        self.spec = spec

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
                self.spec,
            )
            current_symbols = set(selected["symbol"].tolist())
            if not selected.empty:
                weight = 1.0 / float(self.spec.max_positions)
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
        return self.validate_signal_frame(
            pl.from_pandas(pd.DataFrame(rows)).sort(["date", "symbol"])
        )


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML file: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return payload


def nested_value(payload: dict[str, Any], path: tuple[str, ...], fallback: Any) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return fallback
        current = current[key]
    return current


def spec_value(
    payload: dict[str, Any],
    defaults: dict[str, Any],
    path: tuple[str, ...],
    fallback: Any,
) -> Any:
    value = nested_value(payload, path, None)
    if value is not None:
        return value
    return nested_value(defaults, path, fallback)


def resolve_path(raw: Any, fallback: Path) -> Path:
    if raw in (None, ""):
        return fallback
    path = Path(str(raw)).expanduser()
    return path if path.is_absolute() else repo_root / path


def optional_float(raw: Any) -> float | None:
    if raw in (None, "", "none", "None", "null", "NULL", "false", "False", "off"):
        return None
    return float(raw)


def int_or_string(raw: Any) -> int | str:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return str(raw)


def resolve_predictor_cols(payload: dict[str, Any], defaults: dict[str, Any]) -> list[str]:
    features = payload.get("features", {})
    default_features = defaults.get("features", {})
    if not isinstance(features, dict) or not isinstance(default_features, dict):
        raise ValueError("features must be a mapping")

    explicit_cols = features.get("predictor_cols")
    if explicit_cols:
        return [str(column) for column in explicit_cols]

    default_explicit_cols = default_features.get("predictor_cols")
    if default_explicit_cols:
        return [str(column) for column in default_explicit_cols]

    baseline_cols = [
        str(column)
        for column in features.get(
            "baseline_columns",
            default_features.get("baseline_columns", DEFAULT_BASELINE_COLUMNS),
        )
    ]
    selected_groups = [
        str(group).strip().lower()
        for group in features.get(
            "selected_groups",
            default_features.get("selected_groups", ["baseline"]),
        )
        if str(group).strip()
    ]
    if not selected_groups:
        selected_groups = ["baseline"]

    columns: list[str] = []
    for group in selected_groups:
        if group == "baseline":
            columns.extend(baseline_cols)
            continue
        if group not in FEATURE_GROUP_COLUMNS:
            allowed = ", ".join(sorted(FEATURE_GROUP_COLUMNS))
            raise ValueError(f"Unknown feature group {group!r}; allowed: {allowed}")
        columns.extend(str(column) for column in FEATURE_GROUP_COLUMNS[group])
    return list(dict.fromkeys(columns))


def resolve_signal_spec(payload: dict[str, Any], defaults: dict[str, Any]) -> SignalSpec:
    return SignalSpec(
        probability_cols=[
            str(value)
            for value in spec_value(
                payload,
                defaults,
                ("signal", "probability_cols"),
                ["class_3_prob", "class_4_prob"],
            )
        ],
        probability_sum_col=str(
            spec_value(payload, defaults, ("signal", "probability_sum_col"), "class_3_4_prob")
        ),
        probability_threshold=optional_float(
            spec_value(payload, defaults, ("signal", "probability_threshold"), None)
        ),
        min_probability_threshold=optional_float(
            spec_value(payload, defaults, ("signal", "min_probability_threshold"), None)
        ),
        fallback_min_positions=int(
            spec_value(payload, defaults, ("signal", "fallback_min_positions"), 0)
        ),
        rank_col=str(spec_value(payload, defaults, ("signal", "rank_col"), "class_4_prob")),
    )


def resolve_classification_spec(path: Path, index: int) -> ClassificationSpec:
    payload = load_yaml_mapping(path)
    default_path = resolve_path(payload.get("default_params_path"), DEFAULT_PARAMS_PATH)
    defaults = load_yaml_mapping(default_path)
    forward_period = int(spec_value(payload, defaults, ("run", "forward_period"), 21))
    classification_window = int(
        spec_value(payload, defaults, ("run", "classification_window"), 120)
    )
    train_window = int(spec_value(payload, defaults, ("run", "train_window"), 504))
    rebalance_frequency = int_or_string(
        spec_value(payload, defaults, ("run", "rebalance_frequency"), 10)
    )
    max_positions = int(spec_value(payload, defaults, ("run", "max_positions"), 4))
    if min(forward_period, classification_window, train_window, max_positions) <= 0:
        raise ValueError(f"{path}: run periods and max_positions must be positive")

    class_quantiles = tuple(
        float(value)
        for value in spec_value(
            payload,
            defaults,
            ("classification", "class_quantiles"),
            DEFAULT_CLASS_QUANTILES,
        )
    )
    class_scores = np.asarray(
        spec_value(payload, defaults, ("classification", "class_scores"), DEFAULT_CLASS_SCORES),
        dtype=float,
    )
    if len(class_quantiles) != len(class_scores) - 1:
        raise ValueError(f"{path}: class_quantiles must have len(class_scores) - 1 entries")

    xgb_params = dict(spec_value(payload, defaults, ("xgb",), {}))
    xgb_params.setdefault("n_estimators", 5)
    xgb_params.setdefault("learning_rate", 0.1)
    xgb_params.setdefault("max_depth", 3)
    xgb_params.setdefault("n_jobs", -1)
    xgb_params.setdefault("random_state", 123)
    xgb_params.setdefault("objective", "multi:softprob")
    xgb_params.setdefault("eval_metric", "mlogloss")
    xgb_params.setdefault("tree_method", "hist")

    name = path.stem
    return ClassificationSpec(
        name=name or f"model_{index}",
        path=path,
        raw=payload,
        defaults=defaults,
        forward_period=forward_period,
        classification_window=classification_window,
        train_window=train_window,
        rebalance_frequency=rebalance_frequency,
        max_positions=max_positions,
        predictor_cols=resolve_predictor_cols(payload, defaults),
        class_quantiles=class_quantiles,
        class_scores=class_scores,
        signal=resolve_signal_spec(payload, defaults),
        xgb_params=xgb_params,
        benchmark_regime=dict(payload.get("benchmark_regime", {})),
    )


def resolve_output_dirs(specs: list[ClassificationSpec]) -> OutputDirs:
    first = specs[0] if specs else None
    raw = first.raw if first else {}
    defaults = first.defaults if first else load_yaml_mapping(DEFAULT_PARAMS_PATH)
    yaml_dir = resolve_path(
        spec_value(raw, defaults, ("paths", "effective_yaml_dir"), None),
        RUNTIME_DIR / "yaml" / "stock_ml_e2e_classification",
    )
    model_weights_dir = resolve_path(
        spec_value(raw, defaults, ("paths", "model_weights_dir"), None),
        RUNTIME_DIR / "model_weights" / "stock_ml_e2e_classification",
    )
    target_weights_dir = resolve_path(
        spec_value(raw, defaults, ("paths", "target_weights_dir"), None),
        RUNTIME_DIR / "weights" / "stock_ml_e2e_classification",
    )
    report_dir = resolve_path(
        spec_value(raw, defaults, ("paths", "report_pdf_dir"), None),
        DEFAULT_REPORT_DIR,
    )
    for path in (yaml_dir, model_weights_dir, target_weights_dir, report_dir):
        path.mkdir(parents=True, exist_ok=True)
    return OutputDirs(
        yaml_dir=yaml_dir,
        model_weights_dir=model_weights_dir,
        target_weights_dir=target_weights_dir,
        report_dir=report_dir,
    )


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
    regime_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    enabled = bool(regime_cfg.get("enabled", False))
    rule = str(regime_cfg.get("rule", "none")).strip().lower()
    window = regime_cfg.get("window")
    min_periods_raw = regime_cfg.get("min_periods")
    summary: dict[str, Any] = {
        "enabled": enabled,
        "rule": rule if enabled else "disabled",
        "window": window,
        "min_periods": min_periods_raw,
        "prediction_dates_before": (
            int(predictions["date"].nunique()) if not predictions.empty else 0
        ),
        "prediction_dates_after": (
            int(predictions["date"].nunique()) if not predictions.empty else 0
        ),
        "prediction_rows_before": int(len(predictions)),
        "prediction_rows_after": int(len(predictions)),
    }
    if not enabled or predictions.empty or benchmark_close is None or benchmark_close.empty:
        return predictions, summary
    if rule != "above_sma":
        raise ValueError(f"Unsupported benchmark_regime.rule={rule!r}")
    if window is None or int(window) <= 0:
        raise ValueError("benchmark_regime.window must be a positive integer")

    window_int = int(window)
    min_periods = (
        int(min_periods_raw)
        if min_periods_raw is not None
        else max(3, window_int // 3)
    )
    summary["min_periods"] = min_periods

    benchmark = benchmark_close.copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    benchmark = benchmark.drop_duplicates("date").sort_values("date").set_index("date")
    benchmark["sma"] = benchmark["close"].rolling(window_int, min_periods=min_periods).mean()

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
            "window": window_int,
            "prediction_dates_after": int(filtered["date"].nunique()),
            "prediction_rows_after": int(len(filtered)),
        }
    )
    return filtered.reset_index(drop=True), summary


def walk_forward_classification_for_spec(
    frame: pd.DataFrame,
    *,
    predictor: ConfigurableXGBClassifierPredictor,
    spec: ClassificationSpec,
    target_col: str,
    test_intervals: list[DateInterval],
    model_artifact_dir: Path,
    use_model_cache: bool = False,
) -> pd.DataFrame:
    data = frame.sort_values(["date", "symbol"]).copy()
    all_dates = list(pd.DatetimeIndex(pd.to_datetime(data["date"]).drop_duplicates()).sort_values())
    rebalance_dates = build_rebalance_dates(data, spec.rebalance_frequency)
    cache_summary: dict[str, object] = {
        "enabled": bool(use_model_cache),
        "artifact_dir": str(model_artifact_dir),
        "attempts": 0,
        "hits": 0,
        "misses": 0,
        "saves": 0,
        "failed_fits": 0,
    }

    if test_intervals:
        rebalance_dates = filter_dates_in_intervals(rebalance_dates, test_intervals)

    rows: list[pd.DataFrame] = []
    for rebalance_date in rebalance_dates:
        idx = all_dates.index(rebalance_date)
        latest_trainable_idx = idx - spec.forward_period
        if latest_trainable_idx < 0:
            continue

        trainable_dates = all_dates[: latest_trainable_idx + 1]
        if len(trainable_dates) < max(spec.train_window, spec.classification_window):
            continue

        if test_intervals:
            train_dates_list = select_training_dates(
                trainable_dates,
                before=rebalance_date,
                train_window=spec.train_window,
                test_intervals=test_intervals,
            )
            threshold_dates_list = select_training_dates(
                trainable_dates,
                before=rebalance_date,
                train_window=spec.classification_window,
                test_intervals=test_intervals,
            )
        else:
            train_dates_list = trainable_dates[-spec.train_window :]
            threshold_dates_list = trainable_dates[-spec.classification_window :]
        if (
            len(train_dates_list) < spec.train_window
            or len(threshold_dates_list) < spec.classification_window
        ):
            continue

        train_dates = set(train_dates_list)
        threshold_dates = set(threshold_dates_list)
        train_slice = data.loc[data["date"].isin(train_dates)].copy()
        train_slice = train_slice.loc[train_slice[target_col].notna()].copy()
        train_slice = apply_feature_coverage(train_slice, spec.predictor_cols)
        threshold_slice = train_slice.loc[train_slice["date"].isin(threshold_dates)].copy()

        predict_slice = data.loc[data["date"] == rebalance_date].copy()
        predict_slice = apply_feature_coverage(predict_slice, spec.predictor_cols)

        if train_slice.empty or threshold_slice.empty or predict_slice.empty:
            continue

        stem = f"rebalance_{rebalance_date:%Y%m%d}"
        cache_metadata = build_model_cache_metadata(
            rebalance_date,
            target_col=target_col,
            spec=spec,
            train_slice=train_slice,
            threshold_slice=threshold_slice,
        )
        loaded_from_cache = use_model_cache and predictor.load_artifacts(
            model_artifact_dir,
            stem,
            expected_metadata=cache_metadata,
        )
        cache_summary["attempts"] = int(cache_summary["attempts"]) + 1
        if not loaded_from_cache:
            cache_summary["misses"] = int(cache_summary["misses"]) + 1
            if not predictor.fit(
                train_slice,
                spec.predictor_cols,
                target_col,
                threshold_frame=threshold_slice,
            ):
                cache_summary["failed_fits"] = int(cache_summary["failed_fits"]) + 1
                continue

            save_model_artifact(
                predictor,
                model_artifact_dir,
                rebalance_date,
                target_col=target_col,
                spec=spec,
                train_slice=train_slice,
                threshold_slice=threshold_slice,
            )
            cache_summary["saves"] = int(cache_summary["saves"]) + 1
        else:
            cache_summary["hits"] = int(cache_summary["hits"]) + 1

        probabilities = predictor.predict_probabilities(predict_slice, spec.predictor_cols)
        scores = predictor.predict_scores(predict_slice, spec.predictor_cols)
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
        for index, column in enumerate(spec.class_probability_cols):
            out[column] = probabilities[:, index].astype(float)
        out = with_signal_probability_sum(out, spec)
        rows.append(out)

    columns = [
        "date",
        "symbol",
        "predicted_class",
        "realized_class",
        "score",
        "realized_forward_return",
        *spec.class_probability_cols,
        spec.signal.probability_sum_col,
    ]
    if not rows:
        empty = pd.DataFrame(columns=columns)
        empty.attrs["model_cache_summary"] = cache_summary
        return empty
    result = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["date", spec.signal.rank_col], ascending=[True, False])
        .reset_index(drop=True)
    )
    result.attrs["model_cache_summary"] = cache_summary
    return result


def build_model_cache_metadata(
    rebalance_date: pd.Timestamp,
    *,
    target_col: str,
    spec: ClassificationSpec,
    train_slice: pd.DataFrame,
    threshold_slice: pd.DataFrame,
) -> dict[str, Any]:
    train_dates = pd.DatetimeIndex(
        pd.to_datetime(train_slice["date"]).drop_duplicates()
    ).sort_values()
    threshold_dates = pd.DatetimeIndex(
        pd.to_datetime(threshold_slice["date"]).drop_duplicates()
    ).sort_values()
    return {
        "rebalance_date": rebalance_date.date().isoformat(),
        "target_col": target_col,
        "predictor_cols": spec.predictor_cols,
        "train_window": spec.train_window,
        "classification_window": spec.classification_window,
        "target_horizon": spec.forward_period,
        "train_rows": int(len(train_slice)),
        "threshold_rows": int(len(threshold_slice)),
        "train_start": train_dates.min().date().isoformat(),
        "train_end": train_dates.max().date().isoformat(),
        "threshold_start": threshold_dates.min().date().isoformat(),
        "threshold_end": threshold_dates.max().date().isoformat(),
        "train_fingerprint": frame_fingerprint(
            train_slice,
            ["date", "symbol", target_col, *spec.predictor_cols],
        ),
        "threshold_fingerprint": frame_fingerprint(
            threshold_slice,
            ["date", "symbol", target_col, *spec.predictor_cols],
        ),
        "signal": {
            "probability_cols": spec.signal.probability_cols,
            "probability_sum_col": spec.signal.probability_sum_col,
            "probability_threshold": spec.signal.probability_threshold,
            "min_probability_threshold": spec.signal.min_probability_threshold,
            "fallback_min_positions": spec.signal.fallback_min_positions,
            "rank_col": spec.signal.rank_col,
        },
    }


def save_model_artifact(
    predictor: ConfigurableXGBClassifierPredictor,
    model_artifact_dir: Path,
    rebalance_date: pd.Timestamp,
    *,
    target_col: str,
    spec: ClassificationSpec,
    train_slice: pd.DataFrame,
    threshold_slice: pd.DataFrame,
) -> None:
    predictor.save_artifacts(
        model_artifact_dir,
        f"rebalance_{rebalance_date:%Y%m%d}",
        metadata=build_model_cache_metadata(
            rebalance_date,
            target_col=target_col,
            spec=spec,
            train_slice=train_slice,
            threshold_slice=threshold_slice,
        ),
    )


def with_signal_probability_sum(frame: pd.DataFrame, spec: ClassificationSpec) -> pd.DataFrame:
    result = frame.copy()
    if spec.signal.probability_sum_col not in result.columns:
        result[spec.signal.probability_sum_col] = result[spec.signal.probability_cols].sum(axis=1)
    return result


def select_topk_signal_candidates(frame: pd.DataFrame, spec: ClassificationSpec) -> pd.DataFrame:
    candidates = with_signal_probability_sum(frame, spec)
    probability_threshold = (
        spec.signal.min_probability_threshold
        if spec.signal.min_probability_threshold is not None
        else spec.signal.probability_threshold
    )
    if probability_threshold is not None:
        filtered = candidates.loc[
            candidates[spec.signal.probability_sum_col] > probability_threshold
        ].copy()
        fallback_target = min(spec.max_positions, spec.signal.fallback_min_positions)
        if len(filtered) < fallback_target:
            fallback = (
                candidates.loc[~candidates["symbol"].isin(filtered["symbol"])]
                .sort_values([spec.signal.rank_col, "symbol"], ascending=[False, True])
                .head(fallback_target - len(filtered))
            )
            filtered = pd.concat([filtered, fallback], ignore_index=True)
        candidates = filtered
    if candidates.empty:
        return candidates
    return (
        candidates.sort_values([spec.signal.rank_col, "symbol"], ascending=[False, True])
        .head(spec.max_positions)
        .copy()
    )


def summarize_topk_predictions(
    predictions: pd.DataFrame,
    spec: ClassificationSpec,
) -> dict[str, float]:
    scored = predictions.loc[predictions["realized_forward_return"].notna()].copy()
    if scored.empty:
        return {
            "prediction_rows": 0.0,
            "prediction_dates": 0.0,
            "selected_rows": 0.0,
            "selected_dates": 0.0,
            "top_mean_forward_return": float("nan"),
            "class_accuracy": float("nan"),
        }

    selected_rows = [
        select_topk_signal_candidates(group, spec)
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
        else float("nan")
    )
    return {
        "prediction_rows": float(len(scored)),
        "prediction_dates": float(scored["date"].nunique()),
        "selected_rows": float(len(selected)),
        "selected_dates": float(selected["date"].nunique()) if not selected.empty else 0.0,
        "top_mean_forward_return": float(selected["realized_forward_return"].mean())
        if not selected.empty
        else float("nan"),
        "class_accuracy": class_accuracy,
    }


def session_index(raw: pl.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(raw["date"].unique().to_list())).sort_values()


def target_weights_from_signals(
    signals: pl.DataFrame,
    *,
    sessions: pd.DatetimeIndex,
    symbols: list[str],
) -> pd.DataFrame:
    return build_target_schedule(
        signals,
        sessions,
        symbols,
        shift_by_one_bar=False,
    ).targets


def target_matrix_to_signals(targets: pd.DataFrame) -> pl.DataFrame:
    if targets.empty:
        return BaseStrategy.empty_signal_frame()

    targets = targets.sort_index().sort_index(axis=1).fillna(0.0)
    previous = targets.shift(1).fillna(0.0)
    changed = ~np.isclose(
        targets.to_numpy(dtype=float),
        previous.to_numpy(dtype=float),
        atol=1e-12,
        rtol=0.0,
    )
    events = targets.where(pd.DataFrame(changed, index=targets.index, columns=targets.columns))

    rows: list[dict[str, object]] = []
    for timestamp, row in events.iterrows():
        for symbol, target in row.dropna().items():
            weight = abs(float(target))
            rows.append(
                {
                    "date": pd.Timestamp(timestamp).date(),
                    "symbol": str(symbol),
                    "signal": (
                        0
                        if np.isclose(weight, 0.0, atol=1e-12, rtol=0.0)
                        else int(np.sign(target))
                    ),
                    "weight": 0.0 if np.isclose(weight, 0.0, atol=1e-12, rtol=0.0) else weight,
                }
            )
    if not rows:
        return BaseStrategy.empty_signal_frame()
    return PrecomputedSignalStrategy(pl.from_pandas(pd.DataFrame(rows))).signals


def positive_target_events(signals: pl.DataFrame) -> pd.DataFrame:
    if signals.is_empty():
        return pd.DataFrame(columns=["date", "symbol", "weight"])
    return (
        signals.filter(pl.col("weight") > 0)
        .select(["date", "symbol", "weight"])
        .to_pandas()
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )


def held_weights_long_frame(targets: pd.DataFrame) -> pd.DataFrame:
    if targets.empty:
        return pd.DataFrame(columns=["date", "symbol", "weight"])
    long = (
        targets.rename_axis(index="date", columns="symbol")
        .stack()
        .rename("weight")
        .reset_index()
    )
    long = long.loc[~np.isclose(long["weight"].astype(float), 0.0, atol=1e-12, rtol=0.0)]
    long["date"] = pd.to_datetime(long["date"]).dt.date
    return long.sort_values(["date", "symbol"]).reset_index(drop=True)


def export_run_outputs(
    artifacts: RunArtifacts,
    *,
    raw: pl.DataFrame,
    config: BacktestConfig,
    benchmark_rets: pd.Series | None,
    test_intervals: list[DateInterval],
    output_dirs: OutputDirs,
) -> RunArtifacts:
    out_dir = backtest_exports_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_dirs.target_weights_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path | None] = {}
    if artifacts.predictions is not None:
        prediction_path = out_dir / f"{artifacts.run_id}_classifier_predictions.csv"
        artifacts.predictions.to_csv(prediction_path, index=False)
        paths["predictions"] = prediction_path

    signals_path = out_dir / f"{artifacts.run_id}_signals.csv"
    returns_path = out_dir / f"{artifacts.run_id}_returns.csv"
    equity_path = out_dir / f"{artifacts.run_id}_equity_curve.csv"
    trade_log_path = out_dir / f"{artifacts.run_id}_trade_log.csv"
    snapshots_path = out_dir / f"{artifacts.run_id}_portfolio_snapshots.csv"
    target_events_path = (
        output_dirs.target_weights_dir / f"{artifacts.run_id}_target_weight_events.csv"
    )
    held_weights_path = (
        output_dirs.target_weights_dir / f"{artifacts.run_id}_held_target_weights.csv"
    )

    artifacts.signals.write_csv(signals_path)
    artifacts.result.returns.write_csv(returns_path)
    artifacts.result.equity_curve.write_csv(equity_path)
    export_trade_log(artifacts.result, trade_log_path)
    export_portfolio_snapshots(artifacts.result, snapshots_path)
    positive_target_events(artifacts.signals).to_csv(target_events_path, index=False)
    held_weights_long_frame(artifacts.target_weights).to_csv(held_weights_path, index=False)

    paths.update(
        {
            "signals": signals_path,
            "returns": returns_path,
            "equity_curve": equity_path,
            "trade_log": trade_log_path,
            "portfolio_snapshots": snapshots_path,
            "target_weight_events": target_events_path,
            "held_target_weights": held_weights_path,
        }
    )
    if artifacts.model_weights_path is not None:
        paths["model_weights"] = artifacts.model_weights_path

    report_path = save_tearsheet(
        artifacts.result,
        tearsheet_dir(),
        artifacts.run_id,
        benchmark_rets=benchmark_rets,
        test_intervals=test_intervals,
    )
    paths["tearsheet_pdf"] = report_path

    vanilla_artifacts = run_vanilla_backtest_from_signals(
        data=raw,
        signals=artifacts.signals,
        config=config,
    )
    vanilla_export_paths = export_vanilla_backtest_artifacts(
        vanilla_artifacts,
        out_dir=out_dir,
        run_id=artifacts.run_id,
    )
    paths.update({f"vanilla_{key}": value for key, value in vanilla_export_paths.items()})
    paths["vanilla_pdf"] = save_vanilla_comparison_report_pdf(
        vectorbt_result=artifacts.result,
        vanilla_artifacts=vanilla_artifacts,
        run_id=artifacts.run_id,
        output_dir=output_dirs.report_dir,
        vectorbt_report_path=report_path,
        exported_paths=vanilla_export_paths,
        filename=f"{artifacts.run_id}_vanilla_comparison_report.pdf",
        test_intervals=test_intervals,
    )

    return RunArtifacts(
        label=artifacts.label,
        run_id=artifacts.run_id,
        result=artifacts.result,
        predictions=artifacts.predictions,
        signals=artifacts.signals,
        target_weights=artifacts.target_weights,
        model_weights_path=artifacts.model_weights_path,
        eval_summary=artifacts.eval_summary,
        output_paths=paths,
        model_cache_summary=artifacts.model_cache_summary,
    )


def train_and_backtest_classifier(
    spec: ClassificationSpec,
    *,
    raw: pl.DataFrame,
    feature_cache: dict[int, pd.DataFrame],
    config: BacktestConfig,
    engine: Any,
    test_intervals: list[DateInterval],
    benchmark_close: pd.DataFrame | None,
    output_dirs: OutputDirs,
    run_stamp: str,
    use_model_cache: bool = True,
) -> RunArtifacts:
    target_col = f"Return_fwd_{spec.forward_period}"
    if spec.forward_period not in feature_cache:
        featured = build_reference_feature_frame(raw, forward_period=spec.forward_period)
        feature_cache[spec.forward_period] = featured.to_pandas()
    dataset = prepare_dataset(
        pl.from_pandas(feature_cache[spec.forward_period]),
        target_col=target_col,
        predictor_cols=spec.predictor_cols,
    )
    run_id = (
        f"vn_ml_classifier_{spec.name}_top{spec.max_positions}_"
        f"fwd{spec.forward_period}_rb{spec.rebalance_frequency}_{run_stamp}"
    ).replace("/", "_")
    model_cache_id = (
        f"{spec.name}_top{spec.max_positions}_fwd{spec.forward_period}_"
        f"rb{spec.rebalance_frequency}_tw{spec.train_window}_cw{spec.classification_window}"
    ).replace("/", "_")
    model_weights_path = output_dirs.model_weights_dir / model_cache_id
    model_weights_path.mkdir(parents=True, exist_ok=True)

    predictions = walk_forward_classification_for_spec(
        frame=dataset,
        predictor=ConfigurableXGBClassifierPredictor(
            class_quantiles=spec.class_quantiles,
            class_scores=spec.class_scores,
            **spec.xgb_params,
        ),
        spec=spec,
        target_col=target_col,
        test_intervals=test_intervals,
        model_artifact_dir=model_weights_path,
        use_model_cache=use_model_cache,
    )
    model_cache_summary = dict(predictions.attrs.get("model_cache_summary", {}))
    if model_cache_summary:
        print(
            f"Model cache {spec.name}: "
            f"hits={model_cache_summary.get('hits', 0)} "
            f"misses={model_cache_summary.get('misses', 0)} "
            f"saves={model_cache_summary.get('saves', 0)} "
            f"dir={model_cache_summary.get('artifact_dir')}",
            flush=True,
        )
    strategy_predictions, regime_summary = apply_benchmark_regime_filter(
        predictions,
        benchmark_close,
        spec.benchmark_regime,
    )
    strategy = ClassificationTopKStrategy(
        predictions=strategy_predictions,
        sessions=raw["date"].unique().to_list(),
        spec=spec,
    )
    signals = strategy.generate_signals(raw)
    result = engine.run(strategy=strategy, data=raw, config=config)
    targets = target_weights_from_signals(
        result.signals,
        sessions=session_index(raw),
        symbols=config_symbols(config),
    )
    return RunArtifacts(
        label=spec.name,
        run_id=run_id,
        result=result,
        predictions=strategy_predictions,
        signals=result.signals if not result.signals.is_empty() else signals,
        target_weights=targets,
        model_weights_path=model_weights_path,
        eval_summary={
            **summarize_topk_predictions(strategy_predictions, spec),
            "regime_prediction_dates_before": regime_summary["prediction_dates_before"],
            "regime_prediction_dates_after": regime_summary["prediction_dates_after"],
            "regime_prediction_rows_before": regime_summary["prediction_rows_before"],
            "regime_prediction_rows_after": regime_summary["prediction_rows_after"],
        },
        output_paths={},
        model_cache_summary=model_cache_summary,
    )


def build_ensemble_artifacts(
    separate_runs: list[RunArtifacts],
    *,
    weights: list[float],
    raw: pl.DataFrame,
    config: BacktestConfig,
    engine: Any,
    run_stamp: str,
) -> RunArtifacts:
    if len(separate_runs) != len(weights):
        raise ValueError("separate_runs and weights must have the same length")
    sessions = session_index(raw)
    symbols = config_symbols(config)
    blended = pd.DataFrame(0.0, index=sessions, columns=sorted(symbols))
    for run, sleeve_weight in zip(separate_runs, weights, strict=True):
        aligned = (
            run.target_weights.reindex(index=sessions, columns=blended.columns)
            .ffill()
            .fillna(0.0)
        )
        blended = blended.add(aligned * float(sleeve_weight), fill_value=0.0)
    blended = blended.clip(lower=-1.0, upper=1.0)
    signals = target_matrix_to_signals(blended)
    strategy = PrecomputedSignalStrategy(signals)
    result = engine.run(strategy=strategy, data=raw, config=config)
    prediction_rows = sum(
        0 if run.predictions is None else len(run.predictions) for run in separate_runs
    )
    prediction_dates: set[object] = set()
    for run in separate_runs:
        if run.predictions is not None and not run.predictions.empty:
            prediction_dates.update(pd.to_datetime(run.predictions["date"]).dt.date)
    return RunArtifacts(
        label="ensemble",
        run_id=f"vn_ml_classifier_weight_ensemble_{run_stamp}",
        result=result,
        predictions=None,
        signals=result.signals if not result.signals.is_empty() else signals,
        target_weights=blended,
        model_weights_path=None,
        eval_summary={
            "prediction_rows": float(prediction_rows),
            "prediction_dates": float(len(prediction_dates)),
            "selected_rows": float("nan"),
            "selected_dates": float("nan"),
            "top_mean_forward_return": float("nan"),
            "class_accuracy": float("nan"),
        },
        output_paths={},
        model_cache_summary={
            "source_runs": {
                run.label: run.model_cache_summary for run in separate_runs
            }
        },
    )


def metric_row(run: RunArtifacts) -> dict[str, Any]:
    row: dict[str, Any] = {
        "label": run.label,
        "run_id": run.run_id,
        "signal_rows": run.signals.height,
        "signal_dates": run.signals["date"].n_unique() if run.signals.height else 0,
        "held_weight_rows": len(run.target_weights),
    }
    row.update({f"metric_{key}": value for key, value in run.result.metrics.items()})
    row.update({f"eval_{key}": value for key, value in run.eval_summary.items()})
    for key in ["enabled", "attempts", "hits", "misses", "saves", "failed_fits"]:
        if key in run.model_cache_summary:
            row[f"cache_{key}"] = run.model_cache_summary[key]
    return row


def result_equity_series(run: RunArtifacts) -> pd.Series:
    equity = run.result.equity_curve.to_pandas()
    if equity.empty:
        return pd.Series(dtype=float, name=run.label)
    equity["date"] = pd.to_datetime(equity["date"])
    return equity.set_index("date")["equity"].astype(float).rename(run.label)


def save_combined_report_pdf(
    runs: list[RunArtifacts],
    *,
    report_dir: Path,
    run_stamp: str,
    test_intervals: list[DateInterval],
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"vn_ml_classifier_weight_ensemble_summary_{run_stamp}.pdf"
    metrics = pd.DataFrame([metric_row(run) for run in runs])
    equity = pd.concat([result_equity_series(run) for run in runs], axis=1).sort_index()

    plt.close("all")
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        fig.text(
            0.03,
            0.95,
            "VN ML Classifier Separate and Ensemble Report",
            fontsize=15,
            weight="bold",
        )
        fig.text(
            0.03,
            0.92,
            f"Test intervals: {format_date_intervals(test_intervals)}",
            fontsize=9,
        )
        display_cols = [
            "label",
            "metric_sharpe",
            "metric_cagr",
            "metric_max_drawdown",
            "metric_sortino",
            "metric_win_rate",
            "signal_dates",
            "eval_prediction_dates",
            "eval_class_accuracy",
        ]
        display = metrics[[column for column in display_cols if column in metrics.columns]].copy()
        for column in display.columns:
            if column.startswith("metric_") or column.startswith("eval_"):
                display[column] = pd.to_numeric(display[column], errors="coerce").round(4)
        table = ax.table(
            cellText=display.fillna("").astype(str).values,
            colLabels=display.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.35)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        if not equity.empty:
            fig, ax = plt.subplots(figsize=(11, 6.5))
            normalized = equity.divide(equity.ffill().bfill().iloc[0]).replace(
                [np.inf, -np.inf],
                np.nan,
            )
            normalized.plot(ax=ax, linewidth=1.4)
            ax.set_title("Normalized Equity Curves")
            ax.set_ylabel("Growth of 1.0")
            ax.grid(alpha=0.25)
            shade_test_intervals_on_axis(ax, test_intervals)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            fig.autofmt_xdate()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        pdf.infodict().update(
            {
                "Title": "VN ML Classifier Weight Ensemble Summary",
                "Author": "QTradeSystematic",
            }
        )
    plt.close("all")
    return path


def save_summary_files(
    runs: list[RunArtifacts],
    *,
    base_config_path: Path,
    specs: list[ClassificationSpec],
    ensemble_weights: list[float],
    yaml_dir: Path,
    run_stamp: str,
    combined_report_path: Path,
) -> tuple[Path, Path]:
    metrics = pd.DataFrame([metric_row(run) for run in runs])
    metrics_path = yaml_dir / f"vn_ml_classifier_weight_ensemble_metrics_{run_stamp}.csv"
    metrics.to_csv(metrics_path, index=False)

    payload = {
        "run_stamp": run_stamp,
        "base_config": str(base_config_path),
        "model_configs": [str(spec.path) for spec in specs],
        "ensemble": {
            "method": "held_target_weight_blend",
            "weights": {
                run.label: float(weight)
                for run, weight in zip(
                    [run for run in runs if run.label != "ensemble"],
                    ensemble_weights,
                    strict=True,
                )
            },
            "signal_filter": "sleeve-level",
        },
        "combined_report_pdf": str(combined_report_path),
        "runs": [
            {
                "label": run.label,
                "run_id": run.run_id,
                "metrics": run.result.metrics,
                "eval_summary": run.eval_summary,
                "model_weights_path": str(run.model_weights_path)
                if run.model_weights_path is not None
                else None,
                "output_paths": {
                    key: str(value) if value is not None else None
                    for key, value in run.output_paths.items()
                },
            }
            for run in runs
        ],
    }
    yaml_path = yaml_dir / f"vn_ml_classifier_weight_ensemble_effective_{run_stamp}.yaml"
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return metrics_path, yaml_path


def parse_ensemble_weights(raw: str, count: int) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) != count:
        raise ValueError(f"Expected {count} ensemble weights, got {len(values)}")
    total = sum(values)
    if total <= 0:
        raise ValueError("Ensemble weights must sum to a positive value")
    return [value / total for value in values]


def print_dry_run(
    base_config_path: Path,
    base_config: BacktestConfig,
    specs: list[ClassificationSpec],
    weights: list[float],
) -> None:
    print(f"Base config      : {base_config_path}")
    print(f"Universe symbols : {len(config_symbols(base_config))}")
    print(f"Benchmark        : {base_config.benchmark}")
    print(f"Validation       : {format_date_intervals(test_intervals_from_config(base_config))}")
    print("Model specs:")
    for spec, weight in zip(specs, weights, strict=True):
        threshold = (
            spec.signal.min_probability_threshold
            if spec.signal.min_probability_threshold is not None
            else spec.signal.probability_threshold
        )
        print(
            "  - "
            f"{spec.name}: fwd={spec.forward_period}, rb={spec.rebalance_frequency}, "
            f"class_window={spec.classification_window}, train_window={spec.train_window}, "
            f"top={spec.max_positions}, threshold={threshold}, "
            f"fallback={spec.signal.fallback_min_positions}, "
            f"regime={spec.benchmark_regime}, "
            f"predictors={len(spec.predictor_cols)}, ensemble_weight={weight:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train two VN ML classifiers separately and blend their held target weights."
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=BASE_CONFIG_PATH,
        help="Base QTS config for universe, data, validation, engine, and costs.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        action="append",
        default=None,
        help="Classifier parameter YAML. Pass twice to override the default two configs.",
    )
    parser.add_argument(
        "--ensemble-weights",
        default="0.5,0.5",
        help="Comma-separated sleeve weights. Values are normalized to sum to 1.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve configs and print planned runs without fetching data or training.",
    )
    parser.add_argument(
        "--no-model-cache",
        action="store_true",
        help="Force retraining even when matching pickle model weights already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_paths = args.model_config or DEFAULT_MODEL_CONFIGS
    specs = [
        resolve_classification_spec(path.resolve(), index)
        for index, path in enumerate(model_paths, 1)
    ]
    ensemble_weights = parse_ensemble_weights(args.ensemble_weights, len(specs))

    resolved = Config.build(args.base_config)
    config = resolved.raw
    validate_vn_classification_config(config)

    if args.dry_run:
        print_dry_run(args.base_config, config, specs, ensemble_weights)
        return

    output_dirs = resolve_output_dirs(specs)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_manager = build_data_manager(resolved)
    symbols = config_symbols(config)
    test_intervals = test_intervals_from_config(config)

    print(f"Base config      : {args.base_config}")
    print(f"Symbols          : {len(symbols)}")
    print(f"Benchmark        : {config.benchmark}")
    print(f"Validation       : {format_date_intervals(test_intervals)}")
    print(f"Run stamp        : {run_stamp}")

    raw: pl.DataFrame = asyncio.run(
        data_manager.get_ohlcv(
            symbols=symbols,
            start=config.start_date,
            end=config.end_date,
        )
    )
    benchmark_rets = asyncio.run(load_benchmark_returns(data_manager, config))
    benchmark_close = asyncio.run(load_benchmark_close(data_manager, config))

    feature_cache: dict[int, pd.DataFrame] = {}
    separate_runs: list[RunArtifacts] = []
    for spec in specs:
        print(
            f"Training {spec.name}: fwd={spec.forward_period}, "
            f"rb={spec.rebalance_frequency}, top={spec.max_positions}"
        )
        separate_runs.append(
            train_and_backtest_classifier(
                spec,
                raw=raw,
                feature_cache=feature_cache,
                config=config,
                engine=resolved.engine,
                test_intervals=test_intervals,
                benchmark_close=benchmark_close,
                output_dirs=output_dirs,
                run_stamp=run_stamp,
                use_model_cache=not args.no_model_cache,
            )
        )

    ensemble = build_ensemble_artifacts(
        separate_runs,
        weights=ensemble_weights,
        raw=raw,
        config=config,
        engine=resolved.engine,
        run_stamp=run_stamp,
    )
    all_runs = [*separate_runs, ensemble]
    exported_runs = [
        export_run_outputs(
            run,
            raw=raw,
            config=config,
            benchmark_rets=benchmark_rets,
            test_intervals=test_intervals,
            output_dirs=output_dirs,
        )
        for run in all_runs
    ]
    combined_report_path = save_combined_report_pdf(
        exported_runs,
        report_dir=output_dirs.report_dir,
        run_stamp=run_stamp,
        test_intervals=test_intervals,
    )
    metrics_path, effective_yaml_path = save_summary_files(
        exported_runs,
        base_config_path=args.base_config,
        specs=specs,
        ensemble_weights=ensemble_weights,
        yaml_dir=output_dirs.yaml_dir,
        run_stamp=run_stamp,
        combined_report_path=combined_report_path,
    )

    print("\nResults:")
    for run in exported_runs:
        print(f"  {run.label}: {run.result.metrics}")
        print(f"    vanilla PDF: {run.output_paths.get('vanilla_pdf')}")
        print(f"    target weights: {run.output_paths.get('held_target_weights')}")
        if run.model_weights_path is not None:
            print(f"    model weights: {run.model_weights_path}")
    print(f"Combined report : {combined_report_path}")
    print(f"Metrics CSV     : {metrics_path}")
    print(f"Effective YAML  : {effective_yaml_path}")


if __name__ == "__main__":
    main()
