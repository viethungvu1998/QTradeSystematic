"""Train two VN ML rankers, blend held target weights, and report all runs.

The data/backtest configuration comes from ``vn100_ml_example.yaml``. The two
ranker YAMLs only define model-specific choices such as horizon, rebalance
frequency, feature columns, XGBoost params, top-k, and benchmark regime filter.
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
from qts.utils.date_intervals import (
    DateInterval,
    format_date_intervals,
    test_intervals_from_config,
)
from qts.utils.export import export_portfolio_snapshots, export_trade_log
from qts.utils.paths import backtest_exports_dir, tearsheet_dir

from stock_ml_e2e import (
    FEATURE_GROUP_COLUMNS,
    XGBRankerPredictor,
    XGBRankerTopKStrategy,
    build_reference_feature_frame,
    config_symbols,
    load_benchmark_close_frame,
    load_benchmark_returns,
    prepare_dataset,
    validate_vn_ranker_config,
    walk_forward_rank,
)


BASE_CONFIG_PATH = repo_root / "configs" / "strategies" / "ml_factor" / "vn100_ml_example.yaml"
DEFAULT_MODEL_CONFIGS = [
    repo_root / "configs" / "strategies" / "ml_factor" / "vn100_ml_ranking.yaml",
    repo_root / "configs" / "strategies" / "ml_factor" / "vn100_ml_ranking2.yaml",
]
RUNTIME_DIR = repo_root / ".qts_notebooke_runtime" / "combined_ml_e2e_ranker"
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


@dataclass(frozen=True, slots=True)
class RankerSpec:
    """Resolved model-specific ranker settings."""

    name: str
    path: Path
    raw: dict[str, Any]
    forward_period: int
    train_window: int
    rebalance_frequency: int | str
    max_positions: int
    predictor_cols: list[str]
    xgb_params: dict[str, Any]
    min_top_score: float | None
    min_score_spread: float | None
    benchmark_regime: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """Outputs from one separate strategy or the blended ensemble."""

    label: str
    run_id: str
    result: BacktestResult
    raw_predictions: pd.DataFrame | None
    predictions: pd.DataFrame | None
    signals: pl.DataFrame
    target_weights: pd.DataFrame
    regime_summary: dict[str, Any]
    eval_summary: dict[str, float]
    output_paths: dict[str, Path | None]
    model_cache_summary: dict[str, Any] = field(default_factory=dict)


class PrecomputedSignalStrategy(BaseStrategy):
    """Strategy adapter that returns a prepared standard signal frame."""

    def __init__(self, signals: pl.DataFrame) -> None:
        self.signals = self.validate_signal_frame(signals) if not signals.is_empty() else signals

    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        return self.signals


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


def resolve_path(raw: Any, fallback: Path) -> Path:
    if raw in (None, ""):
        return fallback
    path = Path(str(raw)).expanduser()
    return path if path.is_absolute() else repo_root / path


def optional_float(raw: Any) -> float | None:
    if raw in (None, "", "none", "None", "null", "NULL"):
        return None
    return float(raw)


def int_or_string(raw: Any) -> int | str:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return str(raw)


def resolve_predictor_cols(payload: dict[str, Any]) -> list[str]:
    features = payload.get("features", {})
    if not isinstance(features, dict):
        raise ValueError("features must be a mapping")

    explicit_cols = features.get("predictor_cols")
    if explicit_cols:
        return [str(column) for column in explicit_cols]

    baseline_cols = [
        str(column)
        for column in features.get("baseline_columns", DEFAULT_BASELINE_COLUMNS)
    ]
    selected_groups = [
        str(group).strip().lower()
        for group in features.get("selected_groups", ["baseline"])
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


def resolve_ranker_spec(path: Path, index: int) -> RankerSpec:
    payload = load_yaml_mapping(path)
    forward_period = int(nested_value(payload, ("run", "forward_period"), 30))
    train_window = int(nested_value(payload, ("run", "train_window"), 630))
    rebalance_frequency = int_or_string(
        nested_value(payload, ("run", "rebalance_frequency"), 10)
    )
    max_positions = int(nested_value(payload, ("run", "max_positions"), 4))
    if forward_period <= 0:
        raise ValueError(f"{path}: run.forward_period must be positive")
    if train_window <= 0:
        raise ValueError(f"{path}: run.train_window must be positive")
    if max_positions <= 0:
        raise ValueError(f"{path}: run.max_positions must be positive")

    portfolio = payload.get("portfolio", {})
    if not isinstance(portfolio, dict):
        raise ValueError(f"{path}: portfolio must be a mapping")

    xgb_params = dict(payload.get("xgb", {}))
    xgb_params.setdefault("n_estimators", 5)
    xgb_params.setdefault("learning_rate", 0.1)
    xgb_params.setdefault("max_depth", 3)
    xgb_params.setdefault("n_jobs", -1)

    name = path.stem
    return RankerSpec(
        name=name or f"model_{index}",
        path=path,
        raw=payload,
        forward_period=forward_period,
        train_window=train_window,
        rebalance_frequency=rebalance_frequency,
        max_positions=max_positions,
        predictor_cols=resolve_predictor_cols(payload),
        xgb_params=xgb_params,
        min_top_score=optional_float(portfolio.get("min_top_score")),
        min_score_spread=optional_float(portfolio.get("min_score_spread")),
        benchmark_regime=dict(payload.get("benchmark_regime", {})),
    )


def resolve_output_dirs(specs: list[RankerSpec]) -> tuple[Path, Path, Path]:
    first_raw = specs[0].raw if specs else {}
    yaml_dir = resolve_path(
        nested_value(first_raw, ("paths", "effective_yaml_dir"), None),
        RUNTIME_DIR / "yaml",
    )
    weights_dir = resolve_path(
        nested_value(first_raw, ("paths", "portfolio_weights_dir"), None),
        RUNTIME_DIR / "weights",
    )
    report_dir = resolve_path(
        nested_value(first_raw, ("paths", "report_pdf_dir"), None),
        DEFAULT_REPORT_DIR,
    )
    yaml_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return yaml_dir, weights_dir, report_dir


def resolve_model_weights_dir(specs: list[RankerSpec]) -> Path:
    first_raw = specs[0].raw if specs else {}
    model_weights_dir = resolve_path(
        nested_value(first_raw, ("paths", "model_weights_dir"), None),
        RUNTIME_DIR / "model_weights",
    )
    model_weights_dir.mkdir(parents=True, exist_ok=True)
    return model_weights_dir


def apply_param_benchmark_regime_filter(
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


def summarize_topk_predictions(predictions: pd.DataFrame, top_n: int) -> dict[str, float]:
    scored = predictions.loc[predictions["realized_forward_return"].notna()].copy()
    if scored.empty:
        return {
            "prediction_rows": 0.0,
            "prediction_dates": 0.0,
            "top_mean_forward_return": float("nan"),
        }
    top = scored.sort_values(["date", "score"], ascending=[True, False]).groupby("date").head(top_n)
    return {
        "prediction_rows": float(len(scored)),
        "prediction_dates": float(scored["date"].nunique()),
        "top_mean_forward_return": float(top["realized_forward_return"].mean()),
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
    report_dir: Path,
    weights_dir: Path,
) -> RunArtifacts:
    out_dir = backtest_exports_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path | None] = {}
    if artifacts.predictions is not None:
        prediction_path = out_dir / f"{artifacts.run_id}_ranker_predictions.csv"
        artifacts.predictions.to_csv(prediction_path, index=False)
        paths["predictions"] = prediction_path
    if artifacts.raw_predictions is not None:
        raw_prediction_path = out_dir / f"{artifacts.run_id}_raw_ranker_predictions.csv"
        artifacts.raw_predictions.to_csv(raw_prediction_path, index=False)
        paths["raw_predictions"] = raw_prediction_path

    signals_path = out_dir / f"{artifacts.run_id}_signals.csv"
    returns_path = out_dir / f"{artifacts.run_id}_returns.csv"
    equity_path = out_dir / f"{artifacts.run_id}_equity_curve.csv"
    trade_log_path = out_dir / f"{artifacts.run_id}_trade_log.csv"
    snapshots_path = out_dir / f"{artifacts.run_id}_portfolio_snapshots.csv"
    target_events_path = weights_dir / f"{artifacts.run_id}_target_weight_events.csv"
    held_weights_path = weights_dir / f"{artifacts.run_id}_held_target_weights.csv"

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
        output_dir=report_dir,
        vectorbt_report_path=report_path,
        exported_paths=vanilla_export_paths,
        filename=f"{artifacts.run_id}_vanilla_comparison_report.pdf",
        test_intervals=test_intervals,
    )

    return RunArtifacts(
        label=artifacts.label,
        run_id=artifacts.run_id,
        result=artifacts.result,
        raw_predictions=artifacts.raw_predictions,
        predictions=artifacts.predictions,
        signals=artifacts.signals,
        target_weights=artifacts.target_weights,
        regime_summary=artifacts.regime_summary,
        eval_summary=artifacts.eval_summary,
        output_paths=paths,
        model_cache_summary=artifacts.model_cache_summary,
    )


def train_and_backtest_ranker(
    spec: RankerSpec,
    *,
    raw: pl.DataFrame,
    config: BacktestConfig,
    engine: Any,
    benchmark_close: pd.DataFrame | None,
    test_intervals: list[DateInterval],
    run_stamp: str,
    model_weights_dir: Path | None = None,
    use_model_cache: bool = True,
) -> RunArtifacts:
    target_col = f"Return_fwd_{spec.forward_period}"
    featured = build_reference_feature_frame(raw, forward_period=spec.forward_period)
    dataset = prepare_dataset(
        featured,
        target_col=target_col,
        predictor_cols=spec.predictor_cols,
    )
    run_id = (
        f"vn_ml_ranker_{spec.name}_top{spec.max_positions}_"
        f"fwd{spec.forward_period}_rb{spec.rebalance_frequency}_{run_stamp}"
    ).replace("/", "_")
    model_cache_id = (
        f"{spec.name}_top{spec.max_positions}_fwd{spec.forward_period}_"
        f"rb{spec.rebalance_frequency}_tw{spec.train_window}"
    ).replace("/", "_")
    model_artifact_dir = (
        model_weights_dir / model_cache_id if model_weights_dir is not None else None
    )
    if model_artifact_dir is not None:
        model_artifact_dir.mkdir(parents=True, exist_ok=True)

    raw_predictions = walk_forward_rank(
        frame=dataset,
        predictor=XGBRankerPredictor(**spec.xgb_params),
        predictor_cols=spec.predictor_cols,
        target_col=target_col,
        train_window=spec.train_window,
        rebalance_frequency=spec.rebalance_frequency,
        target_horizon=spec.forward_period,
        test_intervals=test_intervals,
        model_artifact_dir=model_artifact_dir,
        use_model_cache=use_model_cache,
    )
    model_cache_summary = dict(raw_predictions.attrs.get("model_cache_summary", {}))
    if model_cache_summary:
        print(
            f"Model cache {spec.name}: "
            f"hits={model_cache_summary.get('hits', 0)} "
            f"misses={model_cache_summary.get('misses', 0)} "
            f"saves={model_cache_summary.get('saves', 0)} "
            f"dir={model_cache_summary.get('artifact_dir')}",
            flush=True,
        )
    strategy_predictions, regime_summary = apply_param_benchmark_regime_filter(
        raw_predictions,
        benchmark_close,
        spec.benchmark_regime,
    )
    strategy = XGBRankerTopKStrategy(
        predictions=strategy_predictions,
        sessions=raw["date"].unique().to_list(),
        top_n=spec.max_positions,
        min_top_score=spec.min_top_score,
        min_score_spread=spec.min_score_spread,
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
        raw_predictions=raw_predictions,
        predictions=strategy_predictions,
        signals=result.signals if not result.signals.is_empty() else signals,
        target_weights=targets,
        regime_summary=regime_summary,
        eval_summary=summarize_topk_predictions(strategy_predictions, spec.max_positions),
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
    run_id = f"vn_ml_ranker_weight_ensemble_{run_stamp}"
    prediction_rows = sum(
        0 if run.predictions is None else len(run.predictions) for run in separate_runs
    )
    prediction_dates: set[object] = set()
    for run in separate_runs:
        if run.predictions is not None and not run.predictions.empty:
            prediction_dates.update(pd.to_datetime(run.predictions["date"]).dt.date)
    return RunArtifacts(
        label="ensemble",
        run_id=run_id,
        result=result,
        raw_predictions=None,
        predictions=None,
        signals=result.signals if not result.signals.is_empty() else signals,
        target_weights=blended,
        regime_summary={
            "enabled": "sleeve-level",
            "rule": "sleeve-level",
            "prediction_rows_before": prediction_rows,
            "prediction_rows_after": prediction_rows,
        },
        eval_summary={
            "prediction_rows": float(prediction_rows),
            "prediction_dates": float(len(prediction_dates)),
            "top_mean_forward_return": float("nan"),
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
    path = report_dir / f"vn_ml_ranker_weight_ensemble_summary_{run_stamp}.pdf"
    metrics = pd.DataFrame([metric_row(run) for run in runs])
    equity = pd.concat([result_equity_series(run) for run in runs], axis=1).sort_index()

    plt.close("all")
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        title = "VN ML Ranker Separate and Ensemble Report"
        subtitle = f"Test intervals: {format_date_intervals(test_intervals)}"
        fig.text(0.03, 0.95, title, fontsize=15, weight="bold")
        fig.text(0.03, 0.92, subtitle, fontsize=9)
        display_cols = [
            "label",
            "metric_sharpe",
            "metric_cagr",
            "metric_max_drawdown",
            "metric_sortino",
            "metric_win_rate",
            "signal_dates",
            "eval_prediction_dates",
        ]
        display = metrics[[column for column in display_cols if column in metrics.columns]].copy()
        display_labels = {
            "label": "Run",
            "metric_sharpe": "Sharpe",
            "metric_cagr": "CAGR",
            "metric_max_drawdown": "MDD",
            "metric_sortino": "Sortino",
            "metric_win_rate": "Win Rate",
            "signal_dates": "Signal Dates",
            "eval_prediction_dates": "Eval Dates",
        }
        for column in display.columns:
            if column.startswith("metric_") or column.startswith("eval_"):
                display[column] = pd.to_numeric(display[column], errors="coerce").round(4)
        col_labels = [display_labels.get(column, column) for column in display.columns]
        table = ax.table(
            cellText=display.fillna("").astype(str).values,
            colLabels=col_labels,
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
                "Title": "VN ML Ranker Weight Ensemble Summary",
                "Author": "QTradeSystematic",
            }
        )
    plt.close("all")
    return path


def save_summary_files(
    runs: list[RunArtifacts],
    *,
    base_config_path: Path,
    specs: list[RankerSpec],
    ensemble_weights: list[float],
    yaml_dir: Path,
    run_stamp: str,
    combined_report_path: Path,
) -> tuple[Path, Path]:
    metrics = pd.DataFrame([metric_row(run) for run in runs])
    metrics_path = yaml_dir / f"vn_ml_ranker_weight_ensemble_metrics_{run_stamp}.csv"
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
            "regime_filter": "sleeve-level",
        },
        "combined_report_pdf": str(combined_report_path),
        "runs": [
            {
                "label": run.label,
                "run_id": run.run_id,
                "metrics": run.result.metrics,
                "regime_summary": run.regime_summary,
                "eval_summary": run.eval_summary,
                "model_cache_summary": run.model_cache_summary,
                "output_paths": {
                    key: str(value) if value is not None else None
                    for key, value in run.output_paths.items()
                },
            }
            for run in runs
        ],
    }
    yaml_path = yaml_dir / f"vn_ml_ranker_weight_ensemble_effective_{run_stamp}.yaml"
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
    base_config: BacktestConfig,
    specs: list[RankerSpec],
    weights: list[float],
    *,
    use_model_cache: bool = True,
) -> None:
    print(f"Base config      : {BASE_CONFIG_PATH}")
    print(f"Universe symbols : {len(config_symbols(base_config))}")
    print(f"Benchmark        : {base_config.benchmark}")
    print(f"Validation       : {format_date_intervals(test_intervals_from_config(base_config))}")
    print(f"Model cache      : {'load/save' if use_model_cache else 'save only'}")
    print("Model specs:")
    for spec, weight in zip(specs, weights, strict=True):
        print(
            "  - "
            f"{spec.name}: fwd={spec.forward_period}, rb={spec.rebalance_frequency}, "
            f"train_window={spec.train_window}, top={spec.max_positions}, "
            f"predictors={len(spec.predictor_cols)}, ensemble_weight={weight:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train two VN ML rankers separately and blend their held target weights."
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
        help="Ranker parameter YAML. Pass twice to override the default two configs.",
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
        resolve_ranker_spec(path.resolve(), index)
        for index, path in enumerate(model_paths, 1)
    ]
    ensemble_weights = parse_ensemble_weights(args.ensemble_weights, len(specs))

    resolved = Config.build(args.base_config)
    config = resolved.raw
    validate_vn_ranker_config(config)

    if args.dry_run:
        print_dry_run(config, specs, ensemble_weights, use_model_cache=not args.no_model_cache)
        return

    yaml_dir, weights_dir, report_dir = resolve_output_dirs(specs)
    model_weights_dir = resolve_model_weights_dir(specs)
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
    benchmark_close = asyncio.run(load_benchmark_close_frame(data_manager, config))

    separate_runs: list[RunArtifacts] = []
    for spec in specs:
        print(
            f"Training {spec.name}: fwd={spec.forward_period}, "
            f"rb={spec.rebalance_frequency}, top={spec.max_positions}"
        )
        separate_runs.append(
            train_and_backtest_ranker(
                spec,
                raw=raw,
                config=config,
                engine=resolved.engine,
                benchmark_close=benchmark_close,
                test_intervals=test_intervals,
                run_stamp=run_stamp,
                model_weights_dir=model_weights_dir,
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
            report_dir=report_dir,
            weights_dir=weights_dir,
        )
        for run in all_runs
    ]
    combined_report_path = save_combined_report_pdf(
        exported_runs,
        report_dir=report_dir,
        run_stamp=run_stamp,
        test_intervals=test_intervals,
    )
    metrics_path, effective_yaml_path = save_summary_files(
        exported_runs,
        base_config_path=args.base_config,
        specs=specs,
        ensemble_weights=ensemble_weights,
        yaml_dir=yaml_dir,
        run_stamp=run_stamp,
        combined_report_path=combined_report_path,
    )

    print("\nResults:")
    for run in exported_runs:
        print(f"  {run.label}: {run.result.metrics}")
        print(f"    vanilla PDF: {run.output_paths.get('vanilla_pdf')}")
        print(f"    target weights: {run.output_paths.get('held_target_weights')}")
    print(f"Combined report : {combined_report_path}")
    print(f"Metrics CSV     : {metrics_path}")
    print(f"Effective YAML  : {effective_yaml_path}")


if __name__ == "__main__":
    main()
