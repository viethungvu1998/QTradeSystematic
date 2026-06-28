"""Compare VN ML classifier, ranker, and classifier-gated ranker hybrid.

The core classifier and ranker implementations are imported from the existing
combined end-to-end scripts. This file only coordinates those flows and applies
the hybrid portfolio rule:

1. build the classifier held-target-weight ensemble;
2. build the ranker held-target-weight ensemble;
3. combine those target weights using either a static blend or a stricter
   classifier gate.
"""
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import sys
import warnings
from dataclasses import dataclass, replace
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

import combine_ml_e2e_classification as classifier_flow
import combine_ml_e2e_ranking as ranker_flow
from qts.config.builder import Config
from qts.config.loader import load_config
from qts.research.backtest.base import BacktestConfig, BacktestResult
from qts.utils.date_intervals import (
    DateInterval,
    format_date_intervals,
    test_intervals_from_config,
)
from qts.utils.paths import backtest_exports_dir, qts_root


BASE_CONFIG_PATH = classifier_flow.BASE_CONFIG_PATH
DEFAULT_CLASSIFIER_CONFIGS = classifier_flow.DEFAULT_MODEL_CONFIGS
DEFAULT_RANKER_CONFIGS = ranker_flow.DEFAULT_MODEL_CONFIGS
RUNTIME_DIR = repo_root / ".qts_notebooke_runtime" / "combined_ml_classifier_ranker"
DEFAULT_REPORT_DIR = repo_root / "output" / "pdf"


@dataclass(frozen=True, slots=True)
class OutputDirs:
    yaml_dir: Path
    target_weights_dir: Path
    report_dir: Path


@dataclass(frozen=True, slots=True)
class HybridTargets:
    targets: pd.DataFrame
    eval_summary: dict[str, float]


@dataclass(frozen=True, slots=True)
class ComparisonRun:
    family: str
    label: str
    run_id: str
    result: BacktestResult
    signals: pl.DataFrame
    target_weights: pd.DataFrame
    eval_summary: dict[str, float]


def resolve_path(raw: Path | str | None, fallback: Path) -> Path:
    if raw in (None, ""):
        return fallback
    path = Path(str(raw)).expanduser()
    return path if path.is_absolute() else repo_root / path


def resolve_output_dirs(args: argparse.Namespace) -> OutputDirs:
    yaml_dir = resolve_path(args.yaml_dir, RUNTIME_DIR / "yaml")
    weights_dir = resolve_path(args.weights_dir, RUNTIME_DIR / "weights")
    report_dir = resolve_path(args.report_dir, DEFAULT_REPORT_DIR)
    for path in (yaml_dir, weights_dir, report_dir):
        path.mkdir(parents=True, exist_ok=True)
    return OutputDirs(
        yaml_dir=yaml_dir,
        target_weights_dir=weights_dir,
        report_dir=report_dir,
    )


def align_target_weights(
    targets: pd.DataFrame,
    *,
    sessions: pd.DatetimeIndex,
    symbols: list[str],
) -> pd.DataFrame:
    if targets.empty:
        return pd.DataFrame(0.0, index=sessions, columns=sorted(symbols))
    aligned = targets.copy()
    aligned.index = pd.to_datetime(aligned.index)
    return (
        aligned.reindex(index=sessions, columns=sorted(symbols))
        .ffill()
        .fillna(0.0)
        .astype(float)
    )


def build_classifier_gated_ranker_targets(
    *,
    ranker_targets: pd.DataFrame,
    classifier_targets: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    symbols: list[str],
    min_classifier_weight: float = 0.0,
    renormalize_gross: bool = False,
) -> HybridTargets:
    """Keep ranker weights only on symbols confirmed by the classifier ensemble."""

    ranker = align_target_weights(ranker_targets, sessions=sessions, symbols=symbols)
    classifier = align_target_weights(classifier_targets, sessions=sessions, symbols=symbols)
    gate = classifier.abs() > float(min_classifier_weight)
    hybrid = ranker.where(gate, 0.0)

    ranker_gross = ranker.abs().sum(axis=1)
    classifier_gross = classifier.abs().sum(axis=1)
    raw_hybrid_gross = hybrid.abs().sum(axis=1)
    if renormalize_gross:
        scale = ranker_gross.divide(raw_hybrid_gross).replace([np.inf, -np.inf], 0.0)
        scale = scale.where(raw_hybrid_gross > 1e-12, 0.0)
        hybrid = hybrid.mul(scale, axis=0)

    hybrid = hybrid.clip(lower=-1.0, upper=1.0)
    hybrid_gross = hybrid.abs().sum(axis=1)
    ranker_positions = (ranker.abs() > 1e-12).sum(axis=1)
    confirmed_positions = ((ranker.abs() > 1e-12) & gate).sum(axis=1)
    nonzero_ranker_dates = ranker_gross > 1e-12

    confirmed_ratio = (
        confirmed_positions.loc[nonzero_ranker_dates].sum()
        / ranker_positions.loc[nonzero_ranker_dates].sum()
        if nonzero_ranker_dates.any()
        and ranker_positions.loc[nonzero_ranker_dates].sum() > 0
        else 0.0
    )
    retained_gross_ratio = (
        raw_hybrid_gross.loc[nonzero_ranker_dates].sum()
        / ranker_gross.loc[nonzero_ranker_dates].sum()
        if nonzero_ranker_dates.any() and ranker_gross.loc[nonzero_ranker_dates].sum() > 0
        else 0.0
    )
    eval_summary = {
        "method": "classifier_gated_ranker",
        "min_classifier_weight": float(min_classifier_weight),
        "renormalize_gross": float(bool(renormalize_gross)),
        "avg_ranker_gross": float(ranker_gross.mean()) if len(ranker_gross) else 0.0,
        "avg_classifier_gross": float(classifier_gross.mean()) if len(classifier_gross) else 0.0,
        "avg_hybrid_gross": float(hybrid_gross.mean()) if len(hybrid_gross) else 0.0,
        "invested_dates": float((hybrid_gross > 1e-12).sum()),
        "confirmed_position_ratio": float(confirmed_ratio),
        "retained_ranker_gross_ratio": float(retained_gross_ratio),
    }
    return HybridTargets(targets=hybrid, eval_summary=eval_summary)


def build_target_weight_blend(
    *,
    classifier_targets: pd.DataFrame,
    ranker_targets: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    symbols: list[str],
    classifier_weight: float,
    ranker_weight: float,
) -> HybridTargets:
    """Blend component target weights with fixed ex-ante sleeve weights."""

    total_weight = float(classifier_weight) + float(ranker_weight)
    if total_weight <= 0:
        raise ValueError("Hybrid blend weights must sum to a positive value")
    classifier_weight = float(classifier_weight) / total_weight
    ranker_weight = float(ranker_weight) / total_weight

    classifier = align_target_weights(classifier_targets, sessions=sessions, symbols=symbols)
    ranker = align_target_weights(ranker_targets, sessions=sessions, symbols=symbols)
    hybrid = classifier.mul(classifier_weight).add(ranker.mul(ranker_weight), fill_value=0.0)
    hybrid = hybrid.clip(lower=-1.0, upper=1.0)

    classifier_gross = classifier.abs().sum(axis=1)
    ranker_gross = ranker.abs().sum(axis=1)
    hybrid_gross = hybrid.abs().sum(axis=1)
    eval_summary = {
        "method": "target_weight_blend",
        "classifier_weight": float(classifier_weight),
        "ranker_weight": float(ranker_weight),
        "avg_classifier_gross": float(classifier_gross.mean()) if len(classifier_gross) else 0.0,
        "avg_ranker_gross": float(ranker_gross.mean()) if len(ranker_gross) else 0.0,
        "avg_hybrid_gross": float(hybrid_gross.mean()) if len(hybrid_gross) else 0.0,
        "invested_dates": float((hybrid_gross > 1e-12).sum()),
    }
    return HybridTargets(targets=hybrid, eval_summary=eval_summary)


def next_session_lookup(sessions: pd.DatetimeIndex) -> dict[object, pd.Timestamp]:
    sorted_sessions = pd.DatetimeIndex(pd.to_datetime(sessions)).sort_values()
    return {
        current.date(): next_session
        for current, next_session in zip(
            sorted_sessions[:-1],
            sorted_sessions[1:],
            strict=False,
        )
    }


def ranker_percentile_score_matrix(
    *,
    ranker_runs: list[ranker_flow.RunArtifacts],
    ranker_weights: list[float],
    sessions: pd.DatetimeIndex,
    symbols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Blend raw ranker scores into held cross-sectional score percentiles."""

    if len(ranker_runs) != len(ranker_weights):
        raise ValueError("ranker_runs and ranker_weights must have the same length")

    sorted_symbols = sorted(symbols)
    neutral = pd.DataFrame(0.5, index=sessions, columns=sorted_symbols)
    any_coverage = pd.DataFrame(False, index=sessions, columns=sorted_symbols)
    if not ranker_runs:
        return neutral, any_coverage, {
            "raw_ranker_runs": 0.0,
            "raw_ranker_rows": 0.0,
            "raw_ranker_prediction_dates": 0.0,
        }

    total_weight = float(sum(ranker_weights))
    if total_weight <= 0:
        raise ValueError("Ranker ensemble weights must sum to a positive value")

    session_map = next_session_lookup(sessions)
    blended = pd.DataFrame(0.0, index=sessions, columns=sorted_symbols)
    raw_rows = 0
    raw_dates: set[object] = set()

    for run, raw_weight in zip(ranker_runs, ranker_weights, strict=True):
        sleeve_weight = float(raw_weight) / total_weight
        predictions = run.raw_predictions
        if predictions is None or predictions.empty:
            blended = blended.add(neutral * sleeve_weight, fill_value=0.0)
            continue

        frame = predictions.loc[:, ["date", "symbol", "score"]].copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        frame["execution_date"] = frame["date"].map(session_map)
        frame = frame.loc[frame["execution_date"].notna()].copy()
        frame = frame.loc[frame["symbol"].isin(sorted_symbols)].copy()
        if frame.empty:
            blended = blended.add(neutral * sleeve_weight, fill_value=0.0)
            continue

        raw_rows += int(len(frame))
        raw_dates.update(frame["date"].drop_duplicates().tolist())
        frame["ranker_percentile"] = frame.groupby("date", observed=True)["score"].rank(
            pct=True,
            method="average",
        )
        events = frame.pivot_table(
            index="execution_date",
            columns="symbol",
            values="ranker_percentile",
            aggfunc="mean",
        )
        events.index = pd.to_datetime(events.index)
        events = events.reindex(index=sessions, columns=sorted_symbols)
        coverage = events.ffill().notna()
        sleeve_scores = events.ffill().fillna(0.5)

        blended = blended.add(sleeve_scores * sleeve_weight, fill_value=0.0)
        any_coverage = any_coverage | coverage

    summary = {
        "raw_ranker_runs": float(len(ranker_runs)),
        "raw_ranker_rows": float(raw_rows),
        "raw_ranker_prediction_dates": float(len(raw_dates)),
    }
    return blended.clip(lower=0.0, upper=1.0), any_coverage, summary


def build_classifier_ranker_score_tilt_targets(
    *,
    classifier_targets: pd.DataFrame,
    ranker_runs: list[ranker_flow.RunArtifacts],
    ranker_weights: list[float],
    sessions: pd.DatetimeIndex,
    symbols: list[str],
    ranker_tilt_strength: float,
    max_position_weight: float | None = None,
) -> HybridTargets:
    """Use raw ranker score percentiles to tilt classifier-held weights."""

    classifier = align_target_weights(classifier_targets, sessions=sessions, symbols=symbols)
    ranker_scores, ranker_coverage, score_summary = ranker_percentile_score_matrix(
        ranker_runs=ranker_runs,
        ranker_weights=ranker_weights,
        sessions=sessions,
        symbols=symbols,
    )
    tilt = 1.0 + float(ranker_tilt_strength) * (ranker_scores - 0.5)
    hybrid = classifier.mul(tilt, axis=0)

    if max_position_weight is None:
        observed = classifier.abs().max().max() if not classifier.empty else 0.0
        max_position_weight = float(observed) if observed and observed > 0 else 1.0
    if max_position_weight <= 0:
        raise ValueError("max_position_weight must be positive")

    hybrid = np.sign(hybrid) * hybrid.abs().clip(upper=float(max_position_weight))
    hybrid = hybrid.where(classifier.abs() > 1e-12, 0.0).astype(float)

    classifier_gross = classifier.abs().sum(axis=1)
    hybrid_gross = hybrid.abs().sum(axis=1)
    classifier_mask = classifier.abs() > 1e-12
    covered_positions = (classifier_mask & ranker_coverage).sum().sum()
    total_positions = classifier_mask.sum().sum()
    tilt_values = tilt.where(classifier_mask).stack()

    eval_summary = {
        "method": "classifier_ranker_score_tilt",
        "ranker_source": "raw_predictions_before_portfolio_filters",
        "uses_benchmark_regime": 0.0,
        "uses_min_top_score": 0.0,
        "ranker_tilt_strength": float(ranker_tilt_strength),
        "avg_tilt_factor": float(tilt_values.mean()) if len(tilt_values) else 1.0,
        "ranker_score_coverage_ratio": (
            float(covered_positions / total_positions) if total_positions else 0.0
        ),
        "retained_classifier_gross_ratio": (
            float(hybrid_gross.sum() / classifier_gross.sum())
            if classifier_gross.sum() > 0
            else 0.0
        ),
        "max_position_weight": float(max_position_weight),
        "avg_classifier_gross": float(classifier_gross.mean()) if len(classifier_gross) else 0.0,
        "avg_hybrid_gross": float(hybrid_gross.mean()) if len(hybrid_gross) else 0.0,
        "invested_dates": float((hybrid_gross > 1e-12).sum()),
        **score_summary,
    }
    return HybridTargets(targets=hybrid, eval_summary=eval_summary)


def build_hybrid_run(
    *,
    classifier_ensemble: classifier_flow.RunArtifacts,
    ranker_ensemble: ranker_flow.RunArtifacts,
    ranker_runs: list[ranker_flow.RunArtifacts],
    ranker_weights: list[float],
    raw: pl.DataFrame,
    config: BacktestConfig,
    engine: Any,
    run_stamp: str,
    hybrid_method: str,
    min_classifier_weight: float,
    renormalize_gross: bool,
    hybrid_ranker_weight: float,
    ranker_tilt_strength: float,
) -> ComparisonRun:
    sessions = ranker_flow.session_index(raw)
    symbols = ranker_flow.config_symbols(config)
    if hybrid_method == "classifier_gated_ranker":
        hybrid = build_classifier_gated_ranker_targets(
            ranker_targets=ranker_ensemble.target_weights,
            classifier_targets=classifier_ensemble.target_weights,
            sessions=sessions,
            symbols=symbols,
            min_classifier_weight=min_classifier_weight,
            renormalize_gross=renormalize_gross,
        )
        label = "classifier_gated_ranker"
    elif hybrid_method == "target_weight_blend":
        hybrid = build_target_weight_blend(
            classifier_targets=classifier_ensemble.target_weights,
            ranker_targets=ranker_ensemble.target_weights,
            sessions=sessions,
            symbols=symbols,
            classifier_weight=1.0 - float(hybrid_ranker_weight),
            ranker_weight=float(hybrid_ranker_weight),
        )
        label = "weight_blend"
    elif hybrid_method == "classifier_ranker_score_tilt":
        max_position_weight = (
            float(classifier_ensemble.target_weights.abs().max().max())
            if not classifier_ensemble.target_weights.empty
            else None
        )
        hybrid = build_classifier_ranker_score_tilt_targets(
            classifier_targets=classifier_ensemble.target_weights,
            ranker_runs=ranker_runs,
            ranker_weights=ranker_weights,
            sessions=sessions,
            symbols=symbols,
            ranker_tilt_strength=ranker_tilt_strength,
            max_position_weight=max_position_weight,
        )
        label = f"classifier_ranker_score_tilt_{ranker_tilt_strength:.2f}"
    else:
        raise ValueError(f"Unsupported hybrid method: {hybrid_method!r}")

    signals = ranker_flow.target_matrix_to_signals(hybrid.targets)
    strategy = ranker_flow.PrecomputedSignalStrategy(signals)
    result = engine.run(strategy=strategy, data=raw, config=config)
    suffix = (
        f"_tilt{int(round(ranker_tilt_strength * 100)):03d}"
        if hybrid_method == "classifier_ranker_score_tilt"
        else ""
    )
    run_id = f"vn_ml_{hybrid_method}_hybrid{suffix}_{run_stamp}"
    return ComparisonRun(
        family="hybrid",
        label=label,
        run_id=run_id,
        result=result,
        signals=result.signals if not result.signals.is_empty() else signals,
        target_weights=hybrid.targets,
        eval_summary=hybrid.eval_summary,
    )


def model_cache_eval_summary(summary: dict[str, Any]) -> dict[str, float]:
    if not summary:
        return {}
    source_runs = summary.get("source_runs")
    rows = (
        list(source_runs.values())
        if isinstance(source_runs, dict)
        else [summary]
    )
    result: dict[str, float] = {}
    for key in ["attempts", "hits", "misses", "saves", "failed_fits"]:
        total = 0.0
        found = False
        for row in rows:
            if isinstance(row, dict) and key in row:
                total += float(row.get(key, 0) or 0)
                found = True
        if found:
            result[f"model_cache_{key}"] = total
    return result


def comparison_from_classifier_run(run: classifier_flow.RunArtifacts) -> ComparisonRun:
    return ComparisonRun(
        family="classifier",
        label=run.label,
        run_id=run.run_id,
        result=run.result,
        signals=run.signals,
        target_weights=run.target_weights,
        eval_summary={
            **run.eval_summary,
            **model_cache_eval_summary(getattr(run, "model_cache_summary", {})),
        },
    )


def comparison_from_ranker_run(run: ranker_flow.RunArtifacts) -> ComparisonRun:
    return ComparisonRun(
        family="ranker",
        label=run.label,
        run_id=run.run_id,
        result=run.result,
        signals=run.signals,
        target_weights=run.target_weights,
        eval_summary={
            **run.eval_summary,
            **model_cache_eval_summary(getattr(run, "model_cache_summary", {})),
        },
    )


def result_equity_series(run: ComparisonRun) -> pd.Series:
    equity = run.result.equity_curve.to_pandas()
    if equity.empty:
        return pd.Series(dtype=float, name=f"{run.family}:{run.label}")
    equity["date"] = pd.to_datetime(equity["date"])
    return equity.set_index("date")["equity"].astype(float).rename(f"{run.family}:{run.label}")


def total_return_and_ending_equity(run: ComparisonRun) -> tuple[float, float]:
    equity = result_equity_series(run).dropna()
    if equity.empty:
        return 0.0, 0.0
    first = float(equity.iloc[0])
    ending = float(equity.iloc[-1])
    total_return = (ending / first - 1.0) if first > 0 else 0.0
    return float(total_return), ending


def target_exposure_summary(targets: pd.DataFrame) -> dict[str, float]:
    if targets.empty:
        return {
            "avg_gross_exposure": 0.0,
            "max_gross_exposure": 0.0,
            "invested_dates": 0.0,
        }
    gross = targets.fillna(0.0).abs().sum(axis=1)
    return {
        "avg_gross_exposure": float(gross.mean()),
        "max_gross_exposure": float(gross.max()),
        "invested_dates": float((gross > 1e-12).sum()),
    }


def parse_float_list(raw: str) -> list[float]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        return []
    return [float(item) for item in values]


def metric_row(run: ComparisonRun) -> dict[str, Any]:
    total_return, ending_equity = total_return_and_ending_equity(run)
    row: dict[str, Any] = {
        "family": run.family,
        "label": run.label,
        "run_id": run.run_id,
        "total_return": total_return,
        "ending_equity": ending_equity,
        "signal_rows": run.signals.height,
        "signal_dates": run.signals["date"].n_unique() if run.signals.height else 0,
        "target_dates": len(run.target_weights),
    }
    row.update(target_exposure_summary(run.target_weights))
    row.update({f"metric_{key}": value for key, value in run.result.metrics.items()})
    row.update({f"eval_{key}": value for key, value in run.eval_summary.items()})
    return row


def choose_hybrid_candidate(
    candidates: list[ComparisonRun],
    *,
    classifier_ensemble: classifier_flow.RunArtifacts,
) -> ComparisonRun:
    if not candidates:
        raise ValueError("At least one hybrid candidate is required")

    classifier_sharpe = float(classifier_ensemble.result.metrics.get("sharpe", float("-inf")))
    classifier_mdd = float(classifier_ensemble.result.metrics.get("max_drawdown", float("inf")))

    def candidate_key(run: ComparisonRun) -> tuple[float, float]:
        sharpe = float(run.result.metrics.get("sharpe", float("-inf")))
        drawdown = float(run.result.metrics.get("max_drawdown", float("inf")))
        return sharpe, -drawdown

    accepted = [
        run
        for run in candidates
        if float(run.result.metrics.get("sharpe", float("-inf"))) > classifier_sharpe
        and float(run.result.metrics.get("max_drawdown", float("inf"))) <= classifier_mdd
    ]
    selected = max(accepted or candidates, key=candidate_key)
    selected_sharpe = float(selected.result.metrics.get("sharpe", float("nan")))
    selected_mdd = float(selected.result.metrics.get("max_drawdown", float("nan")))
    eval_summary = {
        **selected.eval_summary,
        "acceptance_classifier_sharpe": classifier_sharpe,
        "acceptance_classifier_max_drawdown": classifier_mdd,
        "acceptance_selected_sharpe": selected_sharpe,
        "acceptance_selected_max_drawdown": selected_mdd,
        "acceptance_passed": float(any(run is selected for run in accepted)),
    }
    return replace(selected, eval_summary=eval_summary)


def export_comparison_artifacts(
    run: ComparisonRun,
    *,
    output_dirs: OutputDirs,
) -> dict[str, Path]:
    out_dir = backtest_exports_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_dirs.target_weights_dir.mkdir(parents=True, exist_ok=True)

    signals_path = out_dir / f"{run.run_id}_signals.csv"
    returns_path = out_dir / f"{run.run_id}_returns.csv"
    equity_path = out_dir / f"{run.run_id}_equity_curve.csv"
    held_weights_path = output_dirs.target_weights_dir / f"{run.run_id}_held_target_weights.csv"
    target_events_path = output_dirs.target_weights_dir / f"{run.run_id}_target_weight_events.csv"

    run.signals.write_csv(signals_path)
    run.result.returns.write_csv(returns_path)
    run.result.equity_curve.write_csv(equity_path)
    ranker_flow.held_weights_long_frame(run.target_weights).to_csv(
        held_weights_path,
        index=False,
    )
    ranker_flow.positive_target_events(run.signals).to_csv(
        target_events_path,
        index=False,
    )
    return {
        "signals": signals_path,
        "returns": returns_path,
        "equity_curve": equity_path,
        "held_target_weights": held_weights_path,
        "target_weight_events": target_events_path,
    }


def save_comparison_report_pdf(
    runs: list[ComparisonRun],
    *,
    report_dir: Path,
    run_stamp: str,
    test_intervals: list[DateInterval],
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"vn_ml_classifier_ranker_hybrid_comparison_{run_stamp}.pdf"
    metrics = pd.DataFrame([metric_row(run) for run in runs])
    equity = pd.concat([result_equity_series(run) for run in runs], axis=1).sort_index()

    plt.close("all")
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        fig.text(
            0.03,
            0.95,
            "VN ML Classifier-Ranker Hybrid Comparison",
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
            "family",
            "label",
            "total_return",
            "metric_cagr",
            "metric_sharpe",
            "metric_max_drawdown",
            "metric_sortino",
            "metric_win_rate",
            "avg_gross_exposure",
            "signal_dates",
        ]
        display = metrics[[column for column in display_cols if column in metrics.columns]].copy()
        if "label" in display:
            display["label"] = (
                display["label"]
                .astype(str)
                .str.replace("classifier_ranker_score_tilt_", "score_tilt_", regex=False)
            )
        display_labels = {
            "family": "Family",
            "label": "Run",
            "total_return": "Total Ret",
            "metric_cagr": "CAGR",
            "metric_sharpe": "Sharpe",
            "metric_max_drawdown": "MDD",
            "metric_sortino": "Sortino",
            "metric_win_rate": "Win Rate",
            "avg_gross_exposure": "Avg Gross",
            "signal_dates": "Signal Dates",
        }
        for column in display.columns:
            if column not in {"family", "label"}:
                display[column] = pd.to_numeric(display[column], errors="coerce").round(4)
        table = ax.table(
            cellText=display.fillna("").astype(str).values,
            colLabels=[display_labels.get(column, column) for column in display.columns],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
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
            ranker_flow.shade_test_intervals_on_axis(ax, test_intervals)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            fig.autofmt_xdate()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        pdf.infodict().update(
            {
                "Title": "VN ML Classifier-Ranker Hybrid Comparison",
                "Author": "QTradeSystematic",
            }
        )
    plt.close("all")
    return path


def save_summary_files(
    runs: list[ComparisonRun],
    *,
    base_config_path: Path,
    classifier_specs: list[classifier_flow.ClassificationSpec],
    ranker_specs: list[ranker_flow.RankerSpec],
    classifier_weights: list[float],
    ranker_weights: list[float],
    output_paths: dict[str, dict[str, Path]],
    output_dirs: OutputDirs,
    run_stamp: str,
    report_path: Path,
    hybrid_candidates: list[ComparisonRun] | None = None,
) -> tuple[Path, Path]:
    metrics = pd.DataFrame([metric_row(run) for run in runs])
    metrics_path = output_dirs.yaml_dir / f"vn_ml_classifier_ranker_hybrid_metrics_{run_stamp}.csv"
    metrics.to_csv(metrics_path, index=False)

    payload = {
        "run_stamp": run_stamp,
        "qts_root": str(qts_root()),
        "base_config": str(base_config_path),
        "classifier_model_configs": [str(spec.path) for spec in classifier_specs],
        "ranker_model_configs": [str(spec.path) for spec in ranker_specs],
        "classifier_ensemble_weights": {
            spec.name: float(weight)
            for spec, weight in zip(classifier_specs, classifier_weights, strict=True)
        },
        "ranker_ensemble_weights": {
            spec.name: float(weight)
            for spec, weight in zip(ranker_specs, ranker_weights, strict=True)
        },
        "hybrid": {
            "method": next(
                (run.eval_summary.get("method") for run in runs if run.family == "hybrid"),
                "classifier_gated_ranker",
            ),
            "component_inputs": next(
                (
                    "classifier held weights plus raw ranker prediction scores"
                    for run in runs
                    if run.family == "hybrid"
                    and run.eval_summary.get("method") == "classifier_ranker_score_tilt"
                ),
                "classifier and ranker ensemble held target weights",
            ),
            "allocation": next(
                (
                    "classifier-owned weights tilted by raw ranker score percentiles"
                    for run in runs
                    if run.family == "hybrid"
                    and run.eval_summary.get("method") == "classifier_ranker_score_tilt"
                ),
                "static target-weight blend unless classifier_gated_ranker is selected",
            ),
        },
        "hybrid_candidates": [
            metric_row(candidate)
            for candidate in (hybrid_candidates or [])
        ],
        "comparison_report_pdf": str(report_path),
        "metrics_csv": str(metrics_path),
        "runs": [
            {
                "family": run.family,
                "label": run.label,
                "run_id": run.run_id,
                "metrics": run.result.metrics,
                "eval_summary": run.eval_summary,
                "output_paths": {
                    key: str(value)
                    for key, value in output_paths.get(run.run_id, {}).items()
                },
            }
            for run in runs
        ],
    }
    yaml_path = output_dirs.yaml_dir / f"vn_ml_classifier_ranker_hybrid_effective_{run_stamp}.yaml"
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return metrics_path, yaml_path


def print_dry_run(
    *,
    base_config_path: Path,
    base_config: BacktestConfig,
    classifier_specs: list[classifier_flow.ClassificationSpec],
    ranker_specs: list[ranker_flow.RankerSpec],
    classifier_weights: list[float],
    ranker_weights: list[float],
    args: argparse.Namespace,
) -> None:
    print(f"Base config              : {base_config_path}")
    print(f"QTS root                 : {qts_root()}")
    print(f"Universe symbols         : {len(classifier_flow.config_symbols(base_config))}")
    print(f"Benchmark                : {base_config.benchmark}")
    validation_label = format_date_intervals(test_intervals_from_config(base_config))
    print(f"Validation               : {validation_label}")
    print(f"Hybrid method            : {args.hybrid_method}")
    print(f"Hybrid ranker weight     : {args.hybrid_ranker_weight}")
    print(f"Hybrid ranker tilt       : {args.ranker_tilt_strength}")
    print(f"Hybrid tilt grid         : {args.ranker_tilt_strength_grid}")
    print(f"Hybrid min class weight  : {args.min_classifier_weight}")
    print(f"Hybrid renormalize gross : {args.renormalize_gross}")
    print("Classifier core specs:")
    for spec, weight in zip(classifier_specs, classifier_weights, strict=True):
        print(
            "  - "
            f"{spec.name}: fwd={spec.forward_period}, rb={spec.rebalance_frequency}, "
            f"class_window={spec.classification_window}, train_window={spec.train_window}, "
            f"top={spec.max_positions}, predictors={len(spec.predictor_cols)}, "
            f"ensemble_weight={weight:.4f}"
        )
    print("Ranker core specs:")
    for spec, weight in zip(ranker_specs, ranker_weights, strict=True):
        print(
            "  - "
            f"{spec.name}: fwd={spec.forward_period}, rb={spec.rebalance_frequency}, "
            f"train_window={spec.train_window}, top={spec.max_positions}, "
            f"predictors={len(spec.predictor_cols)}, ensemble_weight={weight:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train VN classifier/ranker ensembles, build a classifier-gated "
            "ranker hybrid, and compare returns."
        )
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=BASE_CONFIG_PATH,
        help="Base QTS config for universe, data, validation, engine, and costs.",
    )
    parser.add_argument(
        "--classifier-config",
        type=Path,
        action="append",
        default=None,
        help="Classifier parameter YAML. Pass multiple times to override defaults.",
    )
    parser.add_argument(
        "--ranker-config",
        type=Path,
        action="append",
        default=None,
        help="Ranker parameter YAML. Pass multiple times to override defaults.",
    )
    parser.add_argument(
        "--classifier-ensemble-weights",
        default="0.5,0.5",
        help="Comma-separated classifier sleeve weights. Values are normalized.",
    )
    parser.add_argument(
        "--ranker-ensemble-weights",
        default="0.5,0.5",
        help="Comma-separated ranker sleeve weights. Values are normalized.",
    )
    parser.add_argument(
        "--hybrid-method",
        choices=[
            "classifier_gated_ranker",
            "target_weight_blend",
            "classifier_ranker_score_tilt",
        ],
        default="target_weight_blend",
        help="Hybrid rule applied to the classifier and ranker ensemble target weights.",
    )
    parser.add_argument(
        "--hybrid-ranker-weight",
        type=float,
        default=0.3,
        help="Ranker sleeve weight for --hybrid-method target_weight_blend.",
    )
    parser.add_argument(
        "--ranker-tilt-strength",
        type=float,
        default=0.5,
        help="Single tilt strength for --hybrid-method classifier_ranker_score_tilt.",
    )
    parser.add_argument(
        "--ranker-tilt-strength-grid",
        default="0.25,0.50,0.75",
        help=(
            "Comma-separated tilt strengths to sweep for "
            "--hybrid-method classifier_ranker_score_tilt. Pass an empty string "
            "to use --ranker-tilt-strength only."
        ),
    )
    parser.add_argument(
        "--min-classifier-weight",
        type=float,
        default=0.0,
        help="Minimum absolute classifier ensemble target weight required for the gate.",
    )
    parser.add_argument(
        "--renormalize-gross",
        action="store_true",
        help="Scale remaining hybrid positions back to the ranker ensemble gross exposure.",
    )
    parser.add_argument(
        "--include-separate",
        action="store_true",
        help="Include constituent classifier/ranker model runs in the comparison report.",
    )
    parser.add_argument(
        "--yaml-dir",
        type=Path,
        default=None,
        help="Directory for hybrid comparison metrics and effective YAML.",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help="Directory for hybrid comparison target-weight CSV files.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory for the hybrid comparison PDF.",
    )
    parser.add_argument(
        "--qts-root",
        type=Path,
        default=None,
        help="Canonical QTS root for data cache, database, and backtest exports.",
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
    if args.qts_root is not None:
        root = resolve_path(args.qts_root, args.qts_root)
        root.mkdir(parents=True, exist_ok=True)
        os.environ["QTS_ROOT"] = str(root)

    classifier_paths = args.classifier_config or DEFAULT_CLASSIFIER_CONFIGS
    ranker_paths = args.ranker_config or DEFAULT_RANKER_CONFIGS
    classifier_specs = [
        classifier_flow.resolve_classification_spec(path.resolve(), index)
        for index, path in enumerate(classifier_paths, 1)
    ]
    ranker_specs = [
        ranker_flow.resolve_ranker_spec(path.resolve(), index)
        for index, path in enumerate(ranker_paths, 1)
    ]
    classifier_weights = classifier_flow.parse_ensemble_weights(
        args.classifier_ensemble_weights,
        len(classifier_specs),
    )
    ranker_weights = ranker_flow.parse_ensemble_weights(
        args.ranker_ensemble_weights,
        len(ranker_specs),
    )

    if args.dry_run:
        base_config = load_config(args.base_config)
        classifier_flow.validate_vn_classification_config(base_config)
        ranker_flow.validate_vn_ranker_config(base_config)
        print_dry_run(
            base_config_path=args.base_config,
            base_config=base_config,
            classifier_specs=classifier_specs,
            ranker_specs=ranker_specs,
            classifier_weights=classifier_weights,
            ranker_weights=ranker_weights,
            args=args,
        )
        return

    resolved = Config.build(args.base_config)
    config = resolved.raw
    classifier_flow.validate_vn_classification_config(config)
    ranker_flow.validate_vn_ranker_config(config)

    output_dirs = resolve_output_dirs(args)
    classifier_output_dirs = classifier_flow.resolve_output_dirs(classifier_specs)
    ranker_yaml_dir, ranker_weights_dir, ranker_report_dir = ranker_flow.resolve_output_dirs(
        ranker_specs
    )
    ranker_model_weights_dir = ranker_flow.resolve_model_weights_dir(ranker_specs)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_manager = classifier_flow.build_data_manager(resolved)
    symbols = classifier_flow.config_symbols(config)
    test_intervals = test_intervals_from_config(config)

    print(f"Base config      : {args.base_config}")
    print(f"Symbols          : {len(symbols)}")
    print(f"Benchmark        : {config.benchmark}")
    print(f"Validation       : {format_date_intervals(test_intervals)}")
    print(f"QTS root         : {qts_root()}")
    print(f"Run stamp        : {run_stamp}")

    raw: pl.DataFrame = asyncio.run(
        data_manager.get_ohlcv(
            symbols=symbols,
            start=config.start_date,
            end=config.end_date,
        )
    )
    benchmark_close = asyncio.run(classifier_flow.load_benchmark_close(data_manager, config))

    classifier_feature_cache: dict[int, pd.DataFrame] = {}
    classifier_runs: list[classifier_flow.RunArtifacts] = []
    for spec in classifier_specs:
        print(
            f"Running classifier {spec.name}: fwd={spec.forward_period}, "
            f"rb={spec.rebalance_frequency}, top={spec.max_positions}"
        )
        classifier_runs.append(
            classifier_flow.train_and_backtest_classifier(
                spec,
                raw=raw,
                feature_cache=classifier_feature_cache,
                config=config,
                engine=resolved.engine,
                test_intervals=test_intervals,
                benchmark_close=benchmark_close,
                output_dirs=classifier_output_dirs,
                run_stamp=run_stamp,
                use_model_cache=not args.no_model_cache,
            )
        )
    classifier_ensemble = classifier_flow.build_ensemble_artifacts(
        classifier_runs,
        weights=classifier_weights,
        raw=raw,
        config=config,
        engine=resolved.engine,
        run_stamp=run_stamp,
    )

    ranker_runs: list[ranker_flow.RunArtifacts] = []
    for spec in ranker_specs:
        print(
            f"Running ranker {spec.name}: fwd={spec.forward_period}, "
            f"rb={spec.rebalance_frequency}, top={spec.max_positions}"
        )
        ranker_runs.append(
            ranker_flow.train_and_backtest_ranker(
                spec,
                raw=raw,
                config=config,
                engine=resolved.engine,
                benchmark_close=benchmark_close,
                test_intervals=test_intervals,
                run_stamp=run_stamp,
                model_weights_dir=ranker_model_weights_dir,
                use_model_cache=not args.no_model_cache,
            )
        )
    ranker_ensemble = ranker_flow.build_ensemble_artifacts(
        ranker_runs,
        weights=ranker_weights,
        raw=raw,
        config=config,
        engine=resolved.engine,
        run_stamp=run_stamp,
    )

    if args.hybrid_method == "classifier_ranker_score_tilt":
        tilt_strengths = parse_float_list(args.ranker_tilt_strength_grid) or [
            float(args.ranker_tilt_strength)
        ]
    else:
        tilt_strengths = [float(args.ranker_tilt_strength)]

    hybrid_candidates = [
        build_hybrid_run(
            classifier_ensemble=classifier_ensemble,
            ranker_ensemble=ranker_ensemble,
            ranker_runs=ranker_runs,
            ranker_weights=ranker_weights,
            raw=raw,
            config=config,
            engine=resolved.engine,
            run_stamp=run_stamp,
            hybrid_method=args.hybrid_method,
            min_classifier_weight=args.min_classifier_weight,
            renormalize_gross=args.renormalize_gross,
            hybrid_ranker_weight=args.hybrid_ranker_weight,
            ranker_tilt_strength=tilt_strength,
        )
        for tilt_strength in tilt_strengths
    ]
    hybrid_run = (
        choose_hybrid_candidate(
            hybrid_candidates,
            classifier_ensemble=classifier_ensemble,
        )
        if args.hybrid_method == "classifier_ranker_score_tilt"
        else hybrid_candidates[0]
    )

    comparison_runs: list[ComparisonRun] = []
    if args.include_separate:
        comparison_runs.extend(comparison_from_classifier_run(run) for run in classifier_runs)
        comparison_runs.extend(comparison_from_ranker_run(run) for run in ranker_runs)
    comparison_runs.extend(
        [
            comparison_from_classifier_run(classifier_ensemble),
            comparison_from_ranker_run(ranker_ensemble),
            hybrid_run,
        ]
    )

    output_paths = {
        run.run_id: export_comparison_artifacts(run, output_dirs=output_dirs)
        for run in comparison_runs
    }
    report_path = save_comparison_report_pdf(
        comparison_runs,
        report_dir=output_dirs.report_dir,
        run_stamp=run_stamp,
        test_intervals=test_intervals,
    )
    metrics_path, effective_yaml_path = save_summary_files(
        comparison_runs,
        base_config_path=args.base_config,
        classifier_specs=classifier_specs,
        ranker_specs=ranker_specs,
        classifier_weights=classifier_weights,
        ranker_weights=ranker_weights,
        output_paths=output_paths,
        output_dirs=output_dirs,
        run_stamp=run_stamp,
        report_path=report_path,
        hybrid_candidates=hybrid_candidates,
    )

    metrics = pd.DataFrame([metric_row(run) for run in comparison_runs])
    display_cols = [
        "family",
        "label",
        "total_return",
        "metric_cagr",
        "metric_sharpe",
        "metric_max_drawdown",
        "metric_sortino",
        "metric_win_rate",
        "avg_gross_exposure",
    ]
    display = metrics[[column for column in display_cols if column in metrics.columns]].copy()
    for column in display.columns:
        if column not in {"family", "label"}:
            display[column] = pd.to_numeric(display[column], errors="coerce").round(4)
    print("\nComparison:")
    print(display.to_string(index=False))
    print(f"Combined report : {report_path}")
    print(f"Metrics CSV     : {metrics_path}")
    print(f"Effective YAML  : {effective_yaml_path}")
    print(f"Classifier outputs: {classifier_output_dirs.yaml_dir}")
    print(f"Ranker outputs    : {ranker_yaml_dir}, {ranker_weights_dir}, {ranker_report_dir}")
    print(f"Ranker model cache: {ranker_model_weights_dir}")


if __name__ == "__main__":
    main()
