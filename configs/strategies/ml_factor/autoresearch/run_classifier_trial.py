#!/usr/bin/env python
"""Run one ML factor classifier autoresearch trial.

The qts_autoresearch helper expects each command target to accept trial flags,
write outputs under --output-dir, and emit metrics.json. This wrapper keeps the
notebook classifier as the fixed backtest target while generating a per-trial
YAML config from bounded parameter overrides.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


METRIC_KEYS = ("sharpe", "sortino", "cagr", "max_drawdown", "win_rate")


def find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not locate repository root from wrapper path.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="trial")
    parser.add_argument("--data-csv", type=Path)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--skip-tearsheet", action="store_true")

    parser.add_argument("--forward-period", type=int)
    parser.add_argument("--classification-window", type=int)
    parser.add_argument("--train-window", type=int)
    parser.add_argument("--rebalance-frequency", type=int)
    parser.add_argument("--max-positions", type=int)
    parser.add_argument("--probability-threshold", type=float)
    parser.add_argument("--min-predicted-class", type=int)
    parser.add_argument("--clear-min-predicted-class", action="store_true")
    parser.add_argument("--fallback-min-positions", type=int)
    parser.add_argument("--rank-col")
    parser.add_argument("--probability-cols")
    parser.add_argument("--probability-sum-col")
    parser.add_argument("--class-quantiles")
    parser.add_argument("--class-scores")
    parser.add_argument(
        "--classification-threshold-scope",
        choices=("classification_window", "train"),
    )
    parser.add_argument("--balanced-sample-weight", action="store_true")
    parser.add_argument("--stop-loss", type=float)
    parser.add_argument("--stop-loss-min-hold", type=int)
    parser.add_argument("--stop-loss-benchmark-below-sma", action="store_true")
    parser.add_argument("--take-profit", type=float)
    parser.add_argument("--trailing-stop", type=float)
    parser.add_argument("--disable-benchmark-regime", action="store_true")
    parser.add_argument("--benchmark-regime-window", type=int)
    parser.add_argument("--benchmark-regime-min-periods", type=int)
    parser.add_argument("--rebalance-existing-positions", action="store_true")

    parser.add_argument("--xgb-n-estimators", type=int)
    parser.add_argument("--xgb-learning-rate", type=float)
    parser.add_argument("--xgb-max-depth", type=int)
    parser.add_argument("--xgb-subsample", type=float)
    parser.add_argument("--xgb-colsample-bytree", type=float)
    parser.add_argument("--xgb-min-child-weight", type=float)
    parser.add_argument("--xgb-reg-alpha", type=float)
    parser.add_argument("--xgb-reg-lambda", type=float)
    parser.add_argument("--xgb-gamma", type=float)
    return parser.parse_args()


def resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return payload


def parse_float_list(raw: str) -> list[float]:
    values = [item.strip() for item in raw.split(",")]
    return [float(item) for item in values if item]


def set_if_present(mapping: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        mapping[key] = value


def build_trial_config(args: argparse.Namespace, repo_root: Path) -> tuple[Path, dict[str, Any]]:
    source_config = resolve(repo_root, args.config)
    payload = read_yaml(source_config)

    payload["run_id"] = f"autoresearch_{args.run_name}"

    run = payload.setdefault("run", {})
    set_if_present(run, "forward_period", args.forward_period)
    set_if_present(run, "classification_window", args.classification_window)
    set_if_present(run, "train_window", args.train_window)
    set_if_present(run, "rebalance_frequency", args.rebalance_frequency)
    set_if_present(run, "max_positions", args.max_positions)

    signal = payload.setdefault("signal", {})
    set_if_present(signal, "probability_threshold", args.probability_threshold)
    set_if_present(signal, "min_predicted_class", args.min_predicted_class)
    if args.clear_min_predicted_class:
        signal.pop("min_predicted_class", None)
    set_if_present(signal, "fallback_min_positions", args.fallback_min_positions)
    set_if_present(signal, "rank_col", args.rank_col)
    if args.probability_cols:
        signal["probability_cols"] = [
            item.strip() for item in args.probability_cols.split(",") if item.strip()
        ]
    set_if_present(signal, "probability_sum_col", args.probability_sum_col)

    classification = payload.setdefault("classification", {})
    if args.class_quantiles:
        classification["class_quantiles"] = parse_float_list(args.class_quantiles)
    if args.class_scores:
        classification["class_scores"] = parse_float_list(args.class_scores)
    set_if_present(classification, "threshold_scope", args.classification_threshold_scope)
    if args.balanced_sample_weight:
        classification["balanced_sample_weight"] = True

    exit_rules = payload.setdefault("backtest", {}).setdefault("exit_rules", {})
    set_if_present(exit_rules, "stop_loss", args.stop_loss)
    set_if_present(exit_rules, "stop_loss_min_hold", args.stop_loss_min_hold)
    if args.stop_loss_benchmark_below_sma:
        exit_rules["stop_loss_benchmark_filter"] = "below_sma"
    set_if_present(exit_rules, "take_profit", args.take_profit)
    set_if_present(exit_rules, "trailing_stop", args.trailing_stop)

    if args.disable_benchmark_regime:
        payload.setdefault("benchmark_regime", {})["enabled"] = False
    benchmark_regime = payload.setdefault("benchmark_regime", {})
    set_if_present(benchmark_regime, "window", args.benchmark_regime_window)
    set_if_present(benchmark_regime, "min_periods", args.benchmark_regime_min_periods)
    if args.rebalance_existing_positions:
        payload.setdefault("backtest", {})["rebalance_existing_positions"] = True

    xgb = payload.setdefault("xgb", {})
    set_if_present(xgb, "n_estimators", args.xgb_n_estimators)
    set_if_present(xgb, "learning_rate", args.xgb_learning_rate)
    set_if_present(xgb, "max_depth", args.xgb_max_depth)
    set_if_present(xgb, "subsample", args.xgb_subsample)
    set_if_present(xgb, "colsample_bytree", args.xgb_colsample_bytree)
    set_if_present(xgb, "min_child_weight", args.xgb_min_child_weight)
    set_if_present(xgb, "reg_alpha", args.xgb_reg_alpha)
    set_if_present(xgb, "reg_lambda", args.xgb_reg_lambda)
    set_if_present(xgb, "gamma", args.xgb_gamma)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trial_config = args.output_dir / "trial_config.yaml"
    trial_config.write_text(yaml.safe_dump(payload, sort_keys=False))
    return trial_config, payload


def read_qts_metrics(path: Path) -> dict[str, float]:
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"{path} is empty.")

    metrics: dict[str, float] = {}
    for row in rows[1:]:
        if len(row) < 2:
            continue
        key, value = row[0], row[1]
        if key in METRIC_KEYS:
            metrics[key] = float(value)

    missing = [key for key in METRIC_KEYS if key not in metrics]
    if missing:
        raise ValueError(f"{path} is missing metrics: {missing}")
    return metrics


def main() -> int:
    args = parse_args()
    repo_root = find_repo_root()
    output_dir = resolve(repo_root, args.output_dir)
    args.output_dir = output_dir

    trial_config, _ = build_trial_config(args, repo_root)
    classifier = repo_root / "notebooks" / "ml_factor_classifier.py"
    command = [
        sys.executable,
        str(classifier),
        "--config",
        str(trial_config),
        "--output-dir",
        str(output_dir),
    ]
    if args.data_csv is not None:
        command.extend(["--data-csv", str(resolve(repo_root, args.data_csv))])
    if args.database is not None:
        command.extend(["--database", str(resolve(repo_root, args.database))])
    if args.skip_tearsheet:
        command.append("--skip-tearsheet")

    result = subprocess.run(
        command,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(result.stdout, end="")
    if result.returncode != 0:
        return result.returncode

    metrics = read_qts_metrics(output_dir / "qts_metrics.csv")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
