from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "stock_ml_e2e.py"
RUNTIME_DIR = REPO_ROOT / ".qts_notebooke_runtime"
LEDGER_PATH = RUNTIME_DIR / "alpha_sweep_results.tsv"
EVENTS_PATH = RUNTIME_DIR / "alpha_sweep_events.jsonl"
RANKING_PATH = RUNTIME_DIR / "alpha_sweep_ranking.md"
TARGET_SHARPE = 1.2
TARGET_MAX_DRAWDOWN = 0.35


@dataclass(frozen=True)
class TrialSpec:
    trial_id: int
    feature_set: str
    feature_groups: tuple[str, ...]
    forward_period: int
    rebalance: int
    train_window: int
    top_positions: int
    n_estimators: int
    max_depth: int
    learning_rate: float = 0.1

    @property
    def run_id(self) -> str:
        groups = "_".join(group for group in self.feature_groups if group != "baseline")
        group_part = groups or "baseline"
        return (
            f"sweep_{self.trial_id:04d}_{group_part}_"
            f"fwd{self.forward_period}_rb{self.rebalance}_"
            f"tw{self.train_window}_top{self.top_positions}_"
            f"n{self.n_estimators}_d{self.max_depth}"
        )


def load_notebook_module() -> Any:
    spec = importlib.util.spec_from_file_location("stock_ml_e2e_sweep", NOTEBOOK_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {NOTEBOOK_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def unique_specs(specs: list[TrialSpec]) -> list[TrialSpec]:
    seen: set[tuple[Any, ...]] = set()
    out: list[TrialSpec] = []
    next_id = 1
    for spec in specs:
        key = (
            spec.feature_groups,
            spec.forward_period,
            spec.rebalance,
            spec.train_window,
            spec.top_positions,
            spec.n_estimators,
            spec.max_depth,
            spec.learning_rate,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(
            TrialSpec(
                trial_id=next_id,
                feature_set=spec.feature_set,
                feature_groups=spec.feature_groups,
                forward_period=spec.forward_period,
                rebalance=spec.rebalance,
                train_window=spec.train_window,
                top_positions=spec.top_positions,
                n_estimators=spec.n_estimators,
                max_depth=spec.max_depth,
                learning_rate=spec.learning_rate,
            )
        )
        next_id += 1
    return out


def candidate_specs(module: Any) -> list[TrialSpec]:
    feature_sets: dict[str, tuple[str, ...]] = {
        "baseline": ("baseline",),
        "volume_ratio": ("baseline", "volume_ratio"),
        "roc": ("baseline", "roc"),
        "log": ("baseline", "log"),
        "ma_5": ("baseline", "ma_5"),
        "adx_14": ("baseline", "adx_14"),
        "zscore_10": ("baseline", "zscore_10"),
        "zscore_21": ("baseline", "zscore_21"),
        "volume_ratio_ma5": ("baseline", "volume_ratio", "ma_5"),
        "volume_ratio_roc": ("baseline", "volume_ratio", "roc"),
        "log_roc": ("baseline", "log", "roc"),
        "adx14_volume_ratio": ("baseline", "adx_14", "volume_ratio"),
    }
    specs: list[TrialSpec] = []

    def add(
        feature_set: str,
        *,
        fwd: int,
        rb: int,
        train: int,
        top: int,
        n_estimators: int = 5,
        depth: int = 3,
    ) -> None:
        specs.append(
            TrialSpec(
                trial_id=0,
                feature_set=feature_set,
                feature_groups=feature_sets[feature_set],
                forward_period=fwd,
                rebalance=rb,
                train_window=train,
                top_positions=top,
                n_estimators=n_estimators,
                max_depth=depth,
            )
        )

    for top in [3, 4, 5, 7, 10]:
        add("baseline", fwd=21, rb=12, train=504, top=top)

    for n_estimators in [3, 5, 8, 12, 20]:
        for depth in [2, 3, 4]:
            for top in [5, 7, 10]:
                add(
                    "baseline",
                    fwd=21,
                    rb=12,
                    train=504,
                    top=top,
                    n_estimators=n_estimators,
                    depth=depth,
                )

    for fwd in [21, 24, 30]:
        for rb in [12, 15, 18]:
            for train in [504, 756]:
                for top in [5, 7, 10]:
                    add("baseline", fwd=fwd, rb=rb, train=train, top=top)

    for feature_set in [
        "volume_ratio",
        "roc",
        "log",
        "ma_5",
        "adx_14",
        "zscore_10",
        "zscore_21",
        "volume_ratio_ma5",
        "volume_ratio_roc",
        "log_roc",
        "adx14_volume_ratio",
    ]:
        for fwd in [21, 24, 30]:
            for rb in [12, 15, 18]:
                for train in [504, 756]:
                    for top in [5, 7, 10]:
                        add(feature_set, fwd=fwd, rb=rb, train=train, top=top)

    for fwd in [12, 15, 18, 21, 24, 30, 42]:
        for rb in [6, 8, 10, 12, 15, 21]:
            for top in [1, 2, 3, 5]:
                add("baseline", fwd=fwd, rb=rb, train=504, top=top)

    for train in [120, 180, 252, 378, 504, 756, 1008]:
        for fwd in [18, 21, 24, 30]:
            for rb in [8, 10, 12, 15]:
                for top in [2, 3, 5]:
                    add("baseline", fwd=fwd, rb=rb, train=train, top=top)

    for feature_set in feature_sets:
        if feature_set == "baseline":
            continue
        for fwd in [18, 21, 24, 30]:
            for rb in [8, 10, 12, 15]:
                for train in [252, 504, 756]:
                    for top in [2, 3, 5]:
                        add(feature_set, fwd=fwd, rb=rb, train=train, top=top)

    for n_estimators in [3, 5, 8, 12]:
        for depth in [2, 3, 4]:
            for feature_set in ["baseline", "volume_ratio", "roc", "log"]:
                for top in [2, 3, 5]:
                    add(
                        feature_set,
                        fwd=21,
                        rb=12,
                        train=504,
                        top=top,
                        n_estimators=n_estimators,
                        depth=depth,
                    )

    return unique_specs(specs)


def focused_candidate_specs() -> list[TrialSpec]:
    feature_sets: dict[str, tuple[str, ...]] = {
        "baseline": ("baseline",),
        "volume_ratio": ("baseline", "volume_ratio"),
        "roc": ("baseline", "roc"),
        "log": ("baseline", "log"),
        "ma_5": ("baseline", "ma_5"),
        "adx_14": ("baseline", "adx_14"),
        "zscore_10": ("baseline", "zscore_10"),
        "zscore_21": ("baseline", "zscore_21"),
        "volume_ratio_ma5": ("baseline", "volume_ratio", "ma_5"),
        "volume_ratio_roc": ("baseline", "volume_ratio", "roc"),
        "log_roc": ("baseline", "log", "roc"),
        "adx14_volume_ratio": ("baseline", "adx_14", "volume_ratio"),
    }
    specs: list[TrialSpec] = []

    def add(
        feature_set: str,
        *,
        fwd: int,
        rb: int,
        train: int,
        top: int,
        n_estimators: int,
        depth: int,
        learning_rate: float = 0.1,
    ) -> None:
        specs.append(
            TrialSpec(
                trial_id=0,
                feature_set=feature_set,
                feature_groups=feature_sets[feature_set],
                forward_period=fwd,
                rebalance=rb,
                train_window=train,
                top_positions=top,
                n_estimators=n_estimators,
                max_depth=depth,
                learning_rate=learning_rate,
            )
        )

    # Seed from the current best broad sweep:
    # fwd=30, rebalance=15, train_window=504, top=5, n_estimators=5, depth=3.
    for fwd in [30, 27, 33, 24, 36]:
        for rb in [15, 12, 18]:
            for train in [504, 630]:
                for top in [4, 5, 6, 7]:
                    add("baseline", fwd=fwd, rb=rb, train=train, top=top, n_estimators=5, depth=3)

    for feature_set in [
        "volume_ratio",
        "roc",
        "log",
        "ma_5",
        "adx_14",
        "zscore_10",
        "zscore_21",
        "volume_ratio_ma5",
        "volume_ratio_roc",
        "log_roc",
        "adx14_volume_ratio",
    ]:
        for fwd, rb in [(30, 15), (30, 12), (33, 15), (27, 15), (24, 12)]:
            for train in [504, 630]:
                for top in [4, 5, 6, 7]:
                    for n_estimators, depth in [(5, 3), (8, 2)]:
                        add(
                            feature_set,
                            fwd=fwd,
                            rb=rb,
                            train=train,
                            top=top,
                            n_estimators=n_estimators,
                            depth=depth,
                        )

    for n_estimators in [4, 5, 6, 8, 10, 12]:
        for depth in [2, 3, 4]:
            for top in [4, 5, 6, 7]:
                for fwd, rb in [(30, 15), (30, 12), (33, 15), (27, 15)]:
                    add(
                        "baseline",
                        fwd=fwd,
                        rb=rb,
                        train=504,
                        top=top,
                        n_estimators=n_estimators,
                        depth=depth,
                    )

    return unique_specs(specs)


def predictor_cols(module: Any, groups: tuple[str, ...]) -> list[str]:
    return [
        col
        for group in groups
        for col in module.FEATURE_GROUP_COLUMNS[group]
    ]


def metric(metrics: dict[str, float], key: str) -> float:
    value = metrics.get(key, np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        path,
        sep="\t",
        mode="a",
        header=not path.exists(),
        index=False,
    )


def append_event(event: dict[str, Any]) -> None:
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True, default=str) + "\n")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_ranking() -> None:
    if not LEDGER_PATH.exists():
        return
    rows = pd.read_csv(LEDGER_PATH, sep="\t")
    if rows.empty:
        return
    ok = rows.loc[rows["status"] == "ok"].copy()
    if ok.empty:
        return
    for col in [
        "full_sharpe",
        "full_max_drawdown",
        "active_sharpe",
        "active_max_drawdown",
        "top_mean_forward_return",
    ]:
        ok[col] = pd.to_numeric(ok[col], errors="coerce")
    ok["criteria_met"] = (
        (ok["full_sharpe"] > TARGET_SHARPE)
        & (ok["full_max_drawdown"] < TARGET_MAX_DRAWDOWN)
    )
    ok = ok.sort_values(
        [
            "criteria_met",
            "full_sharpe",
            "full_max_drawdown",
            "active_sharpe",
            "top_mean_forward_return",
        ],
        ascending=[False, False, True, False, False],
    ).reset_index(drop=True)
    table_rows = []
    for i, row in enumerate(ok.head(25).itertuples(index=False), start=1):
        table_rows.append(
            [
                i,
                row.run_id,
                row.feature_set,
                row.forward_period,
                row.rebalance,
                row.train_window,
                row.top_positions,
                row.n_estimators,
                row.max_depth,
                row.full_sharpe,
                row.full_max_drawdown,
                row.active_sharpe,
                row.active_max_drawdown,
                row.top_mean_forward_return,
                "hit" if row.criteria_met else "open",
            ]
        )
    lines = [
        "# Alpha sweep ranking",
        "",
        f"Target: full Sharpe > {TARGET_SHARPE} and full max drawdown < {TARGET_MAX_DRAWDOWN}.",
        "",
        markdown_table(
            [
                "Rank",
                "Run ID",
                "Feature Set",
                "Fwd",
                "Rb",
                "Train",
                "Top",
                "Trees",
                "Depth",
                "Full Sharpe",
                "Full MDD",
                "Active Sharpe",
                "Active MDD",
                "Top Mean Fwd",
                "Status",
            ],
            table_rows,
        ),
    ]
    RANKING_PATH.write_text("\n".join(lines), encoding="utf-8")


def active_metrics(module: Any, result: Any, active_start: Any) -> dict[str, float]:
    return module._active_metrics(result, active_start)


def prepare_datasets(module: Any, raw: Any, forward_periods: set[int]) -> dict[int, pd.DataFrame]:
    all_cols = sorted(
        {
            col
            for group in module.FEATURE_GROUP_COLUMNS
            for col in module.FEATURE_GROUP_COLUMNS[group]
        }
    )
    datasets: dict[int, pd.DataFrame] = {}
    for fwd in sorted(forward_periods):
        print(f"FEATURE_BUILD fwd={fwd}", flush=True)
        featured = module.build_reference_feature_frame(raw, forward_period=fwd)
        target_col = f"Return_fwd_{fwd}"
        datasets[fwd] = module.prepare_dataset(
            featured,
            target_col=target_col,
            predictor_cols=all_cols,
        )
    return datasets


def run_trial(
    module: Any,
    raw: Any,
    engine: Any,
    config: Any,
    sessions: list[Any],
    dataset: pd.DataFrame,
    spec: TrialSpec,
) -> tuple[dict[str, Any], Any, pd.DataFrame, dict[str, float]]:
    cols = predictor_cols(module, spec.feature_groups)
    target_col = f"Return_fwd_{spec.forward_period}"
    predictions = module.walk_forward_rank(
        frame=dataset,
        predictor=module.XGBRankerPredictor(
            n_estimators=spec.n_estimators,
            learning_rate=spec.learning_rate,
            max_depth=spec.max_depth,
            n_jobs=-1,
        ),
        predictor_cols=cols,
        target_col=target_col,
        train_window=spec.train_window,
        rebalance_frequency=spec.rebalance,
        target_horizon=spec.forward_period,
        prediction_start_date=pd.Timestamp(config.test_start_date)
        if config.test_start_date
        else None,
    )
    eval_summary = module.summarize_topk_predictions(predictions, spec.top_positions)
    strategy = module.XGBRankerTopKStrategy(
        predictions=predictions,
        sessions=sessions,
        top_n=spec.top_positions,
    )
    result = engine.run(strategy=strategy, data=raw, config=config)
    active = active_metrics(module, result, config.test_start_date)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": "ok",
        "run_id": spec.run_id,
        "trial_id": spec.trial_id,
        "feature_set": spec.feature_set,
        "feature_groups": ",".join(spec.feature_groups),
        "predictor_count": len(cols),
        "predictors": ",".join(cols),
        "forward_period": spec.forward_period,
        "rebalance": spec.rebalance,
        "train_window": spec.train_window,
        "top_positions": spec.top_positions,
        "n_estimators": spec.n_estimators,
        "max_depth": spec.max_depth,
        "learning_rate": spec.learning_rate,
        "full_sharpe": metric(result.metrics, "sharpe"),
        "full_sortino": metric(result.metrics, "sortino"),
        "full_cagr": metric(result.metrics, "cagr"),
        "full_max_drawdown": metric(result.metrics, "max_drawdown"),
        "full_win_rate": metric(result.metrics, "win_rate"),
        "active_sharpe": metric(active, "sharpe"),
        "active_sortino": metric(active, "sortino"),
        "active_cagr": metric(active, "cagr"),
        "active_max_drawdown": metric(active, "max_drawdown"),
        "active_win_rate": metric(active, "win_rate"),
        "prediction_rows": metric(eval_summary, "prediction_rows"),
        "prediction_dates": metric(eval_summary, "prediction_dates"),
        "top_mean_forward_return": metric(eval_summary, "top_mean_forward_return"),
    }
    row["criteria_met"] = (
        row["full_sharpe"] > TARGET_SHARPE
        and row["full_max_drawdown"] < TARGET_MAX_DRAWDOWN
    )
    return row, result, predictions, eval_summary


def export_best(
    module: Any,
    row: dict[str, Any],
    result: Any,
    predictions: pd.DataFrame,
    eval_summary: dict[str, float],
) -> None:
    run_id = row["run_id"]
    module.export_outputs(
        run_id=run_id,
        predictions=predictions,
        result=result,
        benchmark_rets=None,
    )
    config = module.Config.build(str(module.CONFIG_PATH)).raw
    active_stats = module._active_metrics(result, config.test_start_date)
    report = "\n".join(
        [
            f"# Alpha sweep best: {run_id}",
            "",
            markdown_table(
                ["Item", "Value"],
                [
                    ["feature_set", row["feature_set"]],
                    ["feature_groups", row["feature_groups"]],
                    ["forward_period", row["forward_period"]],
                    ["rebalance", row["rebalance"]],
                    ["train_window", row["train_window"]],
                    ["top_positions", row["top_positions"]],
                    ["n_estimators", row["n_estimators"]],
                    ["max_depth", row["max_depth"]],
                ],
            ),
            "",
            "## Full Metrics",
            markdown_table(["Metric", "Value"], [[k, v] for k, v in result.metrics.items()]),
            "",
            "## Active Metrics",
            markdown_table(["Metric", "Value"], [[k, v] for k, v in active_stats.items()]),
            "",
            "## Ranker Eval",
            markdown_table(["Metric", "Value"], [[k, v] for k, v in eval_summary.items()]),
        ]
    )
    (RUNTIME_DIR / f"{run_id}_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-trials", type=int, default=250)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--focused", action="store_true")
    args = parser.parse_args()

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    if args.reset:
        for path in [LEDGER_PATH, EVENTS_PATH, RANKING_PATH]:
            path.unlink(missing_ok=True)

    module = load_notebook_module()
    specs = (
        focused_candidate_specs()
        if args.focused
        else candidate_specs(module)
    )[: args.max_trials]
    append_event(
        {
            "event": "start",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "max_trials": args.max_trials,
            "candidate_count": len(specs),
        }
    )

    resolved = module.Config.build(str(module.CONFIG_PATH))
    config = resolved.raw
    module.validate_vn_ranker_config(config)
    data_manager = module.build_data_manager(resolved)
    raw = asyncio.run(
        data_manager.get_ohlcv(
            symbols=module.config_symbols(config),
            start=config.start_date,
            end=config.end_date,
        )
    )
    sessions = raw["date"].unique().to_list()
    engine = resolved.engine
    datasets = prepare_datasets(module, raw, {spec.forward_period for spec in specs})

    best_row: dict[str, Any] | None = None
    best_result = None
    best_predictions = None
    best_eval: dict[str, float] | None = None
    hit = False

    for spec in specs:
        print(
            "TRIAL "
            f"{spec.trial_id}/{len(specs)} {spec.run_id} "
            f"features={spec.feature_set}",
            flush=True,
        )
        try:
            row, result, predictions, eval_summary = run_trial(
                module,
                raw,
                engine,
                config,
                sessions,
                datasets[spec.forward_period],
                spec,
            )
        except Exception as exc:
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "status": "error",
                "run_id": spec.run_id,
                "trial_id": spec.trial_id,
                "feature_set": spec.feature_set,
                "feature_groups": ",".join(spec.feature_groups),
                "forward_period": spec.forward_period,
                "rebalance": spec.rebalance,
                "train_window": spec.train_window,
                "top_positions": spec.top_positions,
                "n_estimators": spec.n_estimators,
                "max_depth": spec.max_depth,
                "learning_rate": spec.learning_rate,
                "error": repr(exc),
            }
            append_row(LEDGER_PATH, row)
            append_event(
                {
                    "event": "trial_error",
                    "run_id": spec.run_id,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            continue

        append_row(LEDGER_PATH, row)
        if best_row is None or row["full_sharpe"] > best_row["full_sharpe"]:
            best_row = row
            best_result = result
            best_predictions = predictions
            best_eval = eval_summary
            print(
                "BEST "
                f"sharpe={row['full_sharpe']:.4f} "
                f"mdd={row['full_max_drawdown']:.4f} "
                f"run={row['run_id']}",
                flush=True,
            )
        write_ranking()
        if row["criteria_met"]:
            print(
                "TARGET_HIT "
                f"sharpe={row['full_sharpe']:.4f} "
                f"mdd={row['full_max_drawdown']:.4f} "
                f"run={row['run_id']}",
                flush=True,
            )
            best_row = row
            best_result = result
            best_predictions = predictions
            best_eval = eval_summary
            hit = True
            break

    if best_row is not None and best_result is not None and best_predictions is not None:
        export_best(module, best_row, best_result, best_predictions, best_eval or {})
    write_ranking()
    append_event(
        {
            "event": "finish",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "hit": hit,
            "best": best_row,
        }
    )
    print(f"LEDGER {LEDGER_PATH}")
    print(f"RANKING {RANKING_PATH}")
    return 0 if hit else 2


if __name__ == "__main__":
    raise SystemExit(main())
