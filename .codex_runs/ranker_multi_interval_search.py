from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "stock_ml_e2e.py"
LEDGER = ROOT / ".qts_notebooke_runtime" / "ranker_multi_interval_top5_search.tsv"
HIT_JSON = ROOT / ".qts_notebooke_runtime" / "ranker_multi_interval_top5_hit.json"

os.environ.setdefault("QTS_ROOT", str(ROOT / ".qts_notebooke_runtime" / "isolated_qts_root"))
sys.path.insert(0, str(ROOT))


def load_notebook_module() -> Any:
    spec = importlib.util.spec_from_file_location("stock_ml_e2e_ranker", NOTEBOOK_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {NOTEBOOK_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def predictor_cols(module: Any, groups: list[str]) -> list[str]:
    resolved = ["baseline", *[group for group in groups if group != "baseline"]]
    cols: list[str] = []
    for group in dict.fromkeys(resolved):
        cols.extend(module.FEATURE_GROUP_COLUMNS[group])
    return list(dict.fromkeys(cols))


def experiment_id(groups: list[str]) -> str:
    additions = [group for group in groups if group != "baseline"]
    return "baseline" if not additions else "+".join(additions)


def append_row(row: dict[str, Any]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    frame.to_csv(
        LEDGER,
        sep="\t",
        index=False,
        mode="a",
        header=not LEDGER.exists(),
    )


def metric(metrics: dict[str, float], key: str) -> float:
    try:
        return float(metrics.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def evaluate_predictions(
    *,
    module: Any,
    raw,
    config,
    predictions: pd.DataFrame,
    top_n: int,
    min_top_score: float | None,
    min_score_spread: float | None,
) -> dict[str, Any]:
    strategy = module.XGBRankerTopKStrategy(
        predictions=predictions,
        sessions=raw["date"].unique().to_list(),
        top_n=top_n,
        min_top_score=min_top_score,
        min_score_spread=min_score_spread,
    )
    signals = strategy.generate_signals(raw)
    result = module.Config.build(str(module.CONFIG_PATH)).engine.run(
        strategy=strategy,
        data=raw,
        config=config,
    )
    active_metrics = module._active_metrics(
        result,
        module.test_intervals_from_config(config),
    )
    return {
        "result": result,
        "signals": signals,
        "full_sharpe": metric(result.metrics, "sharpe"),
        "full_max_drawdown": metric(result.metrics, "max_drawdown"),
        "full_cagr": metric(result.metrics, "cagr"),
        "full_sortino": metric(result.metrics, "sortino"),
        "active_sharpe": metric(active_metrics, "sharpe"),
        "active_max_drawdown": metric(active_metrics, "max_drawdown"),
        "active_cagr": metric(active_metrics, "cagr"),
    }


def score_filter_grid(
    predictions: pd.DataFrame,
    top_n: int,
) -> list[tuple[float | None, float | None]]:
    if predictions.empty:
        return [(None, None)]
    rows: list[dict[str, float]] = []
    for _, group in predictions.groupby("date", sort=True):
        ranked = group.sort_values(["score", "symbol"], ascending=[False, True]).head(top_n)
        if ranked.empty:
            continue
        top_score = float(ranked["score"].iloc[0])
        last_score = float(ranked["score"].iloc[-1]) if len(ranked) >= top_n else top_score
        rows.append({"top_score": top_score, "spread": top_score - last_score})
    if not rows:
        return [(None, None)]
    stats = pd.DataFrame(rows)
    top_thresholds: list[float | None] = [None]
    spread_thresholds: list[float | None] = [None]
    for quantile in [0.15, 0.25, 0.35, 0.50, 0.65]:
        top_thresholds.append(float(stats["top_score"].quantile(quantile)))
    for quantile in [0.15, 0.25, 0.35, 0.50, 0.65]:
        spread_thresholds.append(float(stats["spread"].quantile(quantile)))
    grid = [
        (top_score, spread)
        for top_score in list(dict.fromkeys(top_thresholds))
        for spread in list(dict.fromkeys(spread_thresholds))
    ]
    return grid


async def main() -> None:
    module = load_notebook_module()
    resolved = module.Config.build(str(module.CONFIG_PATH))
    config = resolved.raw
    module.validate_vn_ranker_config(config)

    symbols = module.config_symbols(config)
    data_manager = module.build_data_manager(resolved)
    raw = await data_manager.get_ohlcv(
        symbols=symbols,
        start=config.start_date,
        end=config.end_date,
    )
    test_intervals = module.test_intervals_from_config(config)
    sessions = raw["date"].unique().to_list()
    print(
        "Loaded data",
        f"rows={raw.height}",
        f"sessions={len(sessions)}",
        f"intervals={module.format_date_intervals(test_intervals)}",
        flush=True,
    )

    feature_sets = [
        ["baseline"],
        ["baseline", "log"],
        ["baseline", "hist_vol"],
        ["baseline", "zscore"],
        ["baseline", "volume_ratio"],
        ["baseline", "adx"],
        ["baseline", "roc"],
        ["baseline", "rsi"],
        ["baseline", "ma_short"],
        ["baseline", "log", "hist_vol"],
        ["baseline", "log", "zscore"],
        ["baseline", "hist_vol", "zscore"],
        ["baseline", "adx", "zscore"],
    ]
    targeted_models = [
        (["baseline"], fwd, rb, tw, n_estimators, max_depth, learning_rate)
        for fwd, rb, tw, n_estimators, max_depth, learning_rate in [
            (30, 12, 630, 5, 3, 0.10),
            (30, 12, 504, 5, 3, 0.10),
            (30, 12, 756, 5, 3, 0.10),
            (30, 10, 630, 5, 3, 0.10),
            (30, 15, 630, 5, 3, 0.10),
            (30, 18, 756, 5, 3, 0.10),
            (21, 10, 756, 5, 3, 0.10),
            (21, 10, 630, 5, 3, 0.10),
            (21, 12, 756, 5, 3, 0.10),
            (21, 18, 756, 5, 3, 0.10),
            (42, 12, 630, 5, 3, 0.10),
            (42, 18, 756, 5, 3, 0.10),
            (30, 12, 630, 10, 2, 0.05),
            (30, 12, 630, 10, 3, 0.05),
            (30, 12, 630, 20, 2, 0.05),
            (21, 10, 756, 10, 2, 0.05),
            (21, 10, 756, 20, 2, 0.05),
        ]
    ]
    targeted_feature_models = [
        (groups, fwd, rb, tw, 5, 3, 0.10)
        for groups, fwd, rb, tw in product(
            [
                ["baseline", "log"],
                ["baseline", "hist_vol"],
                ["baseline", "zscore"],
                ["baseline", "adx"],
                ["baseline", "log", "hist_vol"],
                ["baseline", "log", "zscore"],
                ["baseline", "hist_vol", "zscore"],
            ],
            [21, 30],
            [10, 12, 18],
            [630, 756],
        )
    ]
    primary_models = [
        # Baseline first, with max_positions fixed downstream at top_n=5.
        (["baseline"], fwd, rb, tw, 5, 3, 0.10)
        for fwd, rb, tw in product(
            [18, 21, 30, 42],
            [5, 8, 10, 12, 15, 18, 21],
            [504, 630, 756, 1008],
        )
    ]
    feature_models = [
        (groups, fwd, rb, tw, 5, 3, 0.10)
        for groups, fwd, rb, tw in product(
            feature_sets[1:],
            [21, 30],
            [10, 12, 18],
            [756, 1008],
        )
    ]
    xgb_local_models = [
        (["baseline"], fwd, rb, tw, n_estimators, max_depth, learning_rate)
        for fwd, rb, tw, n_estimators, max_depth, learning_rate in product(
            [21, 30],
            [10, 12, 18],
            [756, 1008],
            [5, 10, 20],
            [2, 3, 4],
            [0.05, 0.10, 0.15],
        )
    ]
    candidates = [
        *targeted_models,
        *targeted_feature_models,
        *primary_models,
        *feature_models,
        *xgb_local_models,
    ]
    seen: set[tuple[Any, ...]] = set()
    unique_candidates: list[tuple[Any, ...]] = []
    for candidate in candidates:
        key = (tuple(candidate[0]), *candidate[1:])
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    data_cache: dict[int, pd.DataFrame] = {}
    best: dict[str, Any] | None = None
    trial = 0
    for groups, fwd, rb, tw, n_estimators, max_depth, learning_rate in unique_candidates:
        trial += 1
        cols = predictor_cols(module, groups)
        target_col = f"Return_fwd_{fwd}"
        try:
            if fwd not in data_cache:
                featured = module.build_reference_feature_frame(raw, forward_period=fwd)
                data_cache[fwd] = module.prepare_dataset(
                    featured,
                    target_col=target_col,
                    predictor_cols=list(
                        dict.fromkeys(
                            col
                            for feature_groups in feature_sets
                            for col in predictor_cols(module, feature_groups)
                        )
                    ),
                )
            dataset = data_cache[fwd][["date", "symbol", target_col, *cols]].copy()
            predictions = module.walk_forward_rank(
                frame=dataset,
                predictor=module.XGBRankerPredictor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    n_jobs=-1,
                ),
                predictor_cols=cols,
                target_col=target_col,
                train_window=tw,
                rebalance_frequency=rb,
                target_horizon=fwd,
                test_intervals=test_intervals,
            )
            eval_summary = module.summarize_topk_predictions(predictions, 4)
        except Exception as exc:
            append_row(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "trial": trial,
                    "status": "error",
                    "error": repr(exc),
                    "feature_groups": ",".join(groups),
                    "forward_period": fwd,
                    "rebalance": rb,
                    "train_window": tw,
                    "xgb_n_estimators": n_estimators,
                    "xgb_max_depth": max_depth,
                    "xgb_learning_rate": learning_rate,
                }
            )
            print("ERROR", trial, groups, fwd, rb, tw, repr(exc), flush=True)
            continue

        for top_n in [5]:
            filter_grid = score_filter_grid(predictions, top_n)
            for min_top_score, min_score_spread in filter_grid:
                evaluated = evaluate_predictions(
                    module=module,
                    raw=raw,
                    config=config,
                    predictions=predictions,
                    top_n=top_n,
                    min_top_score=min_top_score,
                    min_score_spread=min_score_spread,
                )
                hit = (
                    evaluated["full_sharpe"] > 1.1
                    and evaluated["full_max_drawdown"] < 0.25
                )
                row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "trial": trial,
                    "status": "ok",
                    "experiment_id": experiment_id(groups),
                    "feature_groups": ",".join(groups),
                    "predictor_count": len(cols),
                    "forward_period": fwd,
                    "rebalance": rb,
                    "train_window": tw,
                    "top_positions": top_n,
                    "min_top_score": min_top_score,
                    "min_score_spread": min_score_spread,
                    "xgb_n_estimators": n_estimators,
                    "xgb_max_depth": max_depth,
                    "xgb_learning_rate": learning_rate,
                    "prediction_rows": len(predictions),
                    "prediction_dates": (
                        predictions["date"].nunique() if not predictions.empty else 0
                    ),
                    "top_mean_forward_return": eval_summary.get("top_mean_forward_return"),
                    "full_sharpe": evaluated["full_sharpe"],
                    "full_max_drawdown": evaluated["full_max_drawdown"],
                    "full_cagr": evaluated["full_cagr"],
                    "full_sortino": evaluated["full_sortino"],
                    "active_sharpe": evaluated["active_sharpe"],
                    "active_max_drawdown": evaluated["active_max_drawdown"],
                    "active_cagr": evaluated["active_cagr"],
                    "signal_rows": evaluated["signals"].height,
                    "hit": hit,
                }
                append_row(row)
                if best is None or (
                    row["full_max_drawdown"] < 0.25
                    and row["full_sharpe"] > best["full_sharpe"]
                ):
                    best = row
                print(
                    "TRIAL",
                    trial,
                    "top",
                    top_n,
                    row["feature_groups"],
                    f"fwd={fwd}",
                    f"rb={rb}",
                    f"tw={tw}",
                    f"n={n_estimators}",
                    f"d={max_depth}",
                    f"lr={learning_rate}",
                    f"min_top={min_top_score}",
                    f"min_spread={min_score_spread}",
                    f"signals={row['signal_rows']}",
                    f"sharpe={row['full_sharpe']:.4f}",
                    f"mdd={row['full_max_drawdown']:.4f}",
                    "HIT" if hit else "",
                    flush=True,
                )
                if hit:
                    HIT_JSON.write_text(json.dumps(row, indent=2), encoding="utf-8")
                    print(f"HIT_JSON={HIT_JSON}", flush=True)
                    return
    if best is not None:
        HIT_JSON.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"NO_HIT best={best}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
