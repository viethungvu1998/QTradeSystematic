from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QTS_ROOT = ROOT / ".qts_notebooke_runtime" / "canonical_qts_root"
LEDGER = ROOT / ".qts_notebooke_runtime" / "ranker_canonical_sweep.tsv"
ENSEMBLE_LEDGER = ROOT / ".qts_notebooke_runtime" / "ranker_canonical_ensemble_sweep.tsv"
HIT_YAML = ROOT / ".qts_notebooke_runtime" / "ranker_canonical_sweep_hit.yaml"

os.environ.setdefault("QTS_ROOT", str(DEFAULT_QTS_ROOT))
for path in (ROOT, ROOT / "notebooks"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import combine_ml_e2e_ranking as flow  # noqa: E402


@dataclass(frozen=True, slots=True)
class ModelConfig:
    feature_groups: tuple[str, ...]
    forward_period: int
    rebalance_frequency: int
    train_window: int
    n_estimators: int
    max_depth: int
    learning_rate: float


@dataclass(frozen=True, slots=True)
class SelectorConfig:
    top_positions: int
    min_top_score: float | None
    min_score_spread: float | None
    regime_window: int | None


@dataclass(slots=True)
class Candidate:
    candidate_id: str
    model: ModelConfig
    selector: SelectorConfig
    run: flow.RunArtifacts
    row: dict[str, Any]


def predictor_cols(groups: tuple[str, ...]) -> list[str]:
    cols: list[str] = []
    for group in dict.fromkeys(groups):
        if group not in flow.FEATURE_GROUP_COLUMNS:
            allowed = ", ".join(sorted(flow.FEATURE_GROUP_COLUMNS))
            raise ValueError(f"Unknown feature group {group!r}; allowed: {allowed}")
        cols.extend(str(col) for col in flow.FEATURE_GROUP_COLUMNS[group])
    return list(dict.fromkeys(cols))


def append_tsv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        path,
        sep="\t",
        index=False,
        mode="a",
        header=not path.exists(),
    )


def metric(metrics: dict[str, float], key: str) -> float:
    try:
        return float(metrics.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def model_id(config: ModelConfig) -> str:
    groups = "+".join(config.feature_groups)
    return (
        f"{groups}_fwd{config.forward_period}_rb{config.rebalance_frequency}_"
        f"tw{config.train_window}_n{config.n_estimators}_d{config.max_depth}_"
        f"lr{config.learning_rate:g}"
    )


def selector_id(selector: SelectorConfig) -> str:
    top_score = "none" if selector.min_top_score is None else f"{selector.min_top_score:.5f}"
    spread = "none" if selector.min_score_spread is None else f"{selector.min_score_spread:.5f}"
    regime = "none" if selector.regime_window is None else f"sma{selector.regime_window}"
    return f"top{selector.top_positions}_score{top_score}_spread{spread}_{regime}"


def candidate_id(model: ModelConfig, selector: SelectorConfig) -> str:
    return f"{model_id(model)}_{selector_id(selector)}"


def score_thresholds(predictions: pd.DataFrame, top_n: int) -> list[float | None]:
    if predictions.empty:
        return [None]
    tops: list[float] = []
    for _, group in predictions.groupby("date", sort=True):
        selected = group.sort_values(["score", "symbol"], ascending=[False, True]).head(top_n)
        if not selected.empty:
            tops.append(float(selected["score"].iloc[0]))
    if not tops:
        return [None]
    series = pd.Series(tops, dtype=float)
    thresholds: list[float | None] = [None]
    for quantile in [0.20, 0.35, 0.50, 0.65]:
        value = float(series.quantile(quantile))
        thresholds.append(round(value, 6))
    return list(dict.fromkeys(thresholds))


def spread_thresholds(predictions: pd.DataFrame, top_n: int) -> list[float | None]:
    if predictions.empty or top_n <= 1:
        return [None]
    spreads: list[float] = []
    for _, group in predictions.groupby("date", sort=True):
        selected = group.sort_values(["score", "symbol"], ascending=[False, True]).head(top_n)
        if len(selected) >= top_n:
            spreads.append(float(selected["score"].iloc[0] - selected["score"].iloc[-1]))
    if not spreads:
        return [None]
    series = pd.Series(spreads, dtype=float)
    thresholds: list[float | None] = [None]
    for quantile in [0.25, 0.50]:
        thresholds.append(round(float(series.quantile(quantile)), 6))
    return list(dict.fromkeys(thresholds))


def regime_cfg(window: int | None) -> dict[str, Any]:
    if window is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "rule": "above_sma",
        "window": int(window),
        "min_periods": None,
    }


def evaluate_selector(
    *,
    raw,
    config,
    engine,
    benchmark_close: pd.DataFrame | None,
    predictions: pd.DataFrame,
    model: ModelConfig,
    selector: SelectorConfig,
    run_stamp: str,
) -> Candidate:
    filtered, regime_summary = flow.apply_param_benchmark_regime_filter(
        predictions,
        benchmark_close,
        regime_cfg(selector.regime_window),
    )
    strategy = flow.XGBRankerTopKStrategy(
        predictions=filtered,
        sessions=raw["date"].unique().to_list(),
        top_n=selector.top_positions,
        min_top_score=selector.min_top_score,
        min_score_spread=selector.min_score_spread,
    )
    signals = strategy.generate_signals(raw)
    result = engine.run(strategy=strategy, data=raw, config=config)
    result_signals = result.signals if not result.signals.is_empty() else signals
    targets = flow.target_weights_from_signals(
        result_signals,
        sessions=flow.session_index(raw),
        symbols=flow.config_symbols(config),
    )
    cid = candidate_id(model, selector)
    run = flow.RunArtifacts(
        label=cid,
        run_id=f"sweep_{cid}_{run_stamp}".replace("/", "_"),
        result=result,
        predictions=filtered,
        signals=result_signals,
        target_weights=targets,
        regime_summary=regime_summary,
        eval_summary=flow.summarize_topk_predictions(filtered, selector.top_positions),
        output_paths={},
    )
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": cid,
        "feature_groups": ",".join(model.feature_groups),
        "forward_period": model.forward_period,
        "rebalance_frequency": model.rebalance_frequency,
        "train_window": model.train_window,
        "top_positions": selector.top_positions,
        "min_top_score": selector.min_top_score,
        "min_score_spread": selector.min_score_spread,
        "regime_window": selector.regime_window,
        "xgb_n_estimators": model.n_estimators,
        "xgb_max_depth": model.max_depth,
        "xgb_learning_rate": model.learning_rate,
        "prediction_rows": len(filtered),
        "prediction_dates": filtered["date"].nunique() if not filtered.empty else 0,
        "signal_rows": result_signals.height,
        "signal_dates": result_signals["date"].n_unique() if result_signals.height else 0,
        "sharpe": metric(result.metrics, "sharpe"),
        "max_drawdown": metric(result.metrics, "max_drawdown"),
        "cagr": metric(result.metrics, "cagr"),
        "sortino": metric(result.metrics, "sortino"),
        "win_rate": metric(result.metrics, "win_rate"),
    }
    return Candidate(candidate_id=cid, model=model, selector=selector, run=run, row=row)


def build_model_configs() -> list[ModelConfig]:
    seed_models = [
        (("baseline",), 12, 8, 630, 5, 3, 0.10),
        (("baseline",), 18, 10, 504, 5, 3, 0.10),
        (("baseline",), 30, 12, 504, 5, 3, 0.10),
        (("baseline",), 30, 12, 630, 5, 3, 0.10),
        (("baseline",), 21, 10, 756, 5, 3, 0.10),
        (("baseline",), 24, 12, 630, 5, 3, 0.10),
        (("baseline",), 36, 12, 630, 5, 3, 0.10),
    ]
    timing_models = [
        (("baseline",), fwd, rb, tw, 5, 3, 0.10)
        for fwd, rb, tw in product(
            [10, 12, 15, 18, 21, 24, 30, 36],
            [6, 8, 10, 12, 15],
            [378, 504, 630, 756],
        )
    ]
    model_variants = [
        (("baseline",), fwd, rb, tw, n, depth, lr)
        for fwd, rb, tw in [
            (12, 8, 630),
            (18, 10, 504),
            (30, 12, 504),
            (30, 12, 630),
            (21, 10, 756),
        ]
        for n, depth, lr in [
            (3, 2, 0.10),
            (5, 2, 0.10),
            (8, 2, 0.10),
            (10, 2, 0.05),
            (10, 3, 0.05),
            (12, 2, 0.05),
            (5, 3, 0.08),
            (5, 3, 0.12),
        ]
    ]
    feature_models = [
        (groups, fwd, rb, tw, 5, 3, 0.10)
        for groups, fwd, rb, tw in product(
            [
                ("baseline", "log"),
                ("baseline", "adx"),
                ("baseline", "zscore"),
                ("baseline", "volume_ratio"),
                ("baseline", "roc"),
                ("baseline", "hist_vol"),
                ("baseline", "log", "adx"),
                ("baseline", "adx", "zscore"),
            ],
            [12, 18, 21, 30],
            [8, 10, 12],
            [504, 630, 756],
        )
    ]
    seen: set[tuple[Any, ...]] = set()
    configs: list[ModelConfig] = []
    for raw in [*seed_models, *timing_models, *model_variants, *feature_models]:
        groups, fwd, rb, tw, n, depth, lr = raw
        key = (groups, fwd, rb, tw, n, depth, lr)
        if key in seen:
            continue
        seen.add(key)
        configs.append(
            ModelConfig(
                feature_groups=tuple(groups),
                forward_period=int(fwd),
                rebalance_frequency=int(rb),
                train_window=int(tw),
                n_estimators=int(n),
                max_depth=int(depth),
                learning_rate=float(lr),
            )
        )
    return configs


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qts-root", type=Path, default=DEFAULT_QTS_ROOT)
    parser.add_argument("--target-sharpe", type=float, default=1.3824)
    parser.add_argument("--max-models", type=int, default=80)
    parser.add_argument("--keep-candidates", type=int, default=80)
    parser.add_argument("--reset-ledger", action="store_true")
    args = parser.parse_args()

    qts_root = args.qts_root.resolve()
    os.environ["QTS_ROOT"] = str(qts_root)
    if args.reset_ledger:
        for path in (LEDGER, ENSEMBLE_LEDGER, HIT_YAML):
            path.unlink(missing_ok=True)

    resolved = flow.Config.build(flow.BASE_CONFIG_PATH)
    config = resolved.raw
    flow.validate_vn_ranker_config(config)
    data_manager = flow.build_data_manager(resolved)
    symbols = flow.config_symbols(config)
    test_intervals = flow.test_intervals_from_config(config)
    raw = await data_manager.get_ohlcv(
        symbols=symbols,
        start=config.start_date,
        end=config.end_date,
    )
    benchmark_close = await flow.load_benchmark_close_frame(data_manager, config)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(
        "Loaded",
        f"root={qts_root}",
        f"symbols={len(symbols)}",
        f"rows={raw.height}",
        f"intervals={flow.format_date_intervals(test_intervals)}",
        flush=True,
    )

    feature_cache: dict[int, pd.DataFrame] = {}
    candidates: list[Candidate] = []
    best_candidate: Candidate | None = None
    model_configs = build_model_configs()[: args.max_models]

    for model_index, model in enumerate(model_configs, 1):
        cols = predictor_cols(model.feature_groups)
        target_col = f"Return_fwd_{model.forward_period}"
        try:
            if model.forward_period not in feature_cache:
                featured = flow.build_reference_feature_frame(
                    raw,
                    forward_period=model.forward_period,
                )
                all_cols = list(
                    dict.fromkeys(
                        col
                        for groups in [
                            ("baseline",),
                            ("baseline", "log"),
                            ("baseline", "adx"),
                            ("baseline", "zscore"),
                            ("baseline", "volume_ratio"),
                            ("baseline", "roc"),
                            ("baseline", "hist_vol"),
                            ("baseline", "log", "adx"),
                            ("baseline", "adx", "zscore"),
                        ]
                        for col in predictor_cols(groups)
                    )
                )
                feature_cache[model.forward_period] = flow.prepare_dataset(
                    featured,
                    target_col=target_col,
                    predictor_cols=all_cols,
                )
            dataset = feature_cache[model.forward_period][
                ["date", "symbol", target_col, *cols]
            ].copy()
            predictions = flow.walk_forward_rank(
                frame=dataset,
                predictor=flow.XGBRankerPredictor(
                    n_estimators=model.n_estimators,
                    learning_rate=model.learning_rate,
                    max_depth=model.max_depth,
                    n_jobs=-1,
                ),
                predictor_cols=cols,
                target_col=target_col,
                train_window=model.train_window,
                rebalance_frequency=model.rebalance_frequency,
                target_horizon=model.forward_period,
                test_intervals=test_intervals,
            )
        except Exception as exc:
            append_tsv(
                LEDGER,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "status": "error",
                    "model_index": model_index,
                    "model_id": model_id(model),
                    "error": repr(exc),
                },
            )
            print("ERROR", model_index, model_id(model), repr(exc), flush=True)
            continue

        selector_count = 0
        for top_n in [3, 4, 5]:
            base_score_thresholds = score_thresholds(predictions, top_n)
            for regime_window in [None, 21, 42, 63]:
                filtered_preview, _ = flow.apply_param_benchmark_regime_filter(
                    predictions,
                    benchmark_close,
                    regime_cfg(regime_window),
                )
                thresholds = score_thresholds(filtered_preview, top_n)
                if thresholds == [None]:
                    thresholds = base_score_thresholds
                spreads = [None]
                for min_top_score in thresholds:
                    for min_score_spread in spreads:
                        selector = SelectorConfig(
                            top_positions=top_n,
                            min_top_score=min_top_score,
                            min_score_spread=min_score_spread,
                            regime_window=regime_window,
                        )
                        candidate = evaluate_selector(
                            raw=raw,
                            config=config,
                            engine=resolved.engine,
                            benchmark_close=benchmark_close,
                            predictions=predictions,
                            model=model,
                            selector=selector,
                            run_stamp=run_stamp,
                        )
                        selector_count += 1
                        row = {
                            "status": "ok",
                            "model_index": model_index,
                            **candidate.row,
                        }
                        append_tsv(LEDGER, row)
                        if (
                            np.isfinite(candidate.row["sharpe"])
                            and candidate.row["max_drawdown"] < 0.35
                        ):
                            candidates.append(candidate)
                            candidates.sort(key=lambda item: item.row["sharpe"], reverse=True)
                            del candidates[args.keep_candidates :]
                        if (
                            best_candidate is None
                            or candidate.row["sharpe"] > best_candidate.row["sharpe"]
                        ):
                            best_candidate = candidate
                        print(
                            "CAND",
                            model_index,
                            selector_count,
                            candidate.candidate_id,
                            f"sharpe={candidate.row['sharpe']:.4f}",
                            f"mdd={candidate.row['max_drawdown']:.4f}",
                            flush=True,
                        )

        top_rows = pd.DataFrame([item.row for item in candidates[:5]])
        print(
            "MODEL_DONE",
            model_index,
            model_id(model),
            f"selectors={selector_count}",
            f"best={best_candidate.row['sharpe']:.4f}" if best_candidate else "best=none",
            flush=True,
        )
        if not top_rows.empty:
            print(top_rows[["candidate_id", "sharpe", "max_drawdown"]].to_string(index=False))

        hit = search_ensembles(
            candidates,
            raw=raw,
            config=config,
            engine=resolved.engine,
            run_stamp=run_stamp,
            target_sharpe=args.target_sharpe,
        )
        if hit is not None:
            HIT_YAML.write_text(flow.yaml.safe_dump(hit, sort_keys=False), encoding="utf-8")
            print(f"HIT_YAML={HIT_YAML}", flush=True)
            return

    final = search_ensembles(
        candidates,
        raw=raw,
        config=config,
        engine=resolved.engine,
        run_stamp=run_stamp,
        target_sharpe=args.target_sharpe,
        force_best=True,
    )
    if final is not None:
        HIT_YAML.write_text(flow.yaml.safe_dump(final, sort_keys=False), encoding="utf-8")
    print("NO_TARGET_HIT", f"best={final}", flush=True)


def search_ensembles(
    candidates: list[Candidate],
    *,
    raw,
    config,
    engine,
    run_stamp: str,
    target_sharpe: float,
    force_best: bool = False,
) -> dict[str, Any] | None:
    if len(candidates) < 2:
        return None
    unique: list[Candidate] = []
    seen: set[str] = set()
    for candidate in sorted(candidates, key=lambda item: item.row["sharpe"], reverse=True):
        if candidate.candidate_id in seen:
            continue
        seen.add(candidate.candidate_id)
        unique.append(candidate)
        if len(unique) >= 30:
            break

    best: dict[str, Any] | None = None
    for i, first in enumerate(unique):
        for second in unique[i + 1 :]:
            for first_weight in [0.35, 0.4, 0.5, 0.6, 0.65]:
                second_weight = 1.0 - first_weight
                ensemble = flow.build_ensemble_artifacts(
                    [first.run, second.run],
                    weights=[first_weight, second_weight],
                    raw=raw,
                    config=config,
                    engine=engine,
                    run_stamp=run_stamp,
                )
                sharpe = metric(ensemble.result.metrics, "sharpe")
                max_drawdown = metric(ensemble.result.metrics, "max_drawdown")
                row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "first_candidate": first.candidate_id,
                    "second_candidate": second.candidate_id,
                    "first_weight": first_weight,
                    "second_weight": second_weight,
                    "sharpe": sharpe,
                    "max_drawdown": max_drawdown,
                    "cagr": metric(ensemble.result.metrics, "cagr"),
                    "sortino": metric(ensemble.result.metrics, "sortino"),
                    "signal_rows": ensemble.signals.height,
                    "signal_dates": ensemble.signals["date"].n_unique()
                    if ensemble.signals.height
                    else 0,
                }
                append_tsv(ENSEMBLE_LEDGER, row)
                if best is None or row["sharpe"] > best["sharpe"]:
                    best = {
                        **row,
                        "first": {
                            "model": asdict(first.model),
                            "selector": asdict(first.selector),
                            "metrics": first.row,
                        },
                        "second": {
                            "model": asdict(second.model),
                            "selector": asdict(second.selector),
                            "metrics": second.row,
                        },
                    }
                if sharpe >= target_sharpe and max_drawdown < 0.35:
                    print(
                        "ENSEMBLE_HIT",
                        f"sharpe={sharpe:.4f}",
                        f"mdd={max_drawdown:.4f}",
                        first.candidate_id,
                        second.candidate_id,
                        flush=True,
                    )
                    return best
    if force_best:
        return best
    if best is not None:
        print(
            "ENSEMBLE_BEST",
            f"sharpe={best['sharpe']:.4f}",
            f"mdd={best['max_drawdown']:.4f}",
            flush=True,
        )
    return None


if __name__ == "__main__":
    asyncio.run(main())
