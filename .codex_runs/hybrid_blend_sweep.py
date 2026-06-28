from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "notebooks") not in sys.path:
    sys.path.insert(0, str(ROOT / "notebooks"))

import combine_ml_e2e_classification as classifier_flow  # noqa: E402
import combine_ml_e2e_classification_ranking as hybrid_flow  # noqa: E402
import combine_ml_e2e_ranking as ranker_flow  # noqa: E402
from qts.config.builder import Config  # noqa: E402


def metric(metrics: dict[str, Any], key: str) -> float:
    try:
        return float(metrics.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def append_tsv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        path,
        sep="\t",
        index=False,
        mode="a",
        header=not path.exists(),
    )


def comparison_row(
    run: hybrid_flow.ComparisonRun,
    *,
    method: str,
    ranker_weight: float,
) -> dict[str, Any]:
    exposure = hybrid_flow.target_exposure_summary(run.target_weights)
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "method": method,
        "ranker_weight": ranker_weight,
        "sharpe": metric(run.result.metrics, "sharpe"),
        "max_drawdown": metric(run.result.metrics, "max_drawdown"),
        "cagr": metric(run.result.metrics, "cagr"),
        "sortino": metric(run.result.metrics, "sortino"),
        "win_rate": metric(run.result.metrics, "win_rate"),
        "avg_gross_exposure": exposure["avg_gross_exposure"],
        "signal_dates": run.signals["date"].n_unique() if run.signals.height else 0,
        **{f"eval_{key}": value for key, value in run.eval_summary.items()},
    }


async def main_async(args: argparse.Namespace) -> None:
    if args.qts_root is not None:
        qts_root = args.qts_root.resolve()
        qts_root.mkdir(parents=True, exist_ok=True)
        os.environ["QTS_ROOT"] = str(qts_root)

    if args.reset and args.ledger.exists():
        args.ledger.unlink()

    resolved = Config.build(args.base_config)
    config = resolved.raw
    classifier_flow.validate_vn_classification_config(config)
    ranker_flow.validate_vn_ranker_config(config)
    data_manager = classifier_flow.build_data_manager(resolved)
    symbols = classifier_flow.config_symbols(config)
    test_intervals = hybrid_flow.test_intervals_from_config(config)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(
        "Loaded config",
        f"symbols={len(symbols)}",
        f"validation={hybrid_flow.format_date_intervals(test_intervals)}",
        flush=True,
    )
    raw = await data_manager.get_ohlcv(
        symbols=symbols,
        start=config.start_date,
        end=config.end_date,
    )
    benchmark_close = await classifier_flow.load_benchmark_close(data_manager, config)

    classifier_paths = args.classifier_config or hybrid_flow.DEFAULT_CLASSIFIER_CONFIGS
    classifier_specs = [
        classifier_flow.resolve_classification_spec(path.resolve(), index)
        for index, path in enumerate(classifier_paths, 1)
    ]
    classifier_weights = classifier_flow.parse_ensemble_weights(
        args.classifier_ensemble_weights,
        len(classifier_specs),
    )
    classifier_output_dirs = classifier_flow.resolve_output_dirs(classifier_specs)
    classifier_feature_cache: dict[int, pd.DataFrame] = {}
    classifier_runs = []
    for spec in classifier_specs:
        print("Classifier", spec.name, flush=True)
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

    ranker_paths = args.ranker_config or hybrid_flow.DEFAULT_RANKER_CONFIGS
    ranker_specs = [
        ranker_flow.resolve_ranker_spec(path.resolve(), index)
        for index, path in enumerate(ranker_paths, 1)
    ]
    ranker_weights = ranker_flow.parse_ensemble_weights(
        args.ranker_ensemble_weights,
        len(ranker_specs),
    )
    ranker_model_weights_dir = ranker_flow.resolve_model_weights_dir(ranker_specs)
    ranker_runs = []
    for spec in ranker_specs:
        print("Ranker", spec.name, flush=True)
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

    classifier_cmp = hybrid_flow.comparison_from_classifier_run(classifier_ensemble)
    ranker_cmp = hybrid_flow.comparison_from_ranker_run(ranker_ensemble)
    append_tsv(
        args.ledger,
        comparison_row(classifier_cmp, method="classifier_ensemble", ranker_weight=0.0),
    )
    append_tsv(
        args.ledger,
        comparison_row(ranker_cmp, method="ranker_ensemble", ranker_weight=1.0),
    )
    print(
        "Baseline",
        f"classifier_sharpe={metric(classifier_cmp.result.metrics, 'sharpe'):.4f}",
        f"classifier_mdd={metric(classifier_cmp.result.metrics, 'max_drawdown'):.4f}",
        f"ranker_sharpe={metric(ranker_cmp.result.metrics, 'sharpe'):.4f}",
        f"ranker_mdd={metric(ranker_cmp.result.metrics, 'max_drawdown'):.4f}",
        flush=True,
    )

    best_row: dict[str, Any] | None = None
    weights = [float(item) for item in args.ranker_weight_grid.split(",") if item.strip()]
    for ranker_weight in weights:
        run = hybrid_flow.build_hybrid_run(
            classifier_ensemble=classifier_ensemble,
            ranker_ensemble=ranker_ensemble,
            raw=raw,
            config=config,
            engine=resolved.engine,
            run_stamp=run_stamp,
            hybrid_method="target_weight_blend",
            min_classifier_weight=0.0,
            renormalize_gross=False,
            hybrid_ranker_weight=ranker_weight,
        )
        row = comparison_row(run, method="target_weight_blend", ranker_weight=ranker_weight)
        append_tsv(args.ledger, row)
        if best_row is None or row["sharpe"] > best_row["sharpe"]:
            best_row = row
        print(
            "HYBRID",
            f"ranker_weight={ranker_weight:.3f}",
            f"sharpe={row['sharpe']:.4f}",
            f"mdd={row['max_drawdown']:.4f}",
            flush=True,
        )

    gated = hybrid_flow.build_hybrid_run(
        classifier_ensemble=classifier_ensemble,
        ranker_ensemble=ranker_ensemble,
        raw=raw,
        config=config,
        engine=resolved.engine,
        run_stamp=run_stamp,
        hybrid_method="classifier_gated_ranker",
        min_classifier_weight=0.0,
        renormalize_gross=False,
        hybrid_ranker_weight=0.5,
    )
    gated_row = comparison_row(gated, method="classifier_gated_ranker", ranker_weight=1.0)
    append_tsv(args.ledger, gated_row)
    if best_row is None or gated_row["sharpe"] > best_row["sharpe"]:
        best_row = gated_row
    print(
        "BEST",
        f"method={best_row['method']}",
        f"ranker_weight={best_row['ranker_weight']}",
        f"sharpe={best_row['sharpe']:.4f}",
        f"mdd={best_row['max_drawdown']:.4f}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=hybrid_flow.BASE_CONFIG_PATH)
    parser.add_argument("--classifier-config", type=Path, action="append", default=None)
    parser.add_argument("--ranker-config", type=Path, action="append", default=None)
    parser.add_argument("--classifier-ensemble-weights", default="0.5,0.5")
    parser.add_argument("--ranker-ensemble-weights", default="0.5,0.5")
    parser.add_argument(
        "--ranker-weight-grid",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    )
    parser.add_argument("--qts-root", type=Path, default=None)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=ROOT / ".qts_notebooke_runtime" / "hybrid_blend_sweep.tsv",
    )
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--no-model-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
