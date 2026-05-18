"""Run the VN100 quantamental ML strategy from the command line.

The notebook version is useful for exploration. This script keeps the same
strategy path but makes the data fetch, feature build, walk-forward backtest,
artifacts, and optional MLflow logging runnable end to end.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys
import threading
from dataclasses import asdict
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "qts").exists() and (candidate / "pyproject.toml").exists():
            return candidate
        nested = candidate / "QTradeSystematic"
        if (nested / "qts").exists() and (nested / "pyproject.toml").exists():
            return nested
    raise FileNotFoundError("Could not find the QTradeSystematic project root")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("QTS_ROOT", str(PROJECT_ROOT / ".qts_notebook_runtime"))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import polars as pl
import yaml

import qts  # noqa: F401 — registry side effects
from qts.research.backtest.base import BacktestResult
from qts.research.strategies.vn100_quantamental import (
    ExperimentConfig,
    FeatureConfig,
    MODEL_PARAMS,
    VN100QuantamentalStrategy,
    build_model_frame,
    feature_coverage_report,
    fetch_prices_and_fundamentals,
    fundamental_cache_report,
    load_vn100_symbols,
    make_sweep_arms,
    walk_forward_ml_signals,
    choose_predictors,
)

MLFLOW_AVAILABLE = importlib.util.find_spec("mlflow") is not None
if MLFLOW_AVAILABLE:
    import mlflow
else:
    mlflow = None

pd.options.display.float_format = "{:,.4f}".format

DEFAULT_RUNTIME_ROOT = PROJECT_ROOT / ".qts_notebook_runtime" / "vn100_quantamental_mlflow"
DEFAULT_ASSET_CONFIG = PROJECT_ROOT / "configs" / "assets" / "vn100.yml"
DEFAULT_STRATEGY_CONFIG = PROJECT_ROOT / "configs" / "strategies" / "ml_factor" / "base.yaml"
DEFAULT_MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
DEFAULT_MLFLOW_EXPERIMENT_NAME = "VN100 Quantamental ML"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    parsed = int(value)
    return parsed if parsed > 0 else None


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_date(name: str, default: date) -> date:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else date.fromisoformat(value)


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def parse_csv(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:
            error["value"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if "value" in error:
        raise error["value"]
    return result["value"]


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def summarize_ohlcv(ohlcv: pl.DataFrame) -> pl.DataFrame:
    return (
        ohlcv.group_by("symbol")
        .agg(
            pl.col("date").min().alias("start"),
            pl.col("date").max().alias("end"),
            pl.len().alias("rows"),
            pl.col("close").last().alias("last_close"),
            pl.col("volume").mean().alias("avg_volume"),
        )
        .sort("rows", descending=True)
    )


def serializable(value: Any) -> Any:
    from dataclasses import is_dataclass
    if is_dataclass(value):
        return serializable(asdict(value))
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [serializable(item) for item in value]
    if isinstance(value, date | datetime | Decimal | Path):
        return str(value)
    return value


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def strategy_params(strategy_config: dict[str, Any]) -> dict[str, Any]:
    strategy = strategy_config.get("strategy", {})
    if not isinstance(strategy, dict):
        return {}
    params = strategy.get("params", {})
    return dict(params) if isinstance(params, dict) else {}


def target_period_from_config(strategy_config: dict[str, Any]) -> int | None:
    target_col = strategy_params(strategy_config).get("target_col")
    if not isinstance(target_col, str) or not target_col.startswith("forward_return_"):
        return None
    try:
        return int(target_col.rsplit("_", 1)[1])
    except ValueError:
        return None


def configured_rebalance_period(
    strategy_config: dict[str, Any], override: str | None
) -> str | int:
    if override is not None:
        try:
            parsed = int(override)
            if parsed <= 5:
                raise ValueError(f"--rebalance-period integer must be > 5, got {parsed}")
            return parsed
        except ValueError as exc:
            if "must be > 5" in str(exc):
                raise
            return override
    params = strategy_params(strategy_config)
    value = (
        params.get("rebalance_period")
        or params.get("rebalance_frequency")
        or strategy_config.get("rebalance_frequency")
        or "monthly"
    )
    return str(value)


def configured_model_params(
    strategy_config: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    from copy import deepcopy
    params = deepcopy(MODEL_PARAMS)
    trainer = strategy_params(strategy_config).get("trainer", {})
    if isinstance(trainer, dict):
        trainer_params = trainer.get("params", {})
        if isinstance(trainer_params, dict) and isinstance(
            trainer_params.get("model_params"), dict
        ):
            params.update(trainer_params["model_params"])
    if args.model_n_estimators is not None:
        params["n_estimators"] = args.model_n_estimators
    if args.model_max_depth is not None:
        params["max_depth"] = args.model_max_depth
    return params


def configured_portfolio_params(strategy_config: dict[str, Any]) -> dict[str, Any]:
    portfolio = strategy_params(strategy_config).get("portfolio", {})
    if not isinstance(portfolio, dict):
        return {}
    params = portfolio.get("params", {})
    return dict(params) if isinstance(params, dict) else {}


def build_feature_config(
    strategy_config: dict[str, Any], args: argparse.Namespace
) -> FeatureConfig:
    forward_period = args.forward_period or target_period_from_config(strategy_config) or 21
    return FeatureConfig(
        min_trading_days=args.min_trading_days,
        volume_top_n=args.volume_top_n,
        min_avg_volume=args.min_avg_volume,
        remove_large_gaps=args.remove_large_gaps,
        max_gap_days=args.max_gap_days,
        remove_low_volume=args.remove_low_volume,
        qsmom_fast=args.qsmom_fast,
        qsmom_slow=args.qsmom_slow,
        qsmom_returns=args.qsmom_returns,
        forward_period=forward_period,
        fundamental_termtype=args.fundamental_termtype,
    )


def build_baseline_experiment(
    strategy_config: dict[str, Any],
    args: argparse.Namespace,
    feature: FeatureConfig,
) -> ExperimentConfig:
    portfolio_params = configured_portfolio_params(strategy_config)
    predictor_cols = parse_csv(args.predictor_cols)
    if predictor_cols is None:
        configured_predictors = strategy_params(strategy_config).get("predictor_cols")
        if isinstance(configured_predictors, list):
            predictor_cols = [str(item) for item in configured_predictors]
    return ExperimentConfig(
        name=args.run_name,
        feature=feature,
        predictor_cols=predictor_cols,
        train_window=args.train_window or int(strategy_config.get("train_window") or 504),
        rebalance_period=configured_rebalance_period(strategy_config, args.rebalance_period),
        num_long_positions=args.num_long_positions
        or int(portfolio_params.get("num_long_positions") or 10),
        num_short_positions=args.num_short_positions
        if args.num_short_positions is not None
        else int(portfolio_params.get("num_short_positions") or 0),
        long_threshold=args.long_threshold
        if args.long_threshold is not None
        else portfolio_params.get("long_threshold"),
        short_threshold=args.short_threshold
        if args.short_threshold is not None
        else portfolio_params.get("short_threshold"),
        model_params=configured_model_params(strategy_config, args),
        initial_capital=Decimal(
            str(args.initial_capital or strategy_config.get("initial_capital") or "1000000000")
        ),
    )


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------

def mlflow_artifact_location(tracking_uri: str, runtime_root: Path) -> str | None:
    scheme = urlparse(tracking_uri).scheme
    if scheme in {"", "file", "sqlite"}:
        return (runtime_root / "mlartifacts").as_uri()
    return None


def setup_mlflow(
    enabled: bool, tracking_uri: str, experiment_name: str, runtime_root: Path
) -> bool:
    if not enabled:
        print("MLflow logging disabled.")
        return False
    if not MLFLOW_AVAILABLE or mlflow is None:
        print("MLflow is not installed. Skipping tracking.")
        return False
    try:
        mlflow.set_tracking_uri(tracking_uri)
        if mlflow.get_experiment_by_name(experiment_name) is None:
            artifact_location = mlflow_artifact_location(tracking_uri, runtime_root)
            if artifact_location is None:
                mlflow.create_experiment(experiment_name)
            else:
                mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
        mlflow.set_experiment(experiment_name)
    except Exception as exc:
        print(f"MLflow setup failed for {tracking_uri}; continuing without tracking: {exc}")
        print("Start the shared server with: docker compose -f docker/compose.yaml up -d mlflow")
        return False
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment: {experiment_name}")
    return True


def flatten_params(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_params(value, full_key))
        else:
            flattened[full_key] = value
    return flattened


def mlflow_param_value(value: Any) -> str | int | float | bool | None:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return repr(value)


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------

def write_result_artifacts(run_dir: Path, payload: dict[str, Any]) -> list[Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    result: BacktestResult = payload["result"]
    csv_artifacts = {
        "returns.csv": result.returns.to_pandas(),
        "equity_curve.csv": result.equity_curve.to_pandas(),
        "signals.csv": payload["signals"].to_pandas(),
        "predictor_cols.csv": pd.DataFrame({"predictor_col": payload["predictor_cols"]}),
        "feature_diagnostics.csv": payload["feature_diagnostics"].to_pandas(),
        "factor_sources.csv": pd.DataFrame(
            [
                {"factor": factor, "source_columns": ",".join(columns)}
                for factor, columns in payload["factor_sources"].items()
            ]
        ),
    }
    paths = []
    for filename, frame in csv_artifacts.items():
        path = run_dir / filename
        frame.to_csv(path, index=False)
        paths.append(path)
    model_frame_path = run_dir / "model_frame.parquet"
    payload["model_frame"].write_parquet(model_frame_path)
    paths.append(model_frame_path)
    return paths


def write_run_config(run_dir: Path, args: argparse.Namespace, experiment: ExperimentConfig) -> Path:
    path = run_dir / "run_config.yaml"
    payload = {
        "args": vars(args),
        "experiment": experiment,
        "qts_root": os.environ.get("QTS_ROOT"),
    }
    path.write_text(yaml.safe_dump(serializable(payload), sort_keys=True))
    return path


def log_to_mlflow(
    enabled: bool,
    experiment: ExperimentConfig,
    payload: dict[str, Any],
    runtime_root: Path,
    extra_params: dict[str, Any],
) -> None:
    if not enabled or mlflow is None:
        return
    result: BacktestResult = payload["result"]
    params = {
        "name": experiment.name,
        "train_window": experiment.train_window,
        "rebalance_period": experiment.rebalance_period,
        "num_long_positions": experiment.num_long_positions,
        "num_short_positions": experiment.num_short_positions,
        "long_threshold": experiment.long_threshold,
        "short_threshold": experiment.short_threshold,
        "predictor_cols": payload["predictor_cols"],
        "target_col": payload["target_col"],
        "feature": asdict(experiment.feature),
        "model_params": experiment.model_params or MODEL_PARAMS,
        **extra_params,
    }
    artifact_dir = runtime_root / "mlflow_artifacts" / experiment.name
    artifacts = write_result_artifacts(artifact_dir, payload)
    flat_params = {key: mlflow_param_value(value) for key, value in flatten_params(params).items()}
    try:
        with mlflow.start_run(
            run_name=experiment.name, tags={"asset_type": "vn_stock", "universe": "VN100"}
        ):
            mlflow.log_params(flat_params)
            mlflow.log_metrics({key: float(value) for key, value in result.metrics.items()})
            mlflow.log_metric("signal_rows", float(payload["signals"].height))
            mlflow.log_metric("featured_rows", float(payload["model_frame"].height))
            for artifact in artifacts:
                mlflow.log_artifact(str(artifact))
    except Exception as exc:
        print(f"MLflow logging failed for {experiment.name}; continuing: {exc}")


def result_row(
    experiment: ExperimentConfig, payload: dict[str, Any], status: str = "ok"
) -> dict[str, Any]:
    result: BacktestResult = payload["result"]
    return {
        "name": experiment.name,
        **result.metrics,
        "signal_rows": payload["signals"].height,
        "featured_rows": payload["model_frame"].height,
        "predictor_count": len(payload["predictor_cols"]),
        "rebalance_period": experiment.rebalance_period,
        "train_window": experiment.train_window,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    env_strategy_config = os.environ.get("QTS_NOTEBOOK_STRATEGY_CONFIG")
    parser.add_argument("--asset-config", type=Path, default=DEFAULT_ASSET_CONFIG)
    parser.add_argument(
        "--strategy-config",
        type=Path,
        default=Path(env_strategy_config) if env_strategy_config else None,
        help=f"Optional ML factor config, for example {DEFAULT_STRATEGY_CONFIG}",
    )
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name", default="baseline_qsmom_fundamental_xgb")
    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=env_date("QTS_NOTEBOOK_START_DATE", date(2021, 1, 1)),
    )
    parser.add_argument(
        "--end-date", type=parse_date, default=env_date("QTS_NOTEBOOK_END_DATE", date.today())
    )
    parser.add_argument("--interval", default=os.environ.get("QTS_NOTEBOOK_INTERVAL", "1d"))
    parser.add_argument("--max-symbols", type=int, default=env_int("QTS_NOTEBOOK_MAX_SYMBOLS"))
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument(
        "--fundamental-termtype", type=int, default=env_int("QTS_NOTEBOOK_FUNDAMENTAL_TERMTYPE", 1)
    )
    parser.add_argument(
        "--fundamental-pages", type=int, default=env_int("QTS_NOTEBOOK_FUNDAMENTAL_PAGES", 3)
    )
    parser.add_argument(
        "--fetch-fundamentals",
        action=argparse.BooleanOptionalAction,
        default=env_bool("QTS_NOTEBOOK_FETCH_FUNDAMENTALS", True),
    )
    parser.add_argument(
        "--force-refresh-fundamentals",
        action=argparse.BooleanOptionalAction,
        default=env_bool("QTS_NOTEBOOK_FORCE_REFRESH_FUNDAMENTALS", False),
    )
    parser.add_argument(
        "--run-sweeps",
        action=argparse.BooleanOptionalAction,
        default=env_bool("QTS_NOTEBOOK_RUN_SWEEPS", False),
    )
    parser.add_argument(
        "--mlflow",
        action=argparse.BooleanOptionalAction,
        default=not env_bool("QTS_NOTEBOOK_DISABLE_MLFLOW", False),
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI),
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_MLFLOW_EXPERIMENT_NAME),
    )
    parser.add_argument("--train-window", type=int)
    parser.add_argument(
        "--rebalance-period",
        default=None,
        help="Number of trading days between rebalances (integer > 5). "
             "Legacy strings 'daily'/'weekly'/'monthly' are also accepted.",
    )
    parser.add_argument("--predictor-cols")
    parser.add_argument("--num-long-positions", type=int)
    parser.add_argument("--num-short-positions", type=int)
    parser.add_argument("--long-threshold", type=float)
    parser.add_argument("--short-threshold", type=float)
    parser.add_argument("--initial-capital")
    parser.add_argument("--min-trading-days", type=int, default=252)
    parser.add_argument("--volume-top-n", type=int, default=80)
    parser.add_argument("--min-avg-volume", type=float, default=50_000)
    parser.add_argument("--remove-large-gaps", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-gap-days", type=int, default=21)
    parser.add_argument("--remove-low-volume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qsmom-fast", type=int, default=21)
    parser.add_argument("--qsmom-slow", type=int, default=252)
    parser.add_argument("--qsmom-returns", type=int, default=126)
    parser.add_argument("--forward-period", type=int)
    parser.add_argument("--model-n-estimators", type=int)
    parser.add_argument("--model-max-depth", type=int)
    args = parser.parse_args(argv)
    if args.volume_top_n is not None and args.volume_top_n <= 0:
        args.volume_top_n = None
    return args


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runtime_root = args.runtime_root.resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    os.environ["QTS_ROOT"] = str(runtime_root)

    run_dir = args.output_dir or runtime_root / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    strategy_config = load_yaml_mapping(args.strategy_config) if args.strategy_config else {}
    feature_config = build_feature_config(strategy_config, args)
    baseline_experiment = build_baseline_experiment(strategy_config, args, feature_config)
    write_run_config(run_dir, args, baseline_experiment)

    symbols, benchmark_symbol = load_vn100_symbols(args.asset_config, args.max_symbols)
    request_symbols = symbols + [benchmark_symbol]

    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"QTS_ROOT: {os.environ['QTS_ROOT']}")
    print(f"Run directory: {run_dir}")
    print(f"VN100 equity symbols: {len(symbols)}")
    print(f"Benchmark symbol: {benchmark_symbol}")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Rebalance period: {baseline_experiment.rebalance_period}")

    ohlcv, benchmark_ohlcv, fetch_failures = run_async(
        fetch_prices_and_fundamentals(
            request_symbols,
            benchmark_symbol,
            runtime_root,
            start_date=args.start_date,
            end_date=args.end_date,
            interval=args.interval,
            batch_size=args.batch_size,
            fetch_fundamentals=args.fetch_fundamentals,
            fundamental_termtype=args.fundamental_termtype,
            fundamental_pages=args.fundamental_pages,
            force_refresh_fundamentals=args.force_refresh_fundamentals,
        )
    )

    if ohlcv.is_empty():
        raise RuntimeError(
            "No OHLCV rows were fetched. Check network access or reduce the universe."
        )

    summarize_ohlcv(ohlcv).to_pandas().to_csv(run_dir / "ohlcv_summary.csv", index=False)
    benchmark_ohlcv.write_parquet(run_dir / "benchmark_ohlcv.parquet")
    pd.DataFrame(fetch_failures, columns=["symbol", "error"]).to_csv(
        run_dir / "fetch_failures.csv", index=False
    )
    fundamental_cache_report(symbols, termtype=args.fundamental_termtype).to_pandas().to_csv(
        run_dir / "fundamental_cache.csv", index=False
    )

    print(f"Fetched OHLCV rows: {ohlcv.height:,}")
    print(f"Fetched symbols: {ohlcv.select('symbol').n_unique():,}")
    print(f"Benchmark rows: {benchmark_ohlcv.height:,}")
    print(f"Fetch failures: {len(fetch_failures)}")

    mlflow_enabled = setup_mlflow(
        args.mlflow,
        args.mlflow_tracking_uri,
        args.mlflow_experiment_name,
        runtime_root,
    )

    rows: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}

    print(f"Running baseline: {baseline_experiment.name}")
    baseline_strategy = VN100QuantamentalStrategy(baseline_experiment)
    baseline_payload = baseline_strategy.run_experiment(ohlcv)
    write_result_artifacts(run_dir / baseline_experiment.name, baseline_payload)
    log_to_mlflow(mlflow_enabled, baseline_experiment, baseline_payload, runtime_root, {"run_type": "baseline"})
    rows.append(result_row(baseline_experiment, baseline_payload))
    payloads[baseline_experiment.name] = baseline_payload

    if args.run_sweeps:
        sweep_arms = make_sweep_arms(baseline_experiment)
        print(f"Prepared sweep arms: {len(sweep_arms)}")
        for experiment in sweep_arms:
            print(f"Running sweep arm: {experiment.name}")
            try:
                strategy = VN100QuantamentalStrategy(experiment)
                payload = strategy.run_experiment(ohlcv)
                write_result_artifacts(run_dir / experiment.name, payload)
                log_to_mlflow(mlflow_enabled, experiment, payload, runtime_root, {"run_type": "sweep"})
                rows.append(result_row(experiment, payload))
                payloads[experiment.name] = payload
            except Exception as exc:
                rows.append({"name": experiment.name, "status": "failed", "error": str(exc)})
                print(f"  failed: {exc}")

    results_df = pd.DataFrame(rows)
    results_path = run_dir / "results_summary.csv"
    results_df.to_csv(results_path, index=False)

    ok_results = (
        results_df[results_df["status"] == "ok"] if "status" in results_df else pd.DataFrame()
    )
    if not ok_results.empty and "sharpe" in ok_results:
        best = ok_results.sort_values("sharpe", ascending=False).iloc[0]
        print(f"Best run by Sharpe: {best['name']} ({best['sharpe']:.4f})")
    baseline_result: BacktestResult = baseline_payload["result"]
    print(f"Baseline metrics: {baseline_result.metrics}")
    print(f"Results summary: {results_path}")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
