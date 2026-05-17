"""Generic research and sweep orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
import yaml

from qts.config.builder import Config
from qts.config.loader import load_config_mapping
from qts.core.errors import ConfigError
from qts.orchestration.flow import run_resolved_config
from qts.orchestration.tracking import NullTracker, create_tracker
from qts.research.backtest.base import BacktestResult


@dataclass(frozen=True, slots=True)
class SweepArm:
    name: str
    config: dict[str, Any]
    params: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SweepRunResult:
    arms: list[SweepArm]
    results: list[BacktestResult]
    best_params: dict[str, Any] | None = None
    best_value: float | None = None


class SummaryStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "summary.duckdb"

    def write(
        self,
        *,
        run_name: str,
        result: BacktestResult,
        params: Mapping[str, Any],
    ) -> list[Path]:
        run_dir = self.root / _safe_name(run_name)
        run_dir.mkdir(parents=True, exist_ok=True)
        artifacts = [
            _write_parquet(result.returns, run_dir / "returns.parquet"),
            _write_parquet(result.equity_curve, run_dir / "equity_curve.parquet"),
            _write_parquet(result.signals, run_dir / "signals.parquet"),
        ]
        with duckdb.connect(str(self.database)) as connection:
            connection.execute(
                """
                create table if not exists runs (
                    run_name varchar,
                    engine_name varchar,
                    params varchar,
                    metrics varchar
                )
                """
            )
            connection.execute(
                "insert into runs values (?, ?, ?, ?)",
                [run_name, result.engine_name, repr(dict(params)), repr(result.metrics)],
            )
        return [path for path in artifacts if path is not None]


def deep_overlay_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_overlay_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def expand_grid_sweep(
    base_config: Mapping[str, Any],
    sweep_config: Mapping[str, Any],
) -> list[SweepArm]:
    sweep = sweep_config.get("sweep", {})
    if sweep.get("mode") != "grid":
        raise ConfigError("Only sweep.mode: grid is supported")
    axes = list(sweep.get("axes", []))
    if not axes:
        raise ConfigError("sweep.axes must contain at least one axis")
    values = [list(axis.get("values", [])) for axis in axes]
    if any(not axis_values for axis_values in values):
        raise ConfigError("Each sweep axis must define values")

    arms: list[SweepArm] = []
    for index, combination in enumerate(product(*values), start=1):
        config = deepcopy(dict(base_config))
        params: dict[str, Any] = {}
        for axis, value in zip(axes, combination, strict=True):
            path = str(axis["path"])
            _set_dotted_path(config, path, value)
            params[path] = deepcopy(value)
        arms.append(SweepArm(name=f"arm-{index:03d}", config=config, params=params))
    return arms


def suggest_optuna_params(trial, search_space: list[Mapping[str, Any]]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for item in search_space:
        path = str(item["path"])
        param_type = str(item.get("type", "float")).lower()
        if param_type == "float":
            params[path] = trial.suggest_float(
                path,
                float(item["low"]),
                float(item["high"]),
                step=item.get("step"),
                log=bool(item.get("log", False)),
            )
        elif param_type == "int":
            params[path] = trial.suggest_int(
                path,
                int(item["low"]),
                int(item["high"]),
                step=int(item.get("step", 1)),
                log=bool(item.get("log", False)),
            )
        elif param_type == "categorical":
            choices = list(item.get("choices", []))
            if not choices:
                raise ConfigError(f"Optuna categorical parameter has no choices: {path}")
            params[path] = trial.suggest_categorical(path, choices)
        else:
            raise ConfigError(f"Unknown Optuna parameter type: {param_type}")
    return params


async def run_research_config(
    config: str | Path | Mapping[str, Any],
    *,
    tracker=None,
    run_name: str | None = None,
    nested: bool = False,
    summary_store: SummaryStore | None = None,
    params: Mapping[str, Any] | None = None,
) -> BacktestResult:
    resolved = (
        Config.build(config)
        if isinstance(config, str | Path)
        else Config.build_from_mapping(config)
    )
    if resolved.raw.workflow != "research":
        raise ConfigError("Generic research runs only support workflow: research")
    active_tracker = tracker or NullTracker()
    active_run_name = run_name or "research-run"
    active_params = dict(params or {})
    with active_tracker.start_run(run_name=active_run_name, nested=nested):
        result = await run_resolved_config(resolved)
        active_tracker.log_params(active_params)
        active_tracker.log_metrics(result.metrics)
        if summary_store is not None:
            for artifact in summary_store.write(
                run_name=active_run_name,
                result=result,
                params=active_params,
            ):
                active_tracker.log_artifact(str(artifact))
        return result


async def run_sweep_config(
    sweep_config_path: str | Path,
    *,
    tracker=None,
) -> SweepRunResult:
    sweep_path = Path(sweep_config_path)
    sweep_config = _load_yaml(sweep_path)
    _validate_sweep_top_level(sweep_config)
    base_path = _resolve_path(sweep_path.parent, sweep_config["base_config"])
    base_config = load_config_mapping(base_path)
    overlay = sweep_config.get("execution", {}).get("overrides", {})
    if overlay:
        base_config = deep_overlay_merge(base_config, overlay)
    if base_config.get("workflow") != "research":
        raise ConfigError("Sweep base_config must use workflow: research")

    active_tracker = tracker or create_tracker(sweep_config.get("tracking"))
    storage_config = sweep_config.get("storage", {})
    summary_store = SummaryStore(storage_config.get("root", "runs/research"))

    with active_tracker.start_run(
        run_name=sweep_config.get("tracking", {}).get("run_name", "research-sweep"),
        nested=False,
    ):
        mode = _sweep_mode(sweep_config)
        if mode == "grid":
            return await _run_grid_sweep(base_config, sweep_config, active_tracker, summary_store)
        if mode == "optuna":
            return await _run_optuna_sweep(base_config, sweep_config, active_tracker, summary_store)
        raise ConfigError(f"Unknown sweep.mode: {mode}")


async def _run_grid_sweep(
    base_config: Mapping[str, Any],
    sweep_config: Mapping[str, Any],
    tracker,
    summary_store: SummaryStore,
) -> SweepRunResult:
    arms = _named_arms(expand_grid_sweep(base_config, sweep_config), sweep_config)
    results: list[BacktestResult] = []
    for arm in arms:
        result = await run_research_config(
            arm.config,
            tracker=tracker,
            run_name=arm.name,
            nested=True,
            params=arm.params,
        )
        summary_store.write(run_name=arm.name, result=result, params=arm.params)
        results.append(result)
    best_value, best_params = _best_from_results(results, arms, _objective_config(sweep_config))
    return SweepRunResult(
        arms=arms,
        results=results,
        best_params=best_params,
        best_value=best_value,
    )


async def _run_optuna_sweep(
    base_config: Mapping[str, Any],
    sweep_config: Mapping[str, Any],
    tracker,
    summary_store: SummaryStore,
) -> SweepRunResult:
    optuna = _load_optuna()
    study = _create_optuna_study(optuna, sweep_config, summary_store)
    sweep = sweep_config.get("sweep", {})
    objective = _objective_config(sweep_config)
    search_space = list(sweep.get("search_space", []))
    if not search_space:
        raise ConfigError("Optuna sweep requires sweep.search_space")
    n_trials = int(sweep.get("n_trials", 1))
    if n_trials < 1:
        raise ConfigError("Optuna sweep requires n_trials >= 1")

    arms: list[SweepArm] = []
    results: list[BacktestResult] = []
    best_value: float | None = None
    best_params: dict[str, Any] | None = None

    for _ in range(n_trials):
        trial = study.ask()
        params = suggest_optuna_params(trial, search_space)
        config = deepcopy(dict(base_config))
        for path, value in params.items():
            _set_dotted_path(config, path, value)
        arm = SweepArm(
            name=f"trial-{int(trial.number):03d}",
            config=config,
            params=params,
        )
        result = await run_research_config(
            arm.config,
            tracker=tracker,
            run_name=arm.name,
            nested=True,
            params=arm.params,
        )
        value = _objective_value(result, objective)
        study.tell(trial, value)
        summary_store.write(run_name=arm.name, result=result, params=arm.params)
        arms.append(arm)
        results.append(result)
        if _is_better(value, best_value, objective["direction"]):
            best_value = value
            best_params = dict(params)

    return SweepRunResult(
        arms=arms,
        results=results,
        best_params=best_params or dict(getattr(study, "best_params", {})),
        best_value=best_value if best_value is not None else getattr(study, "best_value", None),
    )


def _set_dotted_path(config: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = config
    for part in parts[:-1]:
        next_value = cursor.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise ConfigError(f"Cannot set dotted path through non-mapping: {path}")
        cursor = next_value
    cursor[parts[-1]] = deepcopy(value)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"Expected mapping YAML: {path}")
    return value


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _validate_sweep_top_level(config: Mapping[str, Any]) -> None:
    allowed = {"base_config", "tracking", "execution", "storage", "sweep", "run_name_template"}
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ConfigError(f"Unknown sweep key(s): {', '.join(unknown)}")
    if "base_config" not in config:
        raise ConfigError("Sweep config must define base_config")


def _sweep_mode(sweep_config: Mapping[str, Any]) -> str:
    sweep = sweep_config.get("sweep", {})
    mode = sweep.get("mode")
    if mode:
        return str(mode).lower()
    if "search_space" in sweep:
        return "optuna"
    if "axes" in sweep:
        return "grid"
    return "optuna"


def _objective_config(sweep_config: Mapping[str, Any]) -> dict[str, str]:
    objective = sweep_config.get("sweep", {}).get("objective", {})
    direction = str(objective.get("direction", "maximize")).lower()
    if direction not in {"maximize", "minimize"}:
        raise ConfigError("Optuna objective direction must be maximize or minimize")
    return {
        "metric": str(objective.get("metric", "sharpe")),
        "direction": direction,
    }


def _objective_value(result: BacktestResult, objective: Mapping[str, str]) -> float:
    metric = objective["metric"]
    try:
        return float(result.metrics[metric])
    except KeyError as exc:
        raise ConfigError(f"Objective metric not found: {metric}") from exc


def _best_from_results(
    results: list[BacktestResult],
    arms: list[SweepArm],
    objective: Mapping[str, str],
) -> tuple[float | None, dict[str, Any] | None]:
    best_value: float | None = None
    best_params: dict[str, Any] | None = None
    for result, arm in zip(results, arms, strict=True):
        value = _objective_value(result, objective)
        if _is_better(value, best_value, objective["direction"]):
            best_value = value
            best_params = dict(arm.params)
    return best_value, best_params


def _is_better(value: float, current: float | None, direction: str) -> bool:
    return current is None or (
        (direction == "maximize" and value > current)
        or (direction == "minimize" and value < current)
    )


def _load_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise ConfigError("Optuna sweeps require the tuning optional dependency") from exc
    return optuna


def _create_optuna_study(optuna, sweep_config: Mapping[str, Any], summary_store: SummaryStore):
    sweep = sweep_config.get("sweep", {})
    objective = _objective_config(sweep_config)
    storage_url = sweep.get("storage_url")
    if storage_url is None:
        storage_url = f"sqlite:///{summary_store.root / 'optuna.db'}"
    return optuna.create_study(
        storage=storage_url,
        sampler=_build_optuna_sampler(optuna, sweep.get("sampler", {})),
        study_name=sweep.get("study_name"),
        direction=objective["direction"],
        load_if_exists=bool(sweep.get("load_if_exists", True)),
    )


def _build_optuna_sampler(optuna, sampler_config: Mapping[str, Any]):
    name = str(sampler_config.get("name", "tpe")).lower()
    params = {key: value for key, value in sampler_config.items() if key != "name"}
    if name == "tpe":
        return optuna.samplers.TPESampler(**params)
    if name == "random":
        return optuna.samplers.RandomSampler(**params)
    raise ConfigError(f"Unknown Optuna sampler: {name}")


def _named_arms(arms: list[SweepArm], sweep_config: Mapping[str, Any]) -> list[SweepArm]:
    template = sweep_config.get("run_name_template")
    if not template:
        return arms
    named: list[SweepArm] = []
    for index, arm in enumerate(arms, start=1):
        params = {key.replace(".", "_"): value for key, value in arm.params.items()}
        name = str(template).format(index=index, **params)
        named.append(SweepArm(name=name, config=arm.config, params=arm.params))
    return named


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _write_parquet(frame: pl.DataFrame, path: Path) -> Path | None:
    if frame.is_empty():
        return None
    frame.write_parquet(path)
    return path
