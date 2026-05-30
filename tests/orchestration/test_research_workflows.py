from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from types import SimpleNamespace

import polars as pl
import pytest

from qts.core.errors import ConfigError
from qts.orchestration import research
from qts.research.backtest.base import BacktestResult


def _base_runtime_config() -> dict[str, object]:
    return {
        "workflow": "research",
        "asset_types": ["stock"],
        "universe": {"stock": ["AAPL"]},
        "start_date": "2024-01-01",
        "end_date": "2024-03-20",
        "initial_capital": 100000,
        "data_sources": {"stock": "fmp"},
        "storage": "duckdb",
        "features": {"technical": True},
        "strategy": {"type": "factor", "params": {}},
        "backtest_engine": "vectorbt",
    }


def _result(value: float = 1.0) -> BacktestResult:
    return BacktestResult(
        engine_name="vectorbt",
        metrics={"sharpe": value, "sortino": value / 2, "cagr": value / 10},
        returns=pl.DataFrame({"date": [date(2024, 1, 1)], "portfolio_return": [value]}),
        equity_curve=pl.DataFrame({"date": [date(2024, 1, 1)], "equity": [100000.0 + value]}),
        signals=pl.DataFrame(
            {"date": [date(2024, 1, 1)], "symbol": ["AAPL"], "signal": [1], "weight": [1.0]}
        ),
    )


def test_deep_overlay_merge_merges_nested_mappings_and_replaces_lists():
    base = {
        "universe": {"stock": ["AAPL"], "crypto": ["BTC/USDT"]},
        "strategy": {"params": {"long_quantile": 0.7, "short_quantile": 0.3}},
    }
    overlay = {
        "universe": {"stock": ["MSFT"]},
        "strategy": {"params": {"long_quantile": 0.8}},
    }

    merged = research.deep_overlay_merge(base, overlay)

    assert merged == {
        "universe": {"stock": ["MSFT"], "crypto": ["BTC/USDT"]},
        "strategy": {"params": {"long_quantile": 0.8, "short_quantile": 0.3}},
    }
    assert base["universe"]["stock"] == ["AAPL"]


def test_expand_grid_sweep_uses_dotted_paths_without_mutating_base():
    base = _base_runtime_config()
    sweep = {
        "sweep": {
            "mode": "grid",
            "axes": [
                {"path": "strategy.params.long_quantile", "values": [0.7, 0.8]},
                {"path": "features.forward_returns.periods", "values": [[1], [5]]},
            ],
        }
    }

    arms = research.expand_grid_sweep(base, sweep)

    assert len(arms) == 4
    assert arms[0].params == {
        "strategy.params.long_quantile": 0.7,
        "features.forward_returns.periods": [1],
    }
    assert arms[-1].config["strategy"]["params"]["long_quantile"] == 0.8
    assert arms[-1].config["features"]["forward_returns"]["periods"] == [5]
    assert "long_quantile" not in base["strategy"]["params"]


def test_expand_grid_sweep_can_override_ml_factor_model_params():
    base = _base_runtime_config()
    base["strategy"] = {
        "type": "ml_factor",
        "params": {
            "model": {
                "name": "xgb_classifier",
                "params": {"n_estimators": 50, "learning_rate": 0.05},
            },
        },
    }
    sweep = {
        "sweep": {
            "mode": "grid",
            "axes": [
                {"path": "strategy.params.model.params.learning_rate", "values": [0.01, 0.05]},
                {"path": "strategy.params.model.params.n_estimators", "values": [50, 100]},
            ],
        }
    }

    arms = research.expand_grid_sweep(base, sweep)

    assert len(arms) == 4
    assert arms[0].params == {
        "strategy.params.model.params.learning_rate": 0.01,
        "strategy.params.model.params.n_estimators": 50,
    }
    assert arms[-1].config["strategy"]["params"]["model"]["params"] == {
        "n_estimators": 100,
        "learning_rate": 0.05,
    }
    assert base["strategy"]["params"]["model"]["params"]["n_estimators"] == 50


class FakeTrial:
    def __init__(self, number: int) -> None:
        self.number = number
        self.calls = []

    def suggest_float(self, name, low, high, *, step=None, log=False):
        self.calls.append(("float", name, low, high, step, log))
        return high if self.number else low

    def suggest_int(self, name, low, high, *, step=1, log=False):
        self.calls.append(("int", name, low, high, step, log))
        return high if self.number else low

    def suggest_categorical(self, name, choices):
        self.calls.append(("categorical", name, list(choices)))
        return list(choices)[self.number % len(choices)]


def test_suggest_optuna_params_supports_float_int_and_categorical():
    trial = FakeTrial(number=0)
    search_space = [
        {
            "path": "strategy.params.long_quantile",
            "type": "float",
            "low": 0.6,
            "high": 0.9,
            "step": 0.1,
        },
        {
            "path": "strategy.params.model.params.max_depth",
            "type": "int",
            "low": 2,
            "high": 6,
        },
        {
            "path": "strategy.params.portfolio.name",
            "type": "categorical",
            "choices": ["equal_weight", "risk_parity"],
        },
    ]

    params = research.suggest_optuna_params(trial, search_space)

    assert params == {
        "strategy.params.long_quantile": 0.6,
        "strategy.params.model.params.max_depth": 2,
        "strategy.params.portfolio.name": "equal_weight",
    }
    assert trial.calls == [
        ("float", "strategy.params.long_quantile", 0.6, 0.9, 0.1, False),
        ("int", "strategy.params.model.params.max_depth", 2, 6, 1, False),
        ("categorical", "strategy.params.portfolio.name", ["equal_weight", "risk_parity"]),
    ]


@pytest.mark.asyncio
async def test_run_research_config_reuses_resolved_execution_path(monkeypatch):
    resolved = SimpleNamespace(raw=SimpleNamespace(workflow="research"))
    captured = {}

    def fake_build_from_mapping(config):
        captured["config"] = config
        return resolved

    async def fake_run_resolved_config(value):
        captured["resolved"] = value
        return _result()

    monkeypatch.setattr(
        research.Config,
        "build_from_mapping",
        staticmethod(fake_build_from_mapping),
    )
    monkeypatch.setattr(research, "run_resolved_config", fake_run_resolved_config)

    result = await research.run_research_config(_base_runtime_config())

    assert result.metrics["sharpe"] == 1.0
    assert captured["resolved"] is resolved


@pytest.mark.asyncio
async def test_run_research_config_rejects_non_research_workflows(monkeypatch):
    resolved = SimpleNamespace(raw=SimpleNamespace(workflow="validation"))
    monkeypatch.setattr(
        research.Config,
        "build_from_mapping",
        staticmethod(lambda config: resolved),
    )

    with pytest.raises(ConfigError, match="research"):
        await research.run_research_config({"workflow": "validation"})


@dataclass(slots=True)
class RecordingTracker:
    events: list[tuple[str, str, bool]]
    metrics: list[dict[str, float]]
    params: list[dict[str, object]]

    @contextmanager
    def start_run(self, *, run_name: str, nested: bool = False, tags=None):
        self.events.append(("start", run_name, nested))
        yield run_name
        self.events.append(("end", run_name, nested))

    def log_params(self, params):
        self.params.append(dict(params))

    def log_metrics(self, metrics):
        self.metrics.append(dict(metrics))

    def log_artifact(self, path):
        return None


@pytest.mark.asyncio
async def test_run_sweep_config_tracks_children_and_persists_summary(monkeypatch, tmp_path):
    base_path = tmp_path / "base.yaml"
    sweep_path = tmp_path / "sweep.yaml"
    base_path.write_text(
        """
workflow: research
asset_types: [stock]
universe:
  stock: [AAPL]
start_date: "2024-01-01"
end_date: "2024-03-20"
initial_capital: 100000
data_sources:
  stock: fmp
storage: duckdb
features:
  technical: true
strategy:
  type: factor
  params: {}
backtest_engine: vectorbt
"""
    )
    sweep_path.write_text(
        f"""
base_config: {base_path}
tracking:
  experiment_name: qts-tests
  run_name: parent-run
storage:
  root: {tmp_path / "runs"}
sweep:
  mode: grid
  axes:
    - path: strategy.params.long_quantile
      values: [0.7, 0.8]
"""
    )

    calls = []

    async def fake_run_research_config(
        config,
        *,
        tracker=None,
        run_name=None,
        nested=False,
        summary_store=None,
        params=None,
    ):
        calls.append((run_name, nested, dict(params or {})))
        return _result(float(config["strategy"]["params"]["long_quantile"]))

    tracker = RecordingTracker(events=[], metrics=[], params=[])
    monkeypatch.setattr(research, "run_research_config", fake_run_research_config)

    result = await research.run_sweep_config(sweep_path, tracker=tracker)

    assert [arm.params["strategy.params.long_quantile"] for arm in result.arms] == [0.7, 0.8]
    assert tracker.events == [
        ("start", "parent-run", False),
        ("end", "parent-run", False),
    ]
    assert calls == [
        ("arm-001", True, {"strategy.params.long_quantile": 0.7}),
        ("arm-002", True, {"strategy.params.long_quantile": 0.8}),
    ]
    assert (tmp_path / "runs" / "summary.duckdb").exists()
    assert len(result.results) == 2


class FakeSampler:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class FakeStudy:
    def __init__(self, direction: str) -> None:
        self.direction = direction
        self.trials = []
        self.told = []
        self.best_params = {}
        self.best_value = None

    def ask(self):
        trial = FakeTrial(number=len(self.trials))
        self.trials.append(trial)
        return trial

    def tell(self, trial, value):
        self.told.append((trial.number, value))
        better = (
            self.best_value is None
            or (self.direction == "maximize" and value > self.best_value)
            or (self.direction == "minimize" and value < self.best_value)
        )
        if better:
            self.best_value = value
            self.best_params = dict(getattr(trial, "params", {}))


class FakeOptuna:
    samplers = SimpleNamespace(TPESampler=FakeSampler, RandomSampler=FakeSampler)

    def __init__(self) -> None:
        self.created = []
        self.study = None

    def create_study(self, **kwargs):
        self.created.append(kwargs)
        self.study = FakeStudy(direction=kwargs["direction"])
        return self.study


@pytest.mark.asyncio
async def test_run_sweep_config_uses_optuna_as_primary_engine(monkeypatch, tmp_path):
    base_path = tmp_path / "base.yaml"
    sweep_path = tmp_path / "sweep.yaml"
    base_path.write_text(
        """
workflow: research
asset_types: [stock]
universe:
  stock: [AAPL]
start_date: "2024-01-01"
end_date: "2024-03-20"
initial_capital: 100000
data_sources:
  stock: fmp
storage: duckdb
features:
  technical: true
strategy:
  type: factor
  params: {}
backtest_engine: vectorbt
"""
    )
    sweep_path.write_text(
        f"""
base_config: {base_path}
tracking:
  run_name: optuna-parent
storage:
  root: {tmp_path / "runs"}
sweep:
  mode: optuna
  n_trials: 2
  study_name: qts-test-study
  sampler:
    name: tpe
    seed: 42
  objective:
    metric: sharpe
    direction: maximize
  search_space:
    - path: strategy.params.long_quantile
      type: float
      low: 0.7
      high: 0.8
"""
    )
    fake_optuna = FakeOptuna()
    calls = []

    async def fake_run_research_config(
        config,
        *,
        tracker=None,
        run_name=None,
        nested=False,
        summary_store=None,
        params=None,
    ):
        value = float(config["strategy"]["params"]["long_quantile"])
        calls.append((run_name, nested, dict(params or {}), value))
        return _result(value)

    tracker = RecordingTracker(events=[], metrics=[], params=[])
    monkeypatch.setattr(research, "_load_optuna", lambda: fake_optuna)
    monkeypatch.setattr(research, "run_research_config", fake_run_research_config)

    result = await research.run_sweep_config(sweep_path, tracker=tracker)

    assert fake_optuna.created[0]["direction"] == "maximize"
    assert fake_optuna.created[0]["study_name"] == "qts-test-study"
    assert fake_optuna.study.told == [(0, 0.7), (1, 0.8)]
    assert calls == [
        ("trial-000", True, {"strategy.params.long_quantile": 0.7}, 0.7),
        ("trial-001", True, {"strategy.params.long_quantile": 0.8}, 0.8),
    ]
    assert [arm.params["strategy.params.long_quantile"] for arm in result.arms] == [0.7, 0.8]
    assert result.best_value == 0.8
