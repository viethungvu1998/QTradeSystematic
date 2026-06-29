"""Schemas and validation for manifest-driven autoresearch."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

TARGET_PREFIXES = ("forward_return", "future_", "target_")
WORKTREE_SNAPSHOT_FILE = "worktree_snapshot.json"
NEXT_STEPS_FILE = "next_steps.md"
DEFAULT_ACTION_ALIASES = {
    "parameter_tuning": {"parameter_tuning", "parameters", "tweak_parameters", "model_params"},
    "hyperparameter_optimization": {
        "hyperparameter_optimization",
        "hyperparams",
        "hyperparam_opt",
        "hyperopt",
    },
    "feature_engineering": {
        "feature_engineering",
        "features",
        "non_leaking_features",
    },
    "strategy_model_update": {
        "strategy_model_update",
        "strategy_model",
        "model_update",
        "trainer",
    },
}
SUPPORTED_OPS = {">", ">=", "<", "<=", "=="}


class AutoresearchError(RuntimeError):
    """Raised for user-correctable autoresearch setup errors."""


@dataclass(frozen=True)
class ObjectiveSpec:
    metric: str
    direction: str = "maximize"
    min_delta: float = 0.0


@dataclass(frozen=True)
class CriteriaTarget:
    metric: str
    op: str
    value: float


@dataclass(frozen=True)
class CriteriaSpec:
    mode: str = "any"
    targets: tuple[CriteriaTarget, ...] = ()


@dataclass(frozen=True)
class QuotaSpec:
    max_trials: int


@dataclass(frozen=True)
class PathsSpec:
    artifact_root: str
    allowed_edit_roots: tuple[str, ...]
    run_root: str | None = None


@dataclass(frozen=True)
class CommandSpec:
    argv: tuple[str, ...]


@dataclass(frozen=True)
class ResearchAction:
    id: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResearchManifest:
    version: int
    run_tag: str
    strategy: str
    algorithm: str
    quota: QuotaSpec
    objective: ObjectiveSpec
    criteria: CriteriaSpec
    paths: PathsSpec
    command: CommandSpec
    research_actions: tuple[ResearchAction, ...] = ()
    next_step_policy: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ResearchManifest:
        validate_manifest_mapping(data)
        objective = data["objective"]
        criteria = data.get("criteria", {})
        paths = data["paths"]
        research_actions = tuple(
            ResearchAction(action_id, tuple(aliases))
            for action_id, aliases in (
                research_action_entry(action)
                for action in data.get("research_limits", {}).get("allowed_actions", [])
            )
            if action_id
        )
        return cls(
            version=int(data["version"]),
            run_tag=str(data["run_tag"]),
            strategy=str(data["strategy"]),
            algorithm=str(data["algorithm"]),
            quota=QuotaSpec(max_trials=int(data["quota"]["max_trials"])),
            objective=ObjectiveSpec(
                metric=str(objective["metric"]),
                direction=str(objective.get("direction", "maximize")),
                min_delta=float(objective.get("min_delta", 0.0) or 0.0),
            ),
            criteria=CriteriaSpec(
                mode=str(criteria.get("mode", "any")),
                targets=tuple(
                    CriteriaTarget(
                        metric=str(target["metric"]),
                        op=str(target["op"]),
                        value=float(target["value"]),
                    )
                    for target in criteria.get("targets", [])
                ),
            ),
            paths=PathsSpec(
                artifact_root=str(paths["artifact_root"]),
                run_root=None if paths.get("run_root") is None else str(paths["run_root"]),
                allowed_edit_roots=tuple(str(item) for item in paths["allowed_edit_roots"]),
            ),
            command=CommandSpec(argv=tuple(str(item) for item in data["command"]["argv"])),
            research_actions=research_actions,
            next_step_policy=dict(data.get("next_step_policy") or {}),
            raw=dict(data),
        )


@dataclass(frozen=True)
class AutoresearchContext:
    manifest_path: Path
    manifest_dir: Path
    manifest: ResearchManifest
    repo_root: Path
    artifact_root: Path
    artifact_root_rel: str
    run_root: Path
    run_root_rel: str
    allowed_edit_roots: tuple[str, ...]
    results_path: Path
    events_path: Path
    summary_path: Path
    snapshot_path: Path
    next_steps_path: Path


def load_manifest(path: Path) -> ResearchManifest:
    """Load and validate an autoresearch manifest."""
    if not path.exists():
        raise AutoresearchError(f"manifest not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise AutoresearchError("manifest must be a mapping")
    return ResearchManifest.from_mapping(data)


def load_context(manifest_path: Path, repo_root_arg: Path | None = None) -> AutoresearchContext:
    """Load a manifest and resolve run paths."""
    from agentic_quant_researcher.utils.paths import find_repo_root, rel_path, resolve_repo_path

    manifest_path = manifest_path.resolve()
    repo_root = repo_root_arg.resolve() if repo_root_arg else find_repo_root(manifest_path)
    manifest = load_manifest(manifest_path)
    artifact_root = resolve_repo_path(repo_root, manifest.paths.artifact_root)
    run_root = resolve_repo_path(
        repo_root,
        manifest.paths.run_root if manifest.paths.run_root else artifact_root / "runs",
    )
    allowed_roots = tuple(
        rel_path(resolve_repo_path(repo_root, item), repo_root)
        for item in manifest.paths.allowed_edit_roots
    )
    return AutoresearchContext(
        manifest_path=manifest_path,
        manifest_dir=manifest_path.parent,
        manifest=manifest,
        repo_root=repo_root,
        artifact_root=artifact_root,
        artifact_root_rel=rel_path(artifact_root, repo_root),
        run_root=run_root,
        run_root_rel=rel_path(run_root, repo_root),
        allowed_edit_roots=allowed_roots,
        results_path=artifact_root / "results.tsv",
        events_path=artifact_root / "events.jsonl",
        summary_path=artifact_root / "summary.md",
        snapshot_path=artifact_root / WORKTREE_SNAPSHOT_FILE,
        next_steps_path=artifact_root / NEXT_STEPS_FILE,
    )


def validate_manifest_mapping(data: dict[str, Any]) -> None:
    required = [
        "version",
        "run_tag",
        "strategy",
        "algorithm",
        "quota",
        "objective",
        "criteria",
        "paths",
        "command",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise AutoresearchError(f"manifest missing keys: {', '.join(missing)}")
    if int(data["version"]) != 1:
        raise AutoresearchError("manifest version must be 1")
    quota = data.get("quota", {})
    if int(quota.get("max_trials", 0)) < 1:
        raise AutoresearchError("quota.max_trials must be >= 1")
    objective = data.get("objective", {})
    if objective.get("direction", "maximize") not in {"maximize", "minimize"}:
        raise AutoresearchError("objective.direction must be maximize or minimize")
    if not objective.get("metric"):
        raise AutoresearchError("objective.metric is required")
    criteria = data.get("criteria", {})
    if criteria.get("mode", "any") not in {"any", "all"}:
        raise AutoresearchError("criteria.mode must be any or all")
    for target in criteria.get("targets", []):
        if target.get("op") not in SUPPORTED_OPS:
            raise AutoresearchError(f"unsupported criteria op: {target.get('op')}")
        if "metric" not in target or "value" not in target:
            raise AutoresearchError("criteria targets require metric, op, and value")
    paths = data.get("paths", {})
    if not paths.get("artifact_root"):
        raise AutoresearchError("paths.artifact_root is required")
    if paths.get("run_root") is not None and not str(paths.get("run_root")).strip():
        raise AutoresearchError("paths.run_root must not be empty when provided")
    if not paths.get("allowed_edit_roots"):
        raise AutoresearchError("paths.allowed_edit_roots must not be empty")
    argv = data.get("command", {}).get("argv")
    if not isinstance(argv, list) or not argv:
        raise AutoresearchError("command.argv must be a non-empty list")
    validate_research_limits(data)
    validate_predictor_names(data)


def validate_research_limits(data: dict[str, Any]) -> None:
    seen: set[str] = set()
    for action in data.get("research_limits", {}).get("allowed_actions", []):
        action_id, _ = research_action_entry(action)
        if not action_id:
            raise AutoresearchError("research_limits.allowed_actions entries need an id")
        if action_id in seen:
            raise AutoresearchError(f"duplicate research action id: {action_id}")
        seen.add(action_id)


def validate_predictor_names(data: dict[str, Any]) -> None:
    forbidden = [
        str(column)
        for column in data.get("predictor_cols", [])
        if str(column).startswith(TARGET_PREFIXES)
    ]
    if forbidden:
        raise AutoresearchError(
            "predictor_cols contains future-looking columns: " + ", ".join(forbidden)
        )


def canonical_research_action(value: str, manifest: ResearchManifest) -> str:
    normalized = normalize_action(value)
    action_map = allowed_research_action_map(manifest)
    if not action_map:
        return normalized
    if normalized in action_map:
        return action_map[normalized]
    allowed = ", ".join(sorted(set(action_map.values())))
    raise AutoresearchError(
        f"research action is not allowed: {value}. Allowed actions: {allowed}"
    )


def allowed_research_action_map(manifest: ResearchManifest) -> dict[str, str]:
    action_map: dict[str, str] = {}
    for action in manifest.research_actions:
        alias_values = set(action.aliases)
        alias_values.add(action.id)
        alias_values.update(DEFAULT_ACTION_ALIASES.get(action.id, set()))
        for alias in alias_values:
            action_map[normalize_action(alias)] = action.id
    return action_map


def research_action_entry(action: object) -> tuple[str, list[str]]:
    if isinstance(action, str):
        return normalize_action(action), []
    if not isinstance(action, dict):
        return "", []
    action_id = normalize_action(str(action.get("id", "")))
    aliases = [normalize_action(str(alias)) for alias in action.get("aliases", [])]
    return action_id, aliases


def normalize_action(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
