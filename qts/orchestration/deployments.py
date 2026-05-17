"""Config-driven Prefect deployment registration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from qts.core.errors import ConfigError


@dataclass(frozen=True, slots=True)
class DeploymentSpec:
    name: str
    flow: str
    parameters: dict[str, Any] = field(default_factory=dict)
    cron: str | None = None


@dataclass(frozen=True, slots=True)
class DeploymentConfig:
    specs: list[DeploymentSpec]
    work_pool_name: str = "qts-process-pool"
    image: str | None = None
    build: bool = False
    push: bool = False


def load_deployment_specs(path: str | Path) -> list[DeploymentSpec]:
    return load_deployment_config(path).specs


def load_deployment_config(path: str | Path) -> DeploymentConfig:
    payload = yaml.safe_load(Path(path).read_text()) or {}
    deployments = payload.get("deployments", [])
    if not isinstance(deployments, list):
        raise ConfigError("deployments must be a list")
    return DeploymentConfig(
        specs=[_deployment_spec(item) for item in deployments],
        work_pool_name=str(payload.get("work_pool_name", "qts-process-pool")),
        image=payload.get("image"),
        build=bool(payload.get("build", False)),
        push=bool(payload.get("push", False)),
    )


def build_prefect_deployments(path: str | Path):
    from qts.orchestration.flow import qts_flow
    from qts.orchestration.flows.data_fetch_flow import data_fetch_flow

    flows = {
        "data_fetch": data_fetch_flow,
        "research": qts_flow,
    }
    deployments = []
    for spec in load_deployment_specs(path):
        try:
            flow_obj = flows[spec.flow]
        except KeyError as exc:
            raise ConfigError(f"Unknown deployment flow: {spec.flow}") from exc
        kwargs = {"parameters": spec.parameters}
        if spec.cron:
            kwargs["cron"] = spec.cron
        deployments.append(flow_obj.to_deployment(spec.name, **kwargs))
    return deployments


def deploy_deployment_config(path: str | Path):
    from qts.orchestration.prefect_compat import deploy

    config = load_deployment_config(path)
    deployments = build_prefect_deployments(path)
    kwargs: dict[str, Any] = {
        "work_pool_name": config.work_pool_name,
        "build": config.build,
        "push": config.push,
    }
    if config.image:
        kwargs["image"] = config.image
    return deploy(*deployments, **kwargs)


def serve_deployment_config(path: str | Path):
    from qts.orchestration.prefect_compat import serve

    return serve(*build_prefect_deployments(path))


def _deployment_spec(value: object) -> DeploymentSpec:
    if not isinstance(value, dict):
        raise ConfigError("Each deployment must be a mapping")
    return DeploymentSpec(
        name=str(value["name"]),
        flow=str(value["flow"]),
        parameters=dict(value.get("parameters", {})),
        cron=value.get("cron"),
    )


if __name__ == "__main__":
    import sys

    deploy_deployment_config(sys.argv[1] if len(sys.argv) > 1 else "configs/deployments/local.yaml")
