"""Register Prefect deployments from repo config."""

from __future__ import annotations

from pathlib import Path

from qts.orchestration.deployments import deploy_deployment_config

_CONFIG = Path("configs/deployments/local.yaml")

if __name__ == "__main__":
    deploy_deployment_config(_CONFIG)
