from __future__ import annotations

from qts.orchestration.deployments import load_deployment_specs


def test_load_deployment_specs_from_config(tmp_path):
    path = tmp_path / "deployments.yaml"
    path.write_text(
        """
deployments:
  - name: stock-ohlcv-daily
    flow: data_fetch
    cron: "0 21 * * 1-5"
    parameters:
      config_path: configs/strategies/factor/base.yaml
      asset_types: [stock]
      data_types: [ohlcv]
  - name: factor-research-manual
    flow: research
    parameters:
      config_path: configs/strategies/factor/base.yaml
"""
    )

    specs = load_deployment_specs(path)

    assert [spec.name for spec in specs] == ["stock-ohlcv-daily", "factor-research-manual"]
    assert specs[0].flow == "data_fetch"
    assert specs[0].parameters["asset_types"] == ["stock"]
    assert specs[1].cron is None
