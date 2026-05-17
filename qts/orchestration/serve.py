"""Register Prefect data-refresh deployments."""

from __future__ import annotations

from qts.orchestration.flows.data_fetch_flow import data_fetch_flow
from qts.orchestration.prefect_compat import serve

_CONFIG = "config/live.yaml"

if __name__ == "__main__":
    serve(
        data_fetch_flow.to_deployment(
            "stock-ohlcv-daily",
            cron="0 21 * * 1-5",
            parameters={
                "config_path": _CONFIG,
                "asset_types": ["stock"],
                "data_types": ["ohlcv"],
            },
        ),
        data_fetch_flow.to_deployment(
            "vn-stock-ohlcv-daily",
            cron="0 9 * * 1-5",
            parameters={
                "config_path": _CONFIG,
                "asset_types": ["vn_stock"],
                "data_types": ["ohlcv"],
            },
        ),
        data_fetch_flow.to_deployment(
            "crypto-ohlcv-daily",
            cron="0 0 * * *",
            parameters={
                "config_path": _CONFIG,
                "asset_types": ["crypto"],
                "data_types": ["ohlcv"],
            },
        ),
        data_fetch_flow.to_deployment(
            "crypto-funding-8h",
            cron="0 */8 * * *",
            parameters={
                "config_path": _CONFIG,
                "asset_types": ["crypto"],
                "data_types": ["funding_rates"],
            },
        ),
        data_fetch_flow.to_deployment(
            "stock-fundamentals-weekly",
            cron="0 8 * * 1",
            parameters={
                "config_path": _CONFIG,
                "asset_types": ["stock"],
                "data_types": ["fundamentals"],
            },
        ),
        data_fetch_flow.to_deployment(
            "vn-stock-fundamentals-weekly",
            cron="0 8 * * 1",
            parameters={
                "config_path": _CONFIG,
                "asset_types": ["vn_stock"],
                "data_types": ["fundamentals"],
            },
        ),
    )
