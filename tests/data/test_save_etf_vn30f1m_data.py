from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import polars as pl

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "notebooks" / "save_etf_vn30f1m_data.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("save_etf_vn30f1m_data", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.urls = []

    def get(self, url, headers=None):
        self.urls.append(url)
        return _FakeResponse(self.payload)


def test_list_all_etfs_normalizes_kbs_payload_to_qts_symbols():
    script = _load_script()
    client = _FakeClient(
        {
            "status": 200,
            "data": ["e1vfvn30", "VN:FUEVFVND", {"symbol": "fuessv30"}, "E1VFVN30"],
        }
    )

    symbols = script.list_all_etfs(client=client)

    assert symbols == ["VN:E1VFVN30", "VN:FUESSV30", "VN:FUEVFVND"]
    assert client.urls == [script.ETF_LIST_URL]
    assert script.DEFAULT_START_DATE == date(2021, 1, 1)


def test_write_outputs_uses_data_vni_etf_csv_files(tmp_path):
    script = _load_script()
    result = script.CrawlResult(
        etf_symbols=["VN:E1VFVN30", "VN:FUEVFVND"],
        etf_prices=pl.DataFrame(
            {
                "date": [date(2026, 1, 2), date(2026, 1, 2)],
                "symbol": ["VN:E1VFVN30", "VN:FUEVFVND"],
                "open": [1.0, 2.0],
                "high": [1.0, 2.0],
                "low": [1.0, 2.0],
                "close": [1.0, 2.0],
                "volume": [100.0, 200.0],
            }
        ),
        vn30f1m_prices=pl.DataFrame(
            {
                "date": [date(2026, 1, 2)],
                "symbol": ["VNF:VN30F1M"],
                "open": [1_000.0],
                "high": [1_001.0],
                "low": [999.0],
                "close": [1_000.5],
                "volume": [10.0],
            }
        ),
        failures=pl.DataFrame(schema={"symbol": pl.Utf8, "asset": pl.Utf8, "error": pl.Utf8}),
        output_dir=tmp_path,
    )
    (tmp_path / "etf_prices_1d.csv").write_text("stale combined file")

    script._write_outputs(
        result,
        start=date(2026, 1, 1),
        end=date(2026, 1, 2),
        etf_interval="1d",
        vn30f1m_interval="1d",
    )

    assert sorted(path.name for path in tmp_path.glob("*.csv")) == [
        "crawl_failures.csv",
        "crawl_summary.csv",
        "etf_symbols.csv",
        "vn30f1m_prices_1d.csv",
    ]
    assert sorted(path.name for path in (tmp_path / "etf_prices").glob("*.csv")) == [
        "E1VFVN30_1d.csv",
        "FUEVFVND_1d.csv",
    ]
    assert not (tmp_path / "etf_prices_1d.csv").exists()
    assert pl.read_csv(tmp_path / "crawl_summary.csv")["etf_price_rows"][0] == 2
    assert pl.read_csv(tmp_path / "etf_prices" / "FUEVFVND_1d.csv")["symbol"].to_list() == [
        "VN:FUEVFVND"
    ]
