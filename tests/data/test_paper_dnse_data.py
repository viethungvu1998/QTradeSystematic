"""Live DNSE API tests — fetches real historical data for each VN asset type.

Requires DNSE_API_KEY and DNSE_API_SECRET in .env (or environment).
Run with:  pytest tests/data/test_paper_dnse_data.py -v -s
"""

from __future__ import annotations

from datetime import date

import pytest
from dotenv import load_dotenv

from qts.data._schemas import DataType, OHLCV_COLUMNS
from qts.data.sources.dnse import DNSEDataSource

load_dotenv()

_START = date(2025, 1, 1)
_END = date(2025, 3, 31)


@pytest.fixture(scope="module")
def dnse() -> DNSEDataSource:
    return DNSEDataSource.from_env()


@pytest.mark.asyncio
async def test_dnse_vn_stock_ohlcv(dnse):
    """VN equity: VNM on HOSE."""
    result = await dnse.get_ohlcv("VN:VNM", _START, _END, "1d")
    print(f"\n[VN_STOCK] VN:VNM  rows={result.height}")
    print(result.head(3))
    assert result.columns == OHLCV_COLUMNS
    assert result.height > 0
    assert (result["symbol"] == "VN:VNM").all()
    assert result["close"].min() > 0


@pytest.mark.asyncio
async def test_dnse_vn_warrant_ohlcv(dnse):
    """Covered warrant: fetch active CW on VNM (symbol depends on expiry)."""
    # CVNM2403 expired — use a symbol that was active in Q1 2025.
    # Adjust if no active CW exists; DNSE returns empty arrays for expired symbols.
    result = await dnse.get_ohlcv("VNW:CVNM2501", _START, _END, "1d")
    print(f"\n[VN_WARRANT] VNW:CVNM2501  rows={result.height}")
    if result.height > 0:
        print(result.head(3))
        assert result.columns == OHLCV_COLUMNS
        assert (result["symbol"] == "VNW:CVNM2501").all()
    else:
        print("  (no data — warrant may have expired; schema still valid)")
        assert result.columns == OHLCV_COLUMNS


@pytest.mark.asyncio
async def test_dnse_vn_futures_ohlcv(dnse):
    """VN30 index futures: active front-month contract."""
    result = await dnse.get_ohlcv("VNF:VN30F2503", _START, _END, "1d")
    print(f"\n[VN_FUTURES] VNF:VN30F2503  rows={result.height}")
    print(result.head(3))
    assert result.columns == OHLCV_COLUMNS
    assert result.height > 0
    assert (result["symbol"] == "VNF:VN30F2503").all()
    assert result["close"].min() > 0


@pytest.mark.asyncio
async def test_dnse_futures_via_fetch_dispatch(dnse):
    """Confirm fetch() routes FUTURES_OHLCV correctly."""
    result = await dnse.fetch(
        DataType.FUTURES_OHLCV, "VNF:VN30F2503",
        start=_START, end=_END, interval="1d",
    )
    assert result.height > 0


@pytest.mark.asyncio
async def test_dnse_intraday_resolution(dnse):
    """15-minute bars for a single day."""
    result = await dnse.get_ohlcv("VN:VNM", date(2025, 3, 3), date(2025, 3, 3), "15m")
    print(f"\n[VN_STOCK intraday] VN:VNM 15m  rows={result.height}")
    if result.height > 0:
        print(result.head(5))
    assert result.columns == OHLCV_COLUMNS
