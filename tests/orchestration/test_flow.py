from __future__ import annotations

from datetime import date

from qts.config.builder import Config
from qts.data.sources.binance import BinanceDataSource
from qts.data.sources.fmp import FMPDataSource
from qts.orchestration.flow import qts_flow


async def test_qts_flow_research_and_live(monkeypatch, config_dir, stock_ohlcv, crypto_ohlcv):
    original_build = Config.build

    def build_with_fixtures(path):
        resolved = original_build(path)
        resolved.stock_source = FMPDataSource(ohlcv_payloads={"AAPL": stock_ohlcv})
        resolved.crypto_source = BinanceDataSource(ohlcv_payloads={"BTC/USDT": crypto_ohlcv})
        return resolved

    monkeypatch.setattr(Config, "build", staticmethod(build_with_fixtures))
    research_result = await qts_flow(str(config_dir / "research.yaml"))
    live_result = await qts_flow(str(config_dir / "live.yaml"))
    assert research_result.metrics["sharpe"] == research_result.metrics["sharpe"]
    assert "fills" in live_result
