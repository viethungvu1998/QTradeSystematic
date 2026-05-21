from __future__ import annotations

from decimal import Decimal

import polars as pl
import pytest

from qts.core.instrument import AssetType, Instrument
from qts.core.portfolio import Position
from qts.execution.sync import PositionSync
from qts.orchestration.tasks.execution_tasks import sync_positions


@pytest.mark.asyncio
async def test_sync_positions_returns_orders_and_live_snapshot():
    aapl = Instrument("AAPL", AssetType.STOCK, "NASDAQ", "USD")
    msft = Instrument("MSFT", AssetType.STOCK, "NASDAQ", "USD")
    brokers = {
        AssetType.STOCK: Broker([Position(aapl, Decimal("10"), Decimal("50"))], Decimal("500")),
        AssetType.CRYPTO: Broker([Position(msft, Decimal("3"), Decimal("100"))], Decimal("300")),
    }

    orders, snapshot = await sync_positions(
        config=None,
        syncer=PositionSync(),
        brokers=brokers,
        target_weights={},
        data=pl.DataFrame(
            {
                "date": [1],
                "symbol": ["AAPL"],
                "close": [50.0],
            }
        ),
    )

    assert orders == []
    assert snapshot.equity == Decimal("800")
    assert [token.token for token in snapshot.tokens] == ["AAPL", "MSFT"]


class Broker:
    def __init__(self, positions: list[Position], account_value: Decimal) -> None:
        self._positions = positions
        self._account_value = account_value

    async def get_positions(self) -> list[Position]:
        return self._positions

    async def get_account_value(self) -> Decimal:
        return self._account_value
