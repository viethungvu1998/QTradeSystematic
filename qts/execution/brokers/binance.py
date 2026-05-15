"""Binance broker adapter."""

from __future__ import annotations

import os
from decimal import Decimal

from qts.core.errors import BrokerError
from qts.core.instrument import AssetType, Instrument
from qts.core.order import Fill, Order, OrderSide
from qts.core.portfolio import Position
from qts.core.registry import Registry
from qts.execution.base import BaseBroker

_DEMO_BASE_URL = "https://testnet.binance.vision"

_SIDE = {OrderSide.BUY: "BUY", OrderSide.SELL: "SELL"}
_TYPE = {"market": "MARKET", "limit": "LIMIT", "stop": "STOP_LOSS", "stop_limit": "STOP_LOSS_LIMIT"}


def _to_binance_symbol(symbol: str) -> str:
    return symbol.replace("/", "")


@Registry.register_broker("binance")
class BinanceBroker(BaseBroker):
    """Binance REST broker. Use from_env() for real credentials; pass client=None for paper."""

    def __init__(self, client=None) -> None:
        self._client = client
        self.connected = False
        self._order_symbols: dict[str, str] = {}  # order_id → binance symbol, needed for cancel

    @classmethod
    def from_env(cls, mode: str = "demo") -> BinanceBroker:
        """Build from env vars. mode='demo' uses testnet; mode='live' uses production."""
        from binance.spot import Spot  # noqa: PLC0415

        if mode == "live":
            client = Spot(
                api_key=os.environ["BINANCE_TRADING_KEY"],
                api_secret=os.environ["BINANCE_TRADING_SECRET_KEY"],
            )
        else:
            client = Spot(
                api_key=os.environ["BINANCE_DEMO_TRADING_API_KEY"],
                api_secret=os.environ["BINANCE_DEMO_TRADING_SECRET_KEY"],
                base_url=_DEMO_BASE_URL,
            )
        return cls(client=client)

    async def connect(self) -> None:
        if self._client is not None:
            try:
                self._client.account()
            except Exception as exc:
                raise BrokerError(f"Binance connect failed: {exc}") from exc
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    async def get_positions(self) -> list[Position]:
        if self._client is None:
            return []
        try:
            info = self._client.account(omitZeroBalances="true")
        except Exception as exc:
            raise BrokerError(f"Binance get_positions failed: {exc}") from exc
        positions = []
        for balance in info.get("balances", []):
            asset = balance["asset"]
            qty = Decimal(balance["free"]) + Decimal(balance["locked"])
            if qty == 0 or asset == "USDT":
                continue
            try:
                ticker = self._client.ticker_price(f"{asset}USDT")
                price = Decimal(ticker["price"])
            except Exception:
                price = Decimal("0")
            instrument = Instrument(
                symbol=f"{asset}/USDT",
                asset_type=AssetType.CRYPTO,
                exchange="BINANCE",
                currency="USDT",
            )
            positions.append(Position(instrument=instrument, quantity=qty, market_price=price))
        return positions

    async def place_order(self, order: Order) -> Fill:
        if self._client is None:
            return Fill(
                order_id=order.client_order_id or "paper-binance",
                instrument=order.instrument,
                side=order.side,
                quantity=order.quantity,
                price=order.limit_price or Decimal("0"),
            )
        binance_sym = _to_binance_symbol(order.instrument.symbol)
        side = _SIDE[order.side]
        order_type = _TYPE.get(order.order_type.value, "MARKET")
        kwargs: dict = {"quantity": str(order.quantity), "newOrderRespType": "FULL"}
        if order.limit_price is not None:
            kwargs["price"] = str(order.limit_price)
            kwargs["timeInForce"] = "GTC"
        if order.client_order_id:
            kwargs["newClientOrderId"] = order.client_order_id
        try:
            resp = self._client.new_order(binance_sym, side, order_type, **kwargs)
        except Exception as exc:
            raise BrokerError(str(exc), order) from exc
        order_id = str(resp["orderId"])
        self._order_symbols[order_id] = binance_sym
        fills = resp.get("fills", [])
        if fills:
            exec_qty = sum(Decimal(f["qty"]) for f in fills)
            avg_price = sum(Decimal(f["price"]) * Decimal(f["qty"]) for f in fills) / exec_qty
            commission = sum(Decimal(f["commission"]) for f in fills)
        else:
            exec_qty = Decimal(str(resp.get("executedQty", str(order.quantity))))
            avg_price = Decimal(str(resp.get("price", str(order.limit_price or "0"))))
            commission = Decimal("0")
        return Fill(
            order_id=order_id,
            instrument=order.instrument,
            side=order.side,
            quantity=exec_qty,
            price=avg_price,
            commission=commission,
        )

    async def cancel_order(self, order_id: str) -> None:
        if self._client is None:
            return
        symbol = self._order_symbols.get(order_id)
        if symbol is None:
            raise BrokerError(f"Cannot cancel {order_id}: symbol unknown (was place_order called?)")
        try:
            self._client.cancel_order(symbol, orderId=int(order_id))
        except Exception as exc:
            raise BrokerError(str(exc)) from exc

    async def get_account_value(self) -> Decimal:
        if self._client is None:
            return Decimal("0")
        try:
            info = self._client.account(omitZeroBalances="true")
        except Exception as exc:
            raise BrokerError(f"Binance get_account_value failed: {exc}") from exc
        total = Decimal("0")
        for balance in info.get("balances", []):
            asset = balance["asset"]
            qty = Decimal(balance["free"]) + Decimal(balance["locked"])
            if asset == "USDT":
                total += qty
                continue
            try:
                ticker = self._client.ticker_price(f"{asset}USDT")
                total += qty * Decimal(ticker["price"])
            except Exception:
                pass  # skip assets with no USDT pair
        return total
