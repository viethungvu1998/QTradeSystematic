"""DNSE (Entrade X) broker adapter for VN equities and VN30 futures.

Auth flow:
  1. connect() → POST /auth-service/login → JWT token
  2. set_trading_token(otp) → POST /order-service/trading-token → trading token
  3. place_order / cancel_order require both tokens

Stock orders:    POST/DELETE /order-service/v2/orders
Derivative orders: POST/DELETE /order-service/derivative/orders
"""

from __future__ import annotations

import os
from decimal import Decimal

from qts.core.errors import BrokerError
from qts.core.instrument import AssetType, Instrument
from qts.core.order import Fill, Order, OrderSide
from qts.core.portfolio import Position
from qts.core.registry import Registry
from qts.execution.base import BaseBroker

_BASE_URL = "https://api.dnse.com.vn"
_SIDE = {OrderSide.BUY: "NB", OrderSide.SELL: "NS"}


def _is_derivative(symbol: str) -> bool:
    return AssetType.from_symbol(symbol) is AssetType.VN_FUTURES


@Registry.register_broker("dnse")
class DNSEBroker(BaseBroker):
    """DNSE broker for VN equities and VN30 futures.

    For paper / unit testing pass a *client* object that implements the same
    interface as this class. For live use, call from_env() or supply
    username/password directly.

    A trading token (OTP-gated) is required before placing or cancelling orders.
    Call set_trading_token(otp) after connect().
    """

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        sub_account: str | None = None,
        client=None,
    ) -> None:
        self._username = username
        self._password = password
        self._sub_account = sub_account
        self._client = client
        self._jwt_token: str | None = None
        self._trading_token: str | None = None
        self._order_meta: dict[str, dict] = {}  # order_id → {account_no, is_derivative}
        self.connected = False

    @classmethod
    def from_env(cls) -> DNSEBroker:
        """Build from DNSE_USERNAME / DNSE_PASSWORD env vars.

        Optionally reads DNSE_SUB_ACCOUNT; if absent, the first sub-account
        returned by the API is used automatically during connect().
        """
        return cls(
            username=os.environ["DNSE_USERNAME"],
            password=os.environ["DNSE_PASSWORD"],
            sub_account=os.environ.get("DNSE_SUB_ACCOUNT"),
        )

    def _auth_headers(self, *, trading: bool = False) -> dict[str, str]:
        headers: dict[str, str] = {"Authorization": f"Bearer {self._jwt_token}"}
        if trading and self._trading_token:
            headers["Trading-Token"] = self._trading_token
        return headers

    async def connect(self) -> None:
        if self._client is not None:
            self.connected = True
            return

        try:
            import httpx  # noqa: PLC0415
        except ImportError as exc:
            raise BrokerError("httpx is required for DNSEBroker") from exc

        async with httpx.AsyncClient(timeout=30) as http:
            resp = await http.post(
                f"{_BASE_URL}/auth-service/login",
                json={"username": self._username, "password": self._password},
            )
            if resp.status_code != 200:
                raise BrokerError(f"DNSE login failed ({resp.status_code}): {resp.text}")
            self._jwt_token = resp.json().get("token")
            if not self._jwt_token:
                raise BrokerError("DNSE login returned no token")

            if self._sub_account is None:
                r = await http.get(
                    f"{_BASE_URL}/order-service/accounts",
                    headers=self._auth_headers(),
                )
                if r.status_code == 200:
                    accounts = r.json().get("accounts", [])
                    if accounts:
                        self._sub_account = str(accounts[0].get("accountNo", ""))

        self.connected = True

    async def disconnect(self) -> None:
        self._jwt_token = None
        self._trading_token = None
        self.connected = False

    async def set_trading_token(self, otp: str, *, smart_otp: bool = True) -> None:
        """Exchange an OTP for a short-lived trading token.

        Must be called after connect() and before place_order / cancel_order.
        smart_otp=True uses the Entrade X app TOTP; False uses email OTP.
        """
        if not self._jwt_token:
            raise BrokerError("Not connected — call connect() first")

        try:
            import httpx  # noqa: PLC0415
        except ImportError as exc:
            raise BrokerError("httpx is required for DNSEBroker") from exc

        otp_key = "smart-otp" if smart_otp else "otp"
        async with httpx.AsyncClient(timeout=30) as http:
            resp = await http.post(
                f"{_BASE_URL}/order-service/trading-token",
                headers={**self._auth_headers(), otp_key: otp},
            )
            if resp.status_code != 200:
                raise BrokerError(f"DNSE trading token failed: {resp.text}")
            self._trading_token = resp.json().get("tradingToken")
            if not self._trading_token:
                raise BrokerError("DNSE trading token response contained no tradingToken")

    async def get_positions(self) -> list[Position]:
        if self._client is not None:
            return self._client.get_positions()
        if not self._jwt_token or not self._sub_account:
            return []

        try:
            import httpx  # noqa: PLC0415
        except ImportError:
            return []

        async with httpx.AsyncClient(timeout=30) as http:
            resp = await http.get(
                f"{_BASE_URL}/deal-service/deals",
                params={"accountNo": self._sub_account},
                headers=self._auth_headers(),
            )
            if resp.status_code != 200:
                return []
            deals = resp.json().get("data", [])

        positions: list[Position] = []
        for deal in deals:
            symbol_raw = str(deal.get("symbol", ""))
            qty = Decimal(str(deal.get("qty", 0)))
            if qty <= 0 or not symbol_raw:
                continue
            avg_cost = Decimal(str(deal.get("tradingPrice", 0)))
            instrument = Instrument(
                symbol=f"VN:{symbol_raw}",
                asset_type=AssetType.VN_STOCK,
                exchange="HOSE",
                currency="VND",
            )
            positions.append(
                Position(
                    instrument=instrument,
                    quantity=qty,
                    market_price=avg_cost,
                    average_cost=avg_cost,
                )
            )
        return positions

    async def place_order(self, order: Order) -> Fill:
        if self._client is not None:
            return await self._client.place_order(order)
        if not self._jwt_token:
            raise BrokerError("Not connected — call connect() first")
        if not self._trading_token:
            raise BrokerError("Trading token not set — call set_trading_token(otp) first")

        try:
            import httpx  # noqa: PLC0415
        except ImportError as exc:
            raise BrokerError("httpx is required for DNSEBroker") from exc

        symbol_raw = order.instrument.symbol.split(":")[-1]
        is_deriv = _is_derivative(order.instrument.symbol)
        url = (
            f"{_BASE_URL}/order-service/derivative/orders"
            if is_deriv
            else f"{_BASE_URL}/order-service/v2/orders"
        )
        order_type = "LO" if order.limit_price is not None else "MP"
        price = float(order.limit_price) if order.limit_price is not None else 0.0

        payload: dict = {
            "accountNo": self._sub_account,
            "symbol": symbol_raw,
            "side": _SIDE[order.side],
            "quantity": int(order.quantity),
            "price": price,
            "orderType": order_type,
        }
        if order.client_order_id:
            payload["clientOrderId"] = order.client_order_id

        async with httpx.AsyncClient(timeout=30) as http:
            resp = await http.post(url, headers=self._auth_headers(trading=True), json=payload)
            if resp.status_code != 200:
                raise BrokerError(
                    f"DNSE place_order failed ({resp.status_code}): {resp.text}", order
                )
            data = resp.json()

        order_id = str(data.get("orderId") or data.get("id") or order.client_order_id or "unknown")
        self._order_meta[order_id] = {"account_no": self._sub_account, "is_derivative": is_deriv}

        fallback_price = order.limit_price or Decimal("0")
        exec_price = Decimal(str(data["price"])) if data.get("price") else fallback_price
        return Fill(
            order_id=order_id,
            instrument=order.instrument,
            side=order.side,
            quantity=order.quantity,
            price=exec_price,
        )

    async def cancel_order(self, order_id: str) -> None:
        if self._client is not None:
            return self._client.cancel_order(order_id)
        if not self._jwt_token or not self._trading_token:
            raise BrokerError(
                "Not connected or trading token missing — call connect() and set_trading_token()"
            )

        try:
            import httpx  # noqa: PLC0415
        except ImportError as exc:
            raise BrokerError("httpx is required for DNSEBroker") from exc

        meta = self._order_meta.get(order_id, {})
        account_no = meta.get("account_no") or self._sub_account
        is_deriv = meta.get("is_derivative", False)
        base_path = "derivative/orders" if is_deriv else "v2/orders"
        url = f"{_BASE_URL}/order-service/{base_path}/{order_id}?accountNo={account_no}"

        async with httpx.AsyncClient(timeout=30) as http:
            resp = await http.delete(url, headers=self._auth_headers(trading=True))
            if resp.status_code not in {200, 204}:
                raise BrokerError(f"DNSE cancel_order failed ({resp.status_code}): {resp.text}")

    async def get_account_value(self) -> Decimal:
        if self._client is not None:
            return Decimal(str(self._client.get_account_value()))
        if not self._jwt_token or not self._sub_account:
            return Decimal("0")

        try:
            import httpx  # noqa: PLC0415
        except ImportError:
            return Decimal("0")

        async with httpx.AsyncClient(timeout=30) as http:
            resp = await http.get(
                f"{_BASE_URL}/order-service/account-balances/{self._sub_account}",
                headers=self._auth_headers(),
            )
            if resp.status_code != 200:
                return Decimal("0")
            data = resp.json()

        for key in ("totalAssets", "netAssetValue", "equity", "cashBalance"):
            val = data.get(key)
            if val is not None:
                return Decimal(str(val))
        return Decimal("0")
