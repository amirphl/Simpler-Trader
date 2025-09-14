"""Bitunix exchange implementation."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable

import requests

from ..exchange import (
    Exchange,
    ExchangeConfig,
    MarginMode,
    OrderResult,
    OrderType,
    Position,
    PositionSide,
)


def _sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _build_query_concat(params: Optional[Dict[str, Any]]) -> str:
    """Sort query params by key and concatenate key+value pairs per Bitunix spec."""
    if not params:
        return ""
    items = sorted(params.items(), key=lambda kv: str(kv[0]))
    parts: List[str] = []
    for key, val in items:
        parts.append(str(key))
        parts.append(str(val))
    return "".join(parts)


def _infer_margin_coin_from_symbol(symbol: str) -> str:
    symbol_upper = (symbol or "").upper()
    for coin in ("USDT", "USD", "USDC", "BTC", "ETH"):
        if symbol_upper.endswith(coin):
            return coin
    return "USDT"


class BitunixClient:
    """Low-level Bitunix REST client (double-SHA256 signature as per spec)."""

    # Futures API hosts from vendor docs (examples use fapi.bitunix.com)
    MAINNET_BASE_URL = "https://fapi.bitunix.com"
    TESTNET_BASE_URL = "https://testnet-openapi.bitunix.com"

    def __init__(
        self,
        config: ExchangeConfig,
        logger: Optional[logging.Logger] = None,
        signer: Optional[Callable[[str, str, str, str, str], str]] = None,
    ):
        self._log = logger or logging.getLogger(__name__)
        base_url = config.base_url or (
            self.TESTNET_BASE_URL if config.testnet else self.MAINNET_BASE_URL
        )
        self._base_url = base_url.rstrip("/")
        self._api_key = config.api_key or ""
        self._secret = config.api_secret or ""
        self._language = config.locale or "en-US"
        self._timeout = config.timeout
        self._max_retries = max(config.max_retries, 1)
        self._signer = signer or self._default_signer

        self._session = requests.Session()
        # Do not inherit env proxies by default; honor explicit config instead.
        self._session.trust_env = False
        if config.proxies:
            self._session.proxies = config.proxies
            self._log.debug("Bitunix: Using proxies %s", config.proxies)

        self._log.info(
            "Bitunix: initialized base_url=%s testnet=%s",
            self._base_url,
            config.testnet,
        )

    def _default_signer(
        self, nonce: str, timestamp_ms: str, api_key: str, query_concat: str, body_string: str
    ) -> str:
        digest = _sha256_hex(f"{nonce}{timestamp_ms}{api_key}{query_concat}{body_string}")
        return _sha256_hex(f"{digest}{self._secret}")

    def _request(
        self,
        method: str,
        path: str,
        query: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        auth: bool = False,
    ) -> Dict[str, Any]:
        url = f"{self._base_url}{path}"
        query = dict(query or {})

        data: Optional[str] = None
        body_string = ""
        if body is not None:
            # Per spec: stringify JSON with no spaces.
            body_string = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
            data = body_string

        headers = {
            "Content-Type": "application/json",
            "language": self._language,
        }

        if auth:
            if not (self._api_key and self._secret):
                raise RuntimeError("Bitunix private endpoint requires API credentials")
            timestamp_ms = str(int(time.time() * 1000))
            nonce = uuid.uuid4().hex
            query_concat = _build_query_concat(query)
            signature = self._signer(nonce, timestamp_ms, self._api_key, query_concat, body_string)
            headers.update(
                {
                    "api-key": self._api_key,
                    "sign": signature,
                    "nonce": nonce,
                    "timestamp": timestamp_ms,
                }
            )

        attempts = 0
        last_exc: Exception | None = None
        while attempts < self._max_retries:
            attempts += 1
            try:
                resp = self._session.request(
                    method=method.upper(),
                    url=url,
                    params=query,
                    data=data,
                    headers=headers,
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                payload = resp.json()
                # All Bitunix responses carry "code" and "msg"; 0 == success
                code = payload.get("code")
                if code != 0:
                    msg = payload.get("msg", "Unknown error")
                    # Gracefully handle known non-fatal errors (e.g., size too small, trading disabled)
                    if code in (30016, 20012, 20003):
                        self._log.warning(
                            "Bitunix non-fatal error code %s: %s (path=%s)", code, msg, path
                        )
                        return {"code": code, "msg": msg, "data": None}
                    raise RuntimeError(f"Bitunix error code {code}: {msg}")
                return payload
            except Exception as exc:
                last_exc = exc
                if attempts >= self._max_retries:
                    break
                delay = min(2 ** (attempts - 1), 5)
                self._log.warning(
                    "Bitunix request retry %s/%s path=%s error=%s",
                    attempts,
                    self._max_retries,
                    path,
                    exc,
                )
                time.sleep(delay)

        raise RuntimeError(f"Bitunix request failed: {last_exc}") from last_exc

    # Public endpoints
    def fetch_tickers(
        self, symbols: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if symbols:
            params["symbols"] = ",".join(symbols)
        data = self._request(
            "GET", "/api/v1/futures/market/tickers", query=params, auth=False
        )
        if not data or data.get("code", 0) != 0:
            self._log.warning(
                "Bitunix: fetch_tickers failed code=%s msg=%s",
                data.get("code"),
                data.get("msg"),
            )
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for item in data.get("data", []) or []:
            out[str(item.get("symbol"))] = item
        return out

    def fetch_price(self, symbol_pair: str) -> Optional[float]:
        try:
            tick = self.fetch_tickers([symbol_pair]).get(symbol_pair)
            if not tick:
                return None
            last_price = tick.get("lastPrice") or tick.get("last") or tick.get("markPrice")
            return float(last_price) if last_price is not None else None
        except Exception as exc:
            self._log.warning("Bitunix: failed to fetch price for %s: %s", symbol_pair, exc)
            return None

    # Private endpoints
    def get_account(self, margin_coin: str) -> Dict[str, Any]:
        data = self._request(
            "GET",
            "/api/v1/futures/account",
            query={"marginCoin": margin_coin},
            auth=True,
        )
        if not data or data.get("code", 0) != 0:
            self._log.warning(
                "Bitunix: get_account failed margin=%s code=%s msg=%s",
                margin_coin,
                data.get("code"),
                data.get("msg"),
            )
            return {}
        return data

    def get_available_balance(self, margin_coin: str) -> Optional[float]:
        try:
            account = self.get_single_account(margin_coin)
            if not account:
                return None
            available = account.get("available")
            return float(available) if available is not None else None
        except Exception as exc:
            self._log.error("Bitunix: failed to fetch balance for %s: %s", margin_coin, exc)
            return None

    def change_leverage(self, margin_coin: str, symbol: str, leverage: int) -> bool:
        body = {
            "symbol": symbol,
            "leverage": int(leverage),
            "marginCoin": margin_coin,
        }
        try:
            data = self._request(
                "POST", "/api/v1/futures/account/change_leverage", body=body, auth=True
            )
            if data.get("code", 0) != 0:
                self._log.warning(
                    "Bitunix: Change leverage failed symbol=%s code=%s msg=%s",
                    symbol,
                    data.get("code"),
                    data.get("msg"),
                )
                return False
            self._log.info(
                "Bitunix: Changed leverage symbol=%s margin=%s lev=%s",
                symbol,
                margin_coin,
                leverage,
            )
            return True
        except Exception as exc:
            self._log.warning(
                "Bitunix: Change leverage error symbol=%s error=%s", symbol, exc
            )
            return False

    def change_margin_mode(self, margin_coin: str, symbol: str, margin_mode: str) -> bool:
        """POST /api/v1/futures/account/change_margin_mode"""
        body = {
            "marginMode": margin_mode,
            "symbol": symbol,
            "marginCoin": margin_coin,
        }
        try:
            data = self._request(
                "POST", "/api/v1/futures/account/change_margin_mode", body=body, auth=True
            )
            if data.get("code", 0) != 0:
                self._log.warning(
                    "Bitunix: Change margin mode failed symbol=%s code=%s msg=%s",
                    symbol,
                    data.get("code"),
                    data.get("msg"),
                )
                return False
            self._log.info(
                "Bitunix: Changed margin mode symbol=%s margin=%s mode=%s",
                symbol,
                margin_coin,
                margin_mode,
            )
            return True
        except Exception as exc:
            self._log.warning(
                "Bitunix: Change margin mode error symbol=%s error=%s", symbol, exc
            )
            return False

    def change_position_mode(self, position_mode: str) -> bool:
        """POST /api/v1/futures/account/change_position_mode"""
        body = {"positionMode": position_mode}
        try:
            data = self._request(
                "POST", "/api/v1/futures/account/change_position_mode", body=body, auth=True
            )
            if data.get("code", 0) != 0:
                self._log.warning(
                    "Bitunix: Change position mode failed code=%s msg=%s",
                    data.get("code"),
                    data.get("msg"),
                )
                return False
            self._log.info("Bitunix: Changed position mode to %s", position_mode)
            return True
        except Exception as exc:
            self._log.warning("Bitunix: Change position mode error: %s", exc)
            return False

    def get_single_account(self, margin_coin: str) -> Dict[str, Any]:
        """GET /api/v1/futures/account (single account)"""
        data = self.get_account(margin_coin)
        payload = data.get("data") or []
        if isinstance(payload, list) and payload:
            return payload[0]
        if isinstance(payload, dict):
            return payload
        return {}

    def get_pending_positions(
        self, symbol: Optional[str] = None, position_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """GET /api/v1/futures/position/get_pending_positions"""
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        if position_id:
            params["positionId"] = position_id
        data = self._request(
            "GET",
            "/api/v1/futures/position/get_pending_positions",
            query=params,
            auth=True,
        )
        if not data or data.get("code", 0) != 0:
            self._log.warning(
                "Bitunix: get_pending_positions failed code=%s msg=%s",
                data.get("code"),
                data.get("msg"),
            )
            return []
        return data.get("data") or []

    def get_history_positions(
        self,
        symbol: Optional[str] = None,
        position_id: Optional[str] = None,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        skip: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """GET /api/v1/futures/position/get_history_positions"""
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        if position_id:
            params["positionId"] = position_id
        if start_time_ms:
            params["startTime"] = int(start_time_ms)
        if end_time_ms:
            params["endTime"] = int(end_time_ms)
        if skip is not None:
            params["skip"] = int(skip)
        if limit is not None:
            params["limit"] = int(limit)
        data = self._request(
            "GET",
            "/api/v1/futures/position/get_history_positions",
            query=params,
            auth=True,
        )
        if not data or data.get("code", 0) != 0:
            self._log.warning(
                "Bitunix: get_history_positions failed code=%s msg=%s",
                data.get("code"),
                data.get("msg"),
            )
            return {"positionList": [], "total": 0}
        payload = data.get("data")
        if isinstance(payload, list):
            return {"positionList": payload, "total": len(payload)}
        if isinstance(payload, dict):
            position_list = payload.get("positionList")
            total = payload.get("total")
            if isinstance(position_list, list) and isinstance(total, int):
                return payload
            return {
                "positionList": position_list if isinstance(position_list, list) else [],
                "total": int(total) if total is not None else len(position_list or []),
            }
        return {"positionList": [], "total": 0}

    def place_tpsl_order(
        self,
        symbol: str,
        position_id: str,
        tp_price: Optional[float] = None,
        tp_stop_type: Optional[str] = None,
        sl_price: Optional[float] = None,
        sl_stop_type: Optional[str] = None,
        tp_order_type: Optional[str] = None,
        tp_order_price: Optional[float] = None,
        sl_order_type: Optional[str] = None,
        sl_order_price: Optional[float] = None,
        tp_qty: Optional[float] = None,
        sl_qty: Optional[float] = None,
    ) -> Dict[str, Any]:
        """POST /api/v1/futures/tpsl/place_order"""
        if tp_price is None and sl_price is None:
            self._log.warning("Bitunix: tp/sl order rejected: missing tp_price and sl_price")
            return {}
        if tp_qty is None and sl_qty is None:
            self._log.warning("Bitunix: tp/sl order rejected: missing tp_qty and sl_qty")
            return {}
        body: Dict[str, Any] = {
            "symbol": symbol,
            "positionId": position_id,
        }
        if tp_price is not None:
            body["tpPrice"] = str(tp_price)
        if tp_stop_type:
            body["tpStopType"] = tp_stop_type
        if sl_price is not None:
            body["slPrice"] = str(sl_price)
        if sl_stop_type:
            body["slStopType"] = sl_stop_type
        if tp_order_type:
            body["tpOrderType"] = tp_order_type
        if tp_order_price is not None:
            body["tpOrderPrice"] = str(tp_order_price)
        if sl_order_type:
            body["slOrderType"] = sl_order_type
        if sl_order_price is not None:
            body["slOrderPrice"] = str(sl_order_price)
        if tp_qty is not None:
            body["tpQty"] = str(tp_qty)
        if sl_qty is not None:
            body["slQty"] = str(sl_qty)

        data = self._request(
            "POST", "/api/v1/futures/tpsl/place_order", body=body, auth=True
        )
        if not data or data.get("code", 0) != 0:
            self._log.warning(
                "Bitunix: place_tpsl_order failed code=%s msg=%s",
                data.get("code"),
                data.get("msg"),
            )
            return {}
        return data.get("data") or {}

    def get_pending_tpsl_orders(
        self,
        symbol: Optional[str] = None,
        position_id: Optional[str] = None,
        side: Optional[int] = None,
        position_mode: Optional[int] = None,
        skip: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """GET /api/v1/futures/tpsl/get_pending_orders"""
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        if position_id:
            params["positionId"] = position_id
        if side is not None:
            params["side"] = int(side)
        if position_mode is not None:
            params["positionMode"] = int(position_mode)
        if skip is not None:
            params["skip"] = int(skip)
        if limit is not None:
            params["limit"] = int(limit)

        data = self._request(
            "GET", "/api/v1/futures/tpsl/get_pending_orders", query=params, auth=True
        )
        if not data or data.get("code", 0) != 0:
            self._log.warning(
                "Bitunix: get_pending_tpsl_orders failed code=%s msg=%s",
                data.get("code"),
                data.get("msg"),
            )
            return []
        return data.get("data") or []

    def get_history_tpsl_orders(
        self,
        symbol: Optional[str] = None,
        side: Optional[int] = None,
        position_mode: Optional[int] = None,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        skip: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """GET /api/v1/futures/tpsl/get_history_orders"""
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        if side is not None:
            params["side"] = int(side)
        if position_mode is not None:
            params["positionMode"] = int(position_mode)
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)
        if skip is not None:
            params["skip"] = int(skip)
        if limit is not None:
            params["limit"] = int(limit)

        data = self._request(
            "GET", "/api/v1/futures/tpsl/get_history_orders", query=params, auth=True
        )
        if not data or data.get("code", 0) != 0:
            self._log.warning(
                "Bitunix: get_history_tpsl_orders failed code=%s msg=%s",
                data.get("code"),
                data.get("msg"),
            )
            return {"orderList": [], "total": 0}
        payload = data.get("data")
        if isinstance(payload, list):
            return {"orderList": payload, "total": len(payload)}
        if isinstance(payload, dict):
            order_list = payload.get("orderList")
            total = payload.get("total")
            if isinstance(order_list, list) and isinstance(total, int):
                return payload
            return {
                "orderList": order_list if isinstance(order_list, list) else [],
                "total": int(total) if total is not None else len(order_list or []),
            }
        return {"orderList": [], "total": 0}

    def modify_position_tpsl_order(
        self,
        symbol: str,
        position_id: str,
        tp_price: Optional[float] = None,
        tp_stop_type: Optional[str] = None,
        sl_price: Optional[float] = None,
        sl_stop_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST /api/v1/futures/tpsl/position/modify_order"""
        if tp_price is None and sl_price is None:
            self._log.warning("Bitunix: modify position tp/sl rejected: missing tp_price and sl_price")
            return {}
        body: Dict[str, Any] = {
            "symbol": symbol,
            "positionId": position_id,
        }
        if tp_price is not None:
            body["tpPrice"] = str(tp_price)
        if tp_stop_type:
            body["tpStopType"] = tp_stop_type
        if sl_price is not None:
            body["slPrice"] = str(sl_price)
        if sl_stop_type:
            body["slStopType"] = sl_stop_type

        data = self._request(
            "POST", "/api/v1/futures/tpsl/position/modify_order", body=body, auth=True
        )
        if not data or data.get("code", 0) != 0:
            self._log.warning(
                "Bitunix: modify_position_tpsl_order failed code=%s msg=%s",
                data.get("code"),
                data.get("msg"),
            )
            return {}
        return data.get("data") or {}

    def modify_tpsl_order(
        self,
        order_id: str,
        tp_price: Optional[float] = None,
        tp_stop_type: Optional[str] = None,
        sl_price: Optional[float] = None,
        sl_stop_type: Optional[str] = None,
        tp_order_type: Optional[str] = None,
        tp_order_price: Optional[float] = None,
        sl_order_type: Optional[str] = None,
        sl_order_price: Optional[float] = None,
        tp_qty: Optional[float] = None,
        sl_qty: Optional[float] = None,
    ) -> Dict[str, Any]:
        """POST /api/v1/futures/tpsl/modify_order"""
        if tp_price is None and sl_price is None:
            self._log.warning("Bitunix: modify tp/sl rejected: missing tp_price and sl_price")
            return {}
        if tp_qty is None and sl_qty is None:
            self._log.warning("Bitunix: modify tp/sl rejected: missing tp_qty and sl_qty")
            return {}
        body: Dict[str, Any] = {"orderId": order_id}
        if tp_price is not None:
            body["tpPrice"] = str(tp_price)
        if tp_stop_type:
            body["tpStopType"] = tp_stop_type
        if sl_price is not None:
            body["slPrice"] = str(sl_price)
        if sl_stop_type:
            body["slStopType"] = sl_stop_type
        if tp_order_type:
            body["tpOrderType"] = tp_order_type
        if tp_order_price is not None:
            body["tpOrderPrice"] = str(tp_order_price)
        if sl_order_type:
            body["slOrderType"] = sl_order_type
        if sl_order_price is not None:
            body["slOrderPrice"] = str(sl_order_price)
        if tp_qty is not None:
            body["tpQty"] = str(tp_qty)
        if sl_qty is not None:
            body["slQty"] = str(sl_qty)

        data = self._request(
            "POST", "/api/v1/futures/tpsl/modify_order", body=body, auth=True
        )
        if not data or data.get("code", 0) != 0:
            self._log.warning(
                "Bitunix: modify_tpsl_order failed code=%s msg=%s",
                data.get("code"),
                data.get("msg"),
            )
            return {}
        return data.get("data") or {}

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        price: Optional[float] = None,
        trade_side: str = "OPEN",
        reduce_only: bool = False,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None,
        client_id: Optional[str] = None,
        tp_stop_type: Optional[str] = None,
        tp_order_type: Optional[str] = None,
        tp_order_price: Optional[float] = None,
        sl_stop_type: Optional[str] = None,
        sl_order_type: Optional[str] = None,
        sl_order_price: Optional[float] = None,
        effect: Optional[str] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "qty": str(qty),
            "orderType": order_type.upper(),
            "tradeSide": trade_side.upper(),
            "reduceOnly": bool(reduce_only),
        }
        if price is not None:
            body["price"] = str(price)
        if client_id:
            body["clientId"] = client_id
        if effect:
            body["effect"] = effect
        if tp_price is not None:
            body["tpPrice"] = str(tp_price)
        if tp_stop_type:
            body["tpStopType"] = tp_stop_type
        if tp_order_type:
            body["tpOrderType"] = tp_order_type
        if tp_order_price is not None:
            body["tpOrderPrice"] = str(tp_order_price)
        if sl_price is not None:
            body["slPrice"] = str(sl_price)
        if sl_stop_type:
            body["slStopType"] = sl_stop_type
        if sl_order_type:
            body["slOrderType"] = sl_order_type
        if sl_order_price is not None:
            body["slOrderPrice"] = str(sl_order_price)

        self._log.debug("Bitunix: placing order %s", json.dumps(body, separators=(",", ":")))
        data = self._request(
            "POST", "/api/v1/futures/trade/place_order", body=body, auth=True
        )
        if data.get("code", 0) != 0:
            self._log.warning(
                "Bitunix order rejected code=%s msg=%s", data.get("code"), data.get("msg")
            )
            return {}
        return data.get("data") or {}


class BitunixExchange(Exchange):
    """Bitunix futures exchange adapter."""

    def __init__(self, config: ExchangeConfig, logger: Optional[logging.Logger] = None) -> None:
        self._config = config
        self._log = logger or logging.getLogger(__name__)
        self._client = BitunixClient(config, self._log)
        self._default_margin_coin = "USDT"

    def fetch_price(self, symbol: str) -> Optional[float]:
        """Fetch last price for a symbol (public endpoint)."""
        return self._client.fetch_price(symbol)

    def get_account_balance(self) -> float:
        margin_coin = self._default_margin_coin
        balance = self._client.get_available_balance(margin_coin)
        if balance is None:
            raise RuntimeError("Failed to fetch Bitunix balance")
        return balance

    def get_24h_tickers(self) -> List[Dict[str, Any]]:
        ticks = self._client.fetch_tickers()
        return list(ticks.values())

    def get_current_positions(self) -> List[Position]:
        try:
            positions = self._client.get_pending_positions()
        except Exception as exc:
            self._log.error("Bitunix: failed to fetch pending positions: %s", exc)
            return []

        normalized: List[Position] = []
        for pos in positions or []:
            try:
                size = float(pos.get("qty", 0) or 0)
                if size == 0:
                    continue
                side_str = str(pos.get("side", "LONG")).upper()
                side = PositionSide.SHORT if side_str == "SHORT" else PositionSide.LONG
                margin_mode_str = str(pos.get("marginMode", "")).upper()
                margin_mode = (
                    MarginMode.ISOLATED if margin_mode_str == "ISOLATION" else MarginMode.CROSS
                )
                entry_price = float(pos.get("avgOpenPrice", 0) or 0)
                unrealized = float(pos.get("unrealizedPNL", 0) or 0)
                liq_raw = pos.get("liqPrice", None)
                liq_price: Optional[float] = None
                try:
                    liq_val = float(liq_raw) if liq_raw is not None else 0.0
                    if liq_val > 0:
                        liq_price = liq_val
                except (TypeError, ValueError):
                    liq_price = None

                normalized.append(
                    Position(
                        symbol=str(pos.get("symbol")),
                        side=side,
                        size=abs(size),
                        entry_price=entry_price,
                        leverage=float(pos.get("leverage", 1) or 1),
                        margin_mode=margin_mode,
                        unrealized_pnl=unrealized,
                        liquidation_price=liq_price,
                        position_id=str(pos.get("positionId") or ""),
                    )
                )
            except Exception:
                continue

        return normalized

    def get_position(self, symbol: str) -> Optional[Position]:
        for pos in self.get_current_positions():
            if pos.symbol == symbol:
                return pos
        return None

    def set_leverage(self, symbol: str, leverage: int) -> None:
        margin_coin = _infer_margin_coin_from_symbol(symbol)
        if not self._client.change_leverage(margin_coin, symbol, leverage):
            raise RuntimeError(f"Failed to set leverage for {symbol}")

    def set_margin_mode(self, symbol: str, margin_mode: MarginMode) -> None:
        margin_coin = _infer_margin_coin_from_symbol(symbol)
        mode_str = "ISOLATION" if margin_mode == MarginMode.ISOLATED else "CROSS"
        if not self._client.change_margin_mode(margin_coin, symbol, mode_str):
            raise RuntimeError(f"Failed to set margin mode {mode_str} for {symbol}")

    def open_market_position(
        self,
        symbol: str,
        side: PositionSide,
        quantity: float,
        leverage: int,
        margin_mode: MarginMode,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
    ) -> OrderResult:
        # self.set_leverage(symbol, leverage)
        response = self._client.place_order(
            symbol=symbol,
            side="BUY" if side == PositionSide.LONG else "SELL",
            qty=quantity,
            order_type=OrderType.MARKET.value,
            trade_side="OPEN",
            reduce_only=False,
            tp_price=take_profit,
            sl_price=stop_loss,
        )
        if not response or not response.get("orderId"):
            raise RuntimeError("Bitunix open position failed: no order id returned")
        order_id = str(response.get("orderId") or "")
        return OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            price=0.0,
            quantity=quantity,
            status="NEW",
            timestamp=datetime.now(timezone.utc),
        )

    def open_limit_position(
        self,
        symbol: str,
        side: PositionSide,
        quantity: float,
        price: float,
        leverage: int,
        margin_mode: MarginMode,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
    ) -> OrderResult:
        # self.set_leverage(symbol, leverage)
        response = self._client.place_order(
            symbol=symbol,
            side="BUY" if side == PositionSide.LONG else "SELL",
            qty=quantity,
            order_type=OrderType.LIMIT.value,
            price=price,
            trade_side="OPEN",
            reduce_only=False,
            tp_price=take_profit,
            sl_price=stop_loss,
        )
        if not response or not response.get("orderId"):
            raise RuntimeError("Bitunix open limit position failed: no order id returned")
        order_id = str(response.get("orderId") or "")
        return OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            price=price,
            quantity=quantity,
            status="NEW",
            timestamp=datetime.now(timezone.utc),
        )

    def close_position(
        self,
        symbol: str,
        side: Optional[PositionSide] = None,
    ) -> OrderResult:
        position = self.get_position(symbol)
        if position:
            close_qty = position.size
            side_to_send = PositionSide.SHORT if position.side == PositionSide.LONG else PositionSide.LONG
        else:
            close_qty = 0.0
            side_to_send = PositionSide.SHORT if side == PositionSide.LONG else PositionSide.LONG

        response = self._client.place_order(
            symbol=symbol,
            side="BUY" if side_to_send == PositionSide.LONG else "SELL",
            qty=close_qty or 0.001,
            order_type=OrderType.MARKET.value,
            trade_side="CLOSE",
            reduce_only=True,
        )
        if not response or not response.get("orderId"):
            raise RuntimeError("Bitunix close position failed: no order id returned")
        order_id = str(response.get("orderId") or "")
        return OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side_to_send,
            order_type=OrderType.MARKET,
            price=0.0,
            quantity=close_qty,
            status="FILLED",
            timestamp=datetime.now(timezone.utc),
        )

    def cancel_all_orders(self, symbol: str) -> None:
        self._log.warning("Bitunix: cancel_all_orders not implemented; skipping %s", symbol)

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[List]:
        # Public klines endpoint not provided; fallback to empty list to avoid breaking callers.
        self._log.warning("Bitunix: klines endpoint not integrated; returning empty list")
        return []

    def test_connection(self) -> bool:
        try:
            self._client.fetch_tickers([])
            return True
        except Exception as exc:
            self._log.error("Bitunix connection test failed: %s", exc)
            raise RuntimeError(f"Bitunix connection test failed: {exc}") from exc

    def close(self) -> None:
        try:
            self._client._session.close()
        except Exception:
            pass

    def update_stop_loss(self, position: Position, stop_price: float) -> bool:
        """Place/replace a stop-loss (used for trailing stops)."""
        if not position.position_id:
            self._log.warning(
                "Bitunix: cannot set stop loss for %s without position_id", position.symbol
            )
            return False

        try:
            result = self._client.modify_position_tpsl_order(
                symbol=position.symbol,
                position_id=position.position_id,
                sl_price=stop_price,
                sl_stop_type="MARK_PRICE",
            )
            if not result:
                # Fallback to create TP/SL if modify is unsupported.
                result = self._client.place_tpsl_order(
                    symbol=position.symbol,
                    position_id=position.position_id,
                    sl_price=stop_price,
                    sl_stop_type="MARK_PRICE",
                    sl_order_type="MARKET",
                    sl_qty=position.size,
                )
                if not result:
                    self._log.warning(
                        "Bitunix: trailing stop update returned empty response for %s",
                        position.symbol,
                    )
                    return False
            self._log.info(
                "Bitunix: updated stop loss for %s (position %s) to %.6f",
                position.symbol,
                position.position_id,
                stop_price,
            )
            return True
        except Exception as exc:
            self._log.warning("Bitunix: failed to update stop loss for %s: %s", position.symbol, exc)
            return False
