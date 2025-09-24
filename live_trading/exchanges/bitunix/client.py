"""Low-level Bitunix REST client."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

import requests

from ...exchange import ExchangeConfig
from .utils import build_query_concat, sha256_hex

class BitunixClient:
    """Low-level Bitunix REST client (double-SHA256 signature as per spec)."""

    # Futures API hosts from vendor docs (examples use fapi.bitunix.com)
    MAINNET_BASE_URL = "https://fapi.bitunix.com"
    TESTNET_BASE_URL = "https://testnet-openapi.bitunix.com"
    _NON_RETRYABLE_ERROR_CODE_MIN = 20000
    _NON_RETRYABLE_ERROR_CODE_MAX = 50000

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
        digest = sha256_hex(f"{nonce}{timestamp_ms}{api_key}{query_concat}{body_string}")
        return sha256_hex(f"{digest}{self._secret}")

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
            query_concat = build_query_concat(query)
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
                    if (
                        isinstance(code, int)
                        and self._NON_RETRYABLE_ERROR_CODE_MIN
                        <= code
                        < self._NON_RETRYABLE_ERROR_CODE_MAX
                    ):
                        raise ValueError(f"Bitunix error code {code}: {msg}")
                    raise RuntimeError(f"Bitunix error code {code}: {msg}")
                return payload
            except ValueError:
                # Deterministic exchange validation errors should fail fast.
                raise
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
            normalized_symbols = [
                str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()
            ]
            if normalized_symbols:
                params["symbols"] = ",".join(normalized_symbols)
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
        payload = data.get("data")
        items: List[Dict[str, Any]]
        if isinstance(payload, list):
            items = [item for item in payload if isinstance(item, dict)]
        elif isinstance(payload, dict):
            items = [payload]
        else:
            items = []

        out: Dict[str, Dict[str, Any]] = {}
        for item in items:
            symbol = str(item.get("symbol", "")).strip()
            if not symbol:
                continue
            out[symbol] = item
        return out

    def get_trading_pairs(
        self, symbols: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """GET /api/v1/futures/market/trading_pairs."""
        params: Dict[str, Any] = {}
        if symbols:
            normalized_symbols = [
                str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()
            ]
            if normalized_symbols:
                params["symbols"] = ",".join(normalized_symbols)

        data = self._request(
            "GET", "/api/v1/futures/market/trading_pairs", query=params, auth=False
        )
        if not data or data.get("code", 0) != 0:
            self._log.warning(
                "Bitunix: get_trading_pairs failed code=%s msg=%s",
                data.get("code"),
                data.get("msg"),
            )
            return {}

        payload = data.get("data")
        items: List[Dict[str, Any]]
        if isinstance(payload, list):
            items = [item for item in payload if isinstance(item, dict)]
        elif isinstance(payload, dict):
            items = [payload]
        else:
            items = []

        out: Dict[str, Dict[str, Any]] = {}
        for item in items:
            symbol = str(item.get("symbol", "")).strip()
            if not symbol:
                continue
            out[symbol] = item
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

    def get_depth(self, symbol: str, limit: Optional[str | int] = None) -> Dict[str, Any]:
        """GET /api/v1/futures/market/depth."""
        if not symbol:
            self._log.warning("Bitunix: get_depth rejected: symbol is required")
            return {"asks": [], "bids": []}

        params: Dict[str, Any] = {"symbol": symbol}
        if limit is not None:
            normalized_limit = str(limit).lower()
            if normalized_limit not in {"1", "5", "15", "50", "max"}:
                self._log.warning(
                    "Bitunix: get_depth rejected symbol=%s: invalid limit=%s",
                    symbol,
                    limit,
                )
                return {"asks": [], "bids": []}
            params["limit"] = normalized_limit

        data = self._request(
            "GET",
            "/api/v1/futures/market/depth",
            query=params,
            auth=False,
        )
        if not data or data.get("code", 0) != 0:
            self._log.warning(
                "Bitunix: get_depth failed symbol=%s code=%s msg=%s",
                symbol,
                data.get("code"),
                data.get("msg"),
            )
            return {"asks": [], "bids": []}
        payload = data.get("data")
        if isinstance(payload, dict):
            asks = payload.get("asks") if isinstance(payload.get("asks"), list) else []
            bids = payload.get("bids") if isinstance(payload.get("bids"), list) else []
            return {"asks": asks, "bids": bids}
        return {"asks": [], "bids": []}

    def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """GET /api/v1/futures/market/funding_rate."""
        if not symbol:
            self._log.warning("Bitunix: get_funding_rate rejected: symbol is required")
            return {}

        data = self._request(
            "GET",
            "/api/v1/futures/market/funding_rate",
            query={"symbol": symbol},
            auth=False,
        )
        if not data or data.get("code", 0) != 0:
            self._log.warning(
                "Bitunix: get_funding_rate failed symbol=%s code=%s msg=%s",
                symbol,
                data.get("code"),
                data.get("msg"),
            )
            return {}

        payload = data.get("data")
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict) and str(item.get("symbol", "")).upper() == symbol.upper():
                    return item
            if payload and isinstance(payload[0], dict):
                return payload[0]
            return {}
        if isinstance(payload, dict):
            return payload
        return {}

    def get_kline_history(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        kline_type: str = "LAST_PRICE",
    ) -> List[Dict[str, Any]]:
        """GET /api/v1/futures/market/kline."""
        if not symbol:
            self._log.warning("Bitunix: get_kline_history rejected: symbol is required")
            return []
        if not interval:
            self._log.warning("Bitunix: get_kline_history rejected symbol=%s: interval is required", symbol)
            return []

        normalized_limit = max(1, min(int(limit), 200))
        normalized_type = str(kline_type or "LAST_PRICE").upper()
        if normalized_type not in {"LAST_PRICE", "MARK_PRICE"}:
            self._log.warning(
                "Bitunix: get_kline_history rejected symbol=%s: invalid type=%s",
                symbol,
                kline_type,
            )
            return []

        params: Dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": normalized_limit,
            "type": normalized_type,
        }
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)

        data = self._request(
            "GET",
            "/api/v1/futures/market/kline",
            query=params,
            auth=False,
        )
        if not data or data.get("code", 0) != 0:
            self._log.warning(
                "Bitunix: get_kline_history failed symbol=%s interval=%s code=%s msg=%s",
                symbol,
                interval,
                data.get("code"),
                data.get("msg"),
            )
            return []

        payload = data.get("data")
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            return [payload]
        return []

    # Private endpoints
    def get_account(self, margin_coin: str) -> Dict[str, Any]:
        if not margin_coin:
            self._log.warning("Bitunix: get_account rejected: margin_coin is required")
            return {}

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
        if not symbol:
            self._log.warning("Bitunix: Change leverage rejected: symbol is required")
            return False
        if not margin_coin:
            self._log.warning("Bitunix: Change leverage rejected: margin_coin is required")
            return False
        if int(leverage) < 1:
            self._log.warning(
                "Bitunix: Change leverage rejected symbol=%s: leverage must be >= 1",
                symbol,
            )
            return False

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

    def adjust_position_margin(
        self,
        margin_coin: str,
        symbol: str,
        amount: float | str,
        side: Optional[str] = None,
        position_id: Optional[str] = None,
    ) -> bool:
        """POST /api/v1/futures/account/adjust_position_margin.

        Bitunix requires one of `side` or `positionId`.
        """
        if not side and not position_id:
            self._log.warning(
                "Bitunix: adjust_position_margin rejected symbol=%s: side or position_id required",
                symbol,
            )
            return False

        body: Dict[str, Any] = {
            "symbol": symbol,
            "marginCoin": margin_coin,
            "amount": str(amount),
        }
        if side:
            side_norm = str(side).upper()
            if side_norm not in ("LONG", "SHORT"):
                self._log.warning(
                    "Bitunix: adjust_position_margin rejected symbol=%s: invalid side=%s",
                    symbol,
                    side,
                )
                return False
            body["side"] = side_norm
        if position_id:
            body["positionId"] = position_id

        try:
            data = self._request(
                "POST",
                "/api/v1/futures/account/adjust_position_margin",
                body=body,
                auth=True,
            )
            if data.get("code", 0) != 0:
                self._log.warning(
                    "Bitunix: Adjust position margin failed symbol=%s code=%s msg=%s",
                    symbol,
                    data.get("code"),
                    data.get("msg"),
                )
                return False
            self._log.info(
                "Bitunix: Adjusted position margin symbol=%s margin=%s amount=%s side=%s position_id=%s",
                symbol,
                margin_coin,
                amount,
                body.get("side"),
                position_id,
            )
            return True
        except Exception as exc:
            self._log.warning(
                "Bitunix: Adjust position margin error symbol=%s error=%s", symbol, exc
            )
            return False

    def get_single_account(self, margin_coin: str) -> Dict[str, Any]:
        """GET /api/v1/futures/account (single account)"""
        if not margin_coin:
            self._log.warning("Bitunix: get_single_account rejected: margin_coin is required")
            return {}

        data = self.get_account(margin_coin)
        payload = data.get("data") or []
        if isinstance(payload, list) and payload:
            return payload[0]
        if isinstance(payload, dict):
            return payload
        self._log.warning(
            "Bitunix: get_single_account returned unexpected payload type margin=%s type=%s",
            margin_coin,
            type(payload).__name__,
        )
        return {}

    def get_pending_positions(
        self, symbol: Optional[str] = None, position_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """GET /api/v1/futures/position/get_pending_positions"""
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = str(symbol).strip().upper()
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
        payload = data.get("data")
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            return [payload]
        return []

    def get_position_tiers(self, symbol: str) -> List[Dict[str, Any]]:
        """GET /api/v1/futures/position/get_position_tiers."""
        normalized_symbol = str(symbol).strip().upper()
        if not normalized_symbol:
            self._log.warning("Bitunix: get_position_tiers rejected: symbol is required")
            return []

        data = self._request(
            "GET",
            "/api/v1/futures/position/get_position_tiers",
            query={"symbol": normalized_symbol},
            auth=False,
        )
        if not data or data.get("code", 0) != 0:
            self._log.warning(
                "Bitunix: get_position_tiers failed symbol=%s code=%s msg=%s",
                normalized_symbol,
                data.get("code"),
                data.get("msg"),
            )
            return []

        payload = data.get("data")
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            return [payload]
        return []

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
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)
        if skip is not None:
            params["skip"] = max(int(skip), 0)
        if limit is not None:
            params["limit"] = max(1, min(int(limit), 100))
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
        normalized_symbol = str(symbol).strip().upper()
        normalized_position_id = str(position_id).strip()
        if not normalized_symbol:
            self._log.warning("Bitunix: tp/sl order rejected: symbol is required")
            return {}
        if not normalized_position_id:
            self._log.warning("Bitunix: tp/sl order rejected: position_id is required")
            return {}
        if tp_price is None and sl_price is None:
            self._log.warning("Bitunix: tp/sl order rejected: missing tp_price and sl_price")
            return {}
        if tp_qty is None and sl_qty is None:
            self._log.warning("Bitunix: tp/sl order rejected: missing tp_qty and sl_qty")
            return {}
        body: Dict[str, Any] = {
            "symbol": normalized_symbol,
            "positionId": normalized_position_id,
        }
        if tp_price is not None:
            body["tpPrice"] = str(tp_price)
        if tp_stop_type:
            tp_stop_type_norm = str(tp_stop_type).upper()
            if tp_stop_type_norm not in {"LAST_PRICE", "MARK_PRICE"}:
                self._log.warning(
                    "Bitunix: tp/sl order rejected: invalid tp_stop_type=%s",
                    tp_stop_type,
                )
                return {}
            body["tpStopType"] = tp_stop_type_norm
        if sl_price is not None:
            body["slPrice"] = str(sl_price)
        if sl_stop_type:
            sl_stop_type_norm = str(sl_stop_type).upper()
            if sl_stop_type_norm not in {"LAST_PRICE", "MARK_PRICE"}:
                self._log.warning(
                    "Bitunix: tp/sl order rejected: invalid sl_stop_type=%s",
                    sl_stop_type,
                )
                return {}
            body["slStopType"] = sl_stop_type_norm
        if tp_order_type:
            tp_order_type_norm = str(tp_order_type).upper()
            if tp_order_type_norm not in {"LIMIT", "MARKET"}:
                self._log.warning(
                    "Bitunix: tp/sl order rejected: invalid tp_order_type=%s",
                    tp_order_type,
                )
                return {}
            body["tpOrderType"] = tp_order_type_norm
        if tp_order_price is not None:
            body["tpOrderPrice"] = str(tp_order_price)
        if sl_order_type:
            sl_order_type_norm = str(sl_order_type).upper()
            if sl_order_type_norm not in {"LIMIT", "MARKET"}:
                self._log.warning(
                    "Bitunix: tp/sl order rejected: invalid sl_order_type=%s",
                    sl_order_type,
                )
                return {}
            body["slOrderType"] = sl_order_type_norm
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
        payload = data.get("data")
        return payload if isinstance(payload, dict) else {}

    def place_position_tpsl_order(
        self,
        symbol: str,
        position_id: str,
        tp_price: Optional[float] = None,
        tp_stop_type: Optional[str] = None,
        sl_price: Optional[float] = None,
        sl_stop_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST /api/v1/futures/tpsl/position/place_order."""
        normalized_symbol = str(symbol).strip().upper()
        normalized_position_id = str(position_id).strip()
        if not normalized_symbol:
            self._log.warning("Bitunix: place position tp/sl rejected: symbol is required")
            return {}
        if not normalized_position_id:
            self._log.warning("Bitunix: place position tp/sl rejected: position_id is required")
            return {}
        if tp_price is None and sl_price is None:
            self._log.warning("Bitunix: place position tp/sl rejected: missing tp_price and sl_price")
            return {}

        body: Dict[str, Any] = {
            "symbol": normalized_symbol,
            "positionId": normalized_position_id,
        }
        if tp_price is not None:
            body["tpPrice"] = str(tp_price)
        if tp_stop_type:
            tp_stop_type_norm = str(tp_stop_type).upper()
            if tp_stop_type_norm not in {"LAST_PRICE", "MARK_PRICE"}:
                self._log.warning(
                    "Bitunix: place position tp/sl rejected: invalid tp_stop_type=%s",
                    tp_stop_type,
                )
                return {}
            body["tpStopType"] = tp_stop_type_norm
        if sl_price is not None:
            body["slPrice"] = str(sl_price)
        if sl_stop_type:
            sl_stop_type_norm = str(sl_stop_type).upper()
            if sl_stop_type_norm not in {"LAST_PRICE", "MARK_PRICE"}:
                self._log.warning(
                    "Bitunix: place position tp/sl rejected: invalid sl_stop_type=%s",
                    sl_stop_type,
                )
                return {}
            body["slStopType"] = sl_stop_type_norm

        data = self._request(
            "POST", "/api/v1/futures/tpsl/position/place_order", body=body, auth=True
        )
        if not data or data.get("code", 0) != 0:
            self._log.warning(
                "Bitunix: place_position_tpsl_order failed symbol=%s code=%s msg=%s",
                normalized_symbol,
                data.get("code"),
                data.get("msg"),
            )
            return {}
        payload = data.get("data")
        return payload if isinstance(payload, dict) else {}

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
            params["symbol"] = str(symbol).strip().upper()
        if position_id:
            params["positionId"] = position_id
        if side is not None:
            params["side"] = int(side)
        if position_mode is not None:
            params["positionMode"] = int(position_mode)
        if skip is not None:
            params["skip"] = max(int(skip), 0)
        if limit is not None:
            params["limit"] = max(1, min(int(limit), 100))

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
        payload = data.get("data")
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            return [payload]
        return []

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
            params["symbol"] = str(symbol).strip().upper()
        if side is not None:
            params["side"] = int(side)
        if position_mode is not None:
            params["positionMode"] = int(position_mode)
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)
        if skip is not None:
            params["skip"] = max(int(skip), 0)
        if limit is not None:
            params["limit"] = max(1, min(int(limit), 100))

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
        normalized_symbol = str(symbol).strip().upper()
        normalized_position_id = str(position_id).strip()
        if not normalized_symbol:
            self._log.warning("Bitunix: modify position tp/sl rejected: symbol is required")
            return {}
        if not normalized_position_id:
            self._log.warning("Bitunix: modify position tp/sl rejected: position_id is required")
            return {}
        if tp_price is None and sl_price is None:
            self._log.warning("Bitunix: modify position tp/sl rejected: missing tp_price and sl_price")
            return {}
        body: Dict[str, Any] = {
            "symbol": normalized_symbol,
            "positionId": normalized_position_id,
        }
        if tp_price is not None:
            body["tpPrice"] = str(tp_price)
        if tp_stop_type:
            tp_stop_type_norm = str(tp_stop_type).upper()
            if tp_stop_type_norm not in {"LAST_PRICE", "MARK_PRICE"}:
                self._log.warning(
                    "Bitunix: modify position tp/sl rejected: invalid tp_stop_type=%s",
                    tp_stop_type,
                )
                return {}
            body["tpStopType"] = tp_stop_type_norm
        if sl_price is not None:
            body["slPrice"] = str(sl_price)
        if sl_stop_type:
            sl_stop_type_norm = str(sl_stop_type).upper()
            if sl_stop_type_norm not in {"LAST_PRICE", "MARK_PRICE"}:
                self._log.warning(
                    "Bitunix: modify position tp/sl rejected: invalid sl_stop_type=%s",
                    sl_stop_type,
                )
                return {}
            body["slStopType"] = sl_stop_type_norm

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
        normalized_order_id = str(order_id).strip()
        if not normalized_order_id:
            self._log.warning("Bitunix: modify tp/sl rejected: order_id is required")
            return {}
        if tp_price is None and sl_price is None:
            self._log.warning("Bitunix: modify tp/sl rejected: missing tp_price and sl_price")
            return {}
        if tp_qty is None and sl_qty is None:
            self._log.warning("Bitunix: modify tp/sl rejected: missing tp_qty and sl_qty")
            return {}
        body: Dict[str, Any] = {"orderId": normalized_order_id}
        if tp_price is not None:
            body["tpPrice"] = str(tp_price)
        if tp_stop_type:
            tp_stop_type_norm = str(tp_stop_type).upper()
            if tp_stop_type_norm not in {"LAST_PRICE", "MARK_PRICE"}:
                self._log.warning(
                    "Bitunix: modify tp/sl rejected: invalid tp_stop_type=%s",
                    tp_stop_type,
                )
                return {}
            body["tpStopType"] = tp_stop_type_norm
        if sl_price is not None:
            body["slPrice"] = str(sl_price)
        if sl_stop_type:
            sl_stop_type_norm = str(sl_stop_type).upper()
            if sl_stop_type_norm not in {"LAST_PRICE", "MARK_PRICE"}:
                self._log.warning(
                    "Bitunix: modify tp/sl rejected: invalid sl_stop_type=%s",
                    sl_stop_type,
                )
                return {}
            body["slStopType"] = sl_stop_type_norm
        if tp_order_type:
            tp_order_type_norm = str(tp_order_type).upper()
            if tp_order_type_norm not in {"LIMIT", "MARKET"}:
                self._log.warning(
                    "Bitunix: modify tp/sl rejected: invalid tp_order_type=%s",
                    tp_order_type,
                )
                return {}
            body["tpOrderType"] = tp_order_type_norm
        if tp_order_price is not None:
            body["tpOrderPrice"] = str(tp_order_price)
        if sl_order_type:
            sl_order_type_norm = str(sl_order_type).upper()
            if sl_order_type_norm not in {"LIMIT", "MARKET"}:
                self._log.warning(
                    "Bitunix: modify tp/sl rejected: invalid sl_order_type=%s",
                    sl_order_type,
                )
                return {}
            body["slOrderType"] = sl_order_type_norm
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
        payload = data.get("data")
        return payload if isinstance(payload, dict) else {}

    def cancel_tpsl_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """POST /api/v1/futures/tpsl/cancel_order."""
        normalized_symbol = str(symbol).strip().upper()
        normalized_order_id = str(order_id).strip()
        if not normalized_symbol:
            self._log.warning("Bitunix: cancel_tpsl_order rejected: symbol is required")
            return {}
        if not normalized_order_id:
            self._log.warning("Bitunix: cancel_tpsl_order rejected: order_id is required")
            return {}

        body = {"symbol": normalized_symbol, "orderId": normalized_order_id}
        data = self._request(
            "POST", "/api/v1/futures/tpsl/cancel_order", body=body, auth=True
        )
        if not data or data.get("code", 0) != 0:
            self._log.warning(
                "Bitunix: cancel_tpsl_order failed symbol=%s order_id=%s code=%s msg=%s",
                normalized_symbol,
                normalized_order_id,
                data.get("code"),
                data.get("msg"),
            )
            return {}
        payload = data.get("data")
        return payload if isinstance(payload, dict) else {}

    def cancel_all_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """POST /api/v1/futures/trade/cancel_all_orders."""
        body: Dict[str, Any] = {}
        if symbol is not None and str(symbol).strip():
            body["symbol"] = str(symbol).strip().upper()

        data = self._request(
            "POST", "/api/v1/futures/trade/cancel_all_orders", body=body, auth=True
        )
        if not data or data.get("code", 0) != 0:
            self._log.warning(
                "Bitunix: cancel_all_orders failed symbol=%s code=%s msg=%s",
                body.get("symbol"),
                data.get("code"),
                data.get("msg"),
            )
            return {}
        payload = data.get("data")
        return payload if isinstance(payload, dict) else {}

    def cancel_orders(
        self, symbol: str, order_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """POST /api/v1/futures/trade/cancel_orders."""
        normalized_symbol = str(symbol).strip().upper()
        if not normalized_symbol:
            self._log.warning("Bitunix: cancel_orders rejected: symbol is required")
            return {}
        if not order_list:
            self._log.warning("Bitunix: cancel_orders rejected: order_list is required")
            return {}

        normalized_orders: List[Dict[str, str]] = []
        for raw in order_list:
            if not isinstance(raw, dict):
                continue
            order_id = str(raw.get("orderId", "")).strip()
            client_id = str(raw.get("clientId", "")).strip()
            if order_id:
                normalized_orders.append({"orderId": order_id})
            elif client_id:
                normalized_orders.append({"clientId": client_id})

        if not normalized_orders:
            self._log.warning(
                "Bitunix: cancel_orders rejected symbol=%s: order_list missing orderId/clientId",
                normalized_symbol,
            )
            return {}

        body: Dict[str, Any] = {
            "symbol": normalized_symbol,
            "orderList": normalized_orders,
        }
        data = self._request(
            "POST", "/api/v1/futures/trade/cancel_orders", body=body, auth=True
        )
        if not data or data.get("code", 0) != 0:
            self._log.warning(
                "Bitunix: cancel_orders failed symbol=%s code=%s msg=%s",
                normalized_symbol,
                data.get("code"),
                data.get("msg"),
            )
            return {}
        payload = data.get("data")
        return payload if isinstance(payload, dict) else {}

    def close_all_position(self, symbol: Optional[str] = None) -> bool:
        """POST /api/v1/futures/trade/close_all_position."""
        body: Dict[str, Any] = {}
        if symbol is not None and str(symbol).strip():
            body["symbol"] = str(symbol).strip().upper()

        data = self._request(
            "POST", "/api/v1/futures/trade/close_all_position", body=body, auth=True
        )
        if not data or data.get("code", 0) != 0:
            self._log.warning(
                "Bitunix: close_all_position failed symbol=%s code=%s msg=%s",
                body.get("symbol"),
                data.get("code"),
                data.get("msg"),
            )
            return False
        return True

    def flash_close_position(self, position_id: str) -> Dict[str, Any]:
        """POST /api/v1/futures/trade/flash_close_position."""
        normalized_position_id = str(position_id).strip()
        if not normalized_position_id:
            self._log.warning("Bitunix: flash_close_position rejected: position_id is required")
            return {}

        body: Dict[str, Any] = {"positionId": normalized_position_id}
        data = self._request(
            "POST", "/api/v1/futures/trade/flash_close_position", body=body, auth=True
        )
        if not data or data.get("code", 0) != 0:
            self._log.warning(
                "Bitunix: flash_close_position failed position_id=%s code=%s msg=%s",
                normalized_position_id,
                data.get("code"),
                data.get("msg"),
            )
            return {}
        payload = data.get("data")
        return payload if isinstance(payload, dict) else {}

    def get_history_orders(
        self,
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
        client_id: Optional[str] = None,
        status: Optional[str] = None,
        order_type: Optional[str] = None,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        skip: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """GET /api/v1/futures/trade/get_history_orders."""
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = str(symbol).strip().upper()
        if order_id:
            params["orderId"] = str(order_id).strip()
        if client_id:
            params["clientId"] = str(client_id).strip()
        if status:
            params["status"] = str(status).strip().upper()
        if order_type:
            params["type"] = str(order_type).strip().upper()
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)
        if skip is not None:
            params["skip"] = max(int(skip), 0)
        if limit is not None:
            params["limit"] = max(1, min(int(limit), 100))

        data = self._request(
            "GET",
            "/api/v1/futures/trade/get_history_orders",
            query=params,
            auth=True,
        )
        if not data or data.get("code", 0) != 0:
            self._log.warning(
                "Bitunix: get_history_orders failed code=%s msg=%s",
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

    def get_history_trades(
        self,
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
        position_id: Optional[str] = None,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        skip: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """GET /api/v1/futures/trade/get_history_trades."""
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = str(symbol).strip().upper()
        if order_id:
            params["orderId"] = str(order_id).strip()
        if position_id:
            params["positionId"] = str(position_id).strip()
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)
        if skip is not None:
            params["skip"] = max(int(skip), 0)
        if limit is not None:
            params["limit"] = max(1, min(int(limit), 100))

        data = self._request(
            "GET",
            "/api/v1/futures/trade/get_history_trades",
            query=params,
            auth=True,
        )
        if not data or data.get("code", 0) != 0:
            self._log.warning(
                "Bitunix: get_history_trades failed code=%s msg=%s",
                data.get("code"),
                data.get("msg"),
            )
            return {"tradeList": [], "total": 0}

        payload = data.get("data")
        if isinstance(payload, list):
            return {"tradeList": payload, "total": len(payload)}
        if isinstance(payload, dict):
            trade_list = payload.get("tradeList")
            total = payload.get("total")
            if isinstance(trade_list, list) and isinstance(total, int):
                return payload
            return {
                "tradeList": trade_list if isinstance(trade_list, list) else [],
                "total": int(total) if total is not None else len(trade_list or []),
            }
        return {"tradeList": [], "total": 0}

    def get_pending_orders(
        self,
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
        client_id: Optional[str] = None,
        status: Optional[str] = None,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        skip: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """GET /api/v1/futures/trade/get_pending_orders."""
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = str(symbol).strip().upper()
        if order_id:
            params["orderId"] = str(order_id).strip()
        if client_id:
            params["clientId"] = str(client_id).strip()
        if status:
            params["status"] = str(status).strip().upper()
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)
        if skip is not None:
            params["skip"] = max(int(skip), 0)
        if limit is not None:
            params["limit"] = max(1, min(int(limit), 100))

        data = self._request(
            "GET",
            "/api/v1/futures/trade/get_pending_orders",
            query=params,
            auth=True,
        )
        if not data or data.get("code", 0) != 0:
            self._log.warning(
                "Bitunix: get_pending_orders failed code=%s msg=%s",
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

    def get_order_detail(
        self,
        order_id: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """GET /api/v1/futures/trade/get_order_detail."""
        normalized_order_id = str(order_id).strip() if order_id is not None else ""
        normalized_client_id = str(client_id).strip() if client_id is not None else ""
        if not normalized_order_id and not normalized_client_id:
            self._log.warning(
                "Bitunix: get_order_detail rejected: order_id or client_id is required"
            )
            return {}

        params: Dict[str, Any] = {}
        if normalized_order_id:
            params["orderId"] = normalized_order_id
        if normalized_client_id:
            params["clientId"] = normalized_client_id

        data = self._request(
            "GET",
            "/api/v1/futures/trade/get_order_detail",
            query=params,
            auth=True,
        )
        if not data or data.get("code", 0) != 0:
            self._log.warning(
                "Bitunix: get_order_detail failed order_id=%s client_id=%s code=%s msg=%s",
                normalized_order_id,
                normalized_client_id,
                data.get("code"),
                data.get("msg"),
            )
            return {}

        payload = data.get("data")
        return payload if isinstance(payload, dict) else {}

    def modify_order(
        self,
        qty: float,
        price: float,
        order_id: Optional[str] = None,
        client_id: Optional[str] = None,
        tp_price: Optional[float] = None,
        tp_stop_type: Optional[str] = None,
        tp_order_type: Optional[str] = None,
        tp_order_price: Optional[float] = None,
        sl_price: Optional[float] = None,
        sl_stop_type: Optional[str] = None,
        sl_order_type: Optional[str] = None,
        sl_order_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """POST /api/v1/futures/trade/modify_order."""
        normalized_order_id = str(order_id).strip() if order_id is not None else ""
        normalized_client_id = str(client_id).strip() if client_id is not None else ""
        if not normalized_order_id and not normalized_client_id:
            self._log.warning(
                "Bitunix: modify_order rejected: order_id or client_id is required"
            )
            return {}

        body: Dict[str, Any] = {
            "qty": str(qty),
            "price": str(price),
        }
        if normalized_order_id:
            body["orderId"] = normalized_order_id
        if normalized_client_id:
            body["clientId"] = normalized_client_id

        if tp_price is not None:
            body["tpPrice"] = str(tp_price)
        if tp_stop_type:
            tp_stop_type_norm = str(tp_stop_type).upper()
            if tp_stop_type_norm not in {"LAST_PRICE", "MARK_PRICE"}:
                self._log.warning(
                    "Bitunix: modify_order rejected: invalid tp_stop_type=%s",
                    tp_stop_type,
                )
                return {}
            body["tpStopType"] = tp_stop_type_norm
        if tp_order_type:
            tp_order_type_norm = str(tp_order_type).upper()
            if tp_order_type_norm not in {"LIMIT", "MARKET"}:
                self._log.warning(
                    "Bitunix: modify_order rejected: invalid tp_order_type=%s",
                    tp_order_type,
                )
                return {}
            body["tpOrderType"] = tp_order_type_norm
        if tp_order_price is not None:
            body["tpOrderPrice"] = str(tp_order_price)

        if sl_price is not None:
            body["slPrice"] = str(sl_price)
        if sl_stop_type:
            sl_stop_type_norm = str(sl_stop_type).upper()
            if sl_stop_type_norm not in {"LAST_PRICE", "MARK_PRICE"}:
                self._log.warning(
                    "Bitunix: modify_order rejected: invalid sl_stop_type=%s",
                    sl_stop_type,
                )
                return {}
            body["slStopType"] = sl_stop_type_norm
        if sl_order_type:
            sl_order_type_norm = str(sl_order_type).upper()
            if sl_order_type_norm not in {"LIMIT", "MARKET"}:
                self._log.warning(
                    "Bitunix: modify_order rejected: invalid sl_order_type=%s",
                    sl_order_type,
                )
                return {}
            body["slOrderType"] = sl_order_type_norm
        if sl_order_price is not None:
            body["slOrderPrice"] = str(sl_order_price)

        data = self._request(
            "POST", "/api/v1/futures/trade/modify_order", body=body, auth=True
        )
        if not data or data.get("code", 0) != 0:
            self._log.warning(
                "Bitunix: modify_order failed order_id=%s client_id=%s code=%s msg=%s",
                normalized_order_id,
                normalized_client_id,
                data.get("code"),
                data.get("msg"),
            )
            return {}
        payload = data.get("data")
        return payload if isinstance(payload, dict) else {}

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        price: Optional[float] = None,
        trade_side: str = "OPEN",
        position_id: Optional[str] = None,
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
        normalized_symbol = str(symbol).strip().upper()
        normalized_side = str(side).strip().upper()
        normalized_order_type = str(order_type).strip().upper()
        normalized_trade_side = str(trade_side).strip().upper()
        normalized_position_id = str(position_id).strip() if position_id is not None else ""
        normalized_client_id = str(client_id).strip() if client_id is not None else ""

        if not normalized_symbol:
            self._log.warning("Bitunix: place_order rejected: symbol is required")
            return {}
        if normalized_side not in {"BUY", "SELL"}:
            self._log.warning("Bitunix: place_order rejected: invalid side=%s", side)
            return {}
        if normalized_order_type not in {"LIMIT", "MARKET"}:
            self._log.warning(
                "Bitunix: place_order rejected: invalid order_type=%s", order_type
            )
            return {}
        if normalized_trade_side not in {"OPEN", "CLOSE"}:
            self._log.warning(
                "Bitunix: place_order rejected: invalid trade_side=%s", trade_side
            )
            return {}
        if normalized_order_type == "LIMIT" and price is None:
            self._log.warning("Bitunix: place_order rejected: price is required for LIMIT")
            return {}
        if normalized_order_type == "LIMIT" and effect is None:
            self._log.warning("Bitunix: place_order rejected: effect is required for LIMIT")
            return {}
        if normalized_trade_side == "CLOSE" and not normalized_position_id:
            self._log.warning("Bitunix: place_order rejected: position_id is required for CLOSE")
            return {}
        if effect is not None:
            effect_norm = str(effect).strip().upper()
            if effect_norm not in {"IOC", "FOK", "GTC", "POST_ONLY"}:
                self._log.warning("Bitunix: place_order rejected: invalid effect=%s", effect)
                return {}
        else:
            effect_norm = ""

        tp_stop_type_norm = ""
        if tp_stop_type:
            tp_stop_type_norm = str(tp_stop_type).upper()
            if tp_stop_type_norm not in {"LAST_PRICE", "MARK_PRICE"}:
                self._log.warning(
                    "Bitunix: place_order rejected: invalid tp_stop_type=%s", tp_stop_type
                )
                return {}

        sl_stop_type_norm = ""
        if sl_stop_type:
            sl_stop_type_norm = str(sl_stop_type).upper()
            if sl_stop_type_norm not in {"LAST_PRICE", "MARK_PRICE"}:
                self._log.warning(
                    "Bitunix: place_order rejected: invalid sl_stop_type=%s", sl_stop_type
                )
                return {}

        tp_order_type_norm = ""
        if tp_order_type:
            tp_order_type_norm = str(tp_order_type).upper()
            if tp_order_type_norm not in {"LIMIT", "MARKET"}:
                self._log.warning(
                    "Bitunix: place_order rejected: invalid tp_order_type=%s", tp_order_type
                )
                return {}
            if tp_order_type_norm == "LIMIT" and tp_order_price is None:
                self._log.warning(
                    "Bitunix: place_order rejected: tp_order_price required when tp_order_type=LIMIT"
                )
                return {}

        sl_order_type_norm = ""
        if sl_order_type:
            sl_order_type_norm = str(sl_order_type).upper()
            if sl_order_type_norm not in {"LIMIT", "MARKET"}:
                self._log.warning(
                    "Bitunix: place_order rejected: invalid sl_order_type=%s", sl_order_type
                )
                return {}
            if sl_order_type_norm == "LIMIT" and sl_order_price is None:
                self._log.warning(
                    "Bitunix: place_order rejected: sl_order_price required when sl_order_type=LIMIT"
                )
                return {}

        body: Dict[str, Any] = {
            "symbol": normalized_symbol,
            "side": normalized_side,
            "qty": str(qty),
            "orderType": normalized_order_type,
            "tradeSide": normalized_trade_side,
            "reduceOnly": bool(reduce_only),
        }
        if normalized_position_id:
            body["positionId"] = normalized_position_id
        if price is not None:
            body["price"] = str(price)
        if normalized_client_id:
            body["clientId"] = normalized_client_id
        if effect_norm:
            body["effect"] = effect_norm
        if tp_price is not None:
            body["tpPrice"] = str(tp_price)
        if tp_stop_type_norm:
            body["tpStopType"] = tp_stop_type_norm
        if tp_order_type_norm:
            body["tpOrderType"] = tp_order_type_norm
        if tp_order_price is not None:
            body["tpOrderPrice"] = str(tp_order_price)
        if sl_price is not None:
            body["slPrice"] = str(sl_price)
        if sl_stop_type_norm:
            body["slStopType"] = sl_stop_type_norm
        if sl_order_type_norm:
            body["slOrderType"] = sl_order_type_norm
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
        payload = data.get("data")
        return payload if isinstance(payload, dict) else {}
