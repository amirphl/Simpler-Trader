"""Weex Exchange implementation for futures and spot trading."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import random
import socket
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener

from ..exchange import (
    Exchange,
    ExchangeConfig,
    MarginMode,
    OrderResult,
    OrderType,
    Position,
    PositionSide,
)


class WeexTradingMode(Enum):
    """Trading mode for Weex."""

    SPOT = "spot"
    FUTURES = "futures"


class WeexExchange(Exchange):
    """Weex exchange client supporting both spot and futures trading.

    API Documentation: https://doc.wbfex.biz (assumed endpoint)

    Note: This implementation uses standard REST API patterns.
    You may need to adjust endpoints and parameters based on actual Weex API documentation.
    """

    # Base URLs (adjust based on actual Weex documentation)
    MAINNET_BASE_URL = "https://api-contract.weex.com"
    TESTNET_BASE_URL = "https://testnet-api.weex.com"
    _RETRY_BASE_DELAY = 1.0
    _RETRY_MAX_DELAY = 30.0

    def __init__(
        self,
        config: ExchangeConfig,
        trading_mode: WeexTradingMode = WeexTradingMode.FUTURES,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize Weex exchange client.

        Args:
            config: Exchange configuration
            trading_mode: SPOT or FUTURES trading mode
            logger: Optional logger instance
        """
        self._config = config
        self._trading_mode = trading_mode
        self._log = logger or logging.getLogger(__name__)
        self._time_offset_ms: Optional[int] = None

        # Set base URL based on testnet flag
        self._base_url = (
            self.TESTNET_BASE_URL if config.testnet else self.MAINNET_BASE_URL
        ).rstrip("/")

        # Setup HTTP opener with proxy if provided. If no proxies were
        # supplied in config, fall back to common environment variables
        # (HTTP_PROXY, HTTPS_PROXY, ALL_PROXY) to allow users to provide
        # proxy settings via env.
        handlers = []
        proxies = config.proxies or {}
        # environment fallback
        if not proxies:
            # read common proxy environment variables
            import os

            http_env = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
            https_env = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
            all_env = os.getenv("ALL_PROXY") or os.getenv("all_proxy")

            if all_env:
                proxies = {"http": all_env, "https": all_env}
            else:
                if http_env:
                    proxies["http"] = http_env
                if https_env:
                    proxies["https"] = https_env

        if proxies:
            handlers.append(ProxyHandler(proxies))
            self._log.debug(f"Using proxies for Weex requests: {proxies}")

        self._opener = build_opener(*handlers)

        self._log.info(
            f"Initialized Weex exchange client: mode={trading_mode.value}, "
            f"testnet={config.testnet}"
        )

    def _sign_request(self, params: Dict) -> str:
        """Generate signature for authenticated requests.

        Args:
            params: Request parameters

        Returns:
            HMAC-SHA256 signature
        """
        # Sort parameters alphabetically
        sorted_params = sorted(params.items())
        query_string = "&".join([f"{k}={v}" for k, v in sorted_params])

        # Create signature
        signature = hmac.new(
            self._config.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return signature

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        signed: bool = False,
    ) -> any:
        """Make HTTP request to Weex API with documented signing."""
        base_params = dict(params or {})
        max_attempts = max(1, self._config.max_retries)
        last_exc: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            params = dict(base_params)
            should_retry = False

            try:
                method_upper = method.upper()
                query_string = (
                    urlencode(params) if method_upper == "GET" and params else ""
                )
                body_str = "" if method_upper == "GET" else json.dumps(params or {})

                # timestamp_ms = self._signed_timestamp_ms()
                # sign_target = f"{timestamp_ms}{method_upper}{endpoint}"
                # if query_string:
                #     sign_target = f"{sign_target}?{query_string}"
                # sign_target = f"{sign_target}{body_str}"

                # signature = (
                #     base64.b64encode(
                #         hmac.new(
                #             self._config.api_secret.encode("utf-8"),
                #             sign_target.encode("utf-8"),
                #             hashlib.sha256,
                #         ).digest()
                #     ).decode("utf-8")
                #     if signed
                #     else ""
                # )

                timestamp_ms = str(int(time.time() * 1000))
                signature = _generate_signature(
                    self._config.api_secret,
                    timestamp_ms,
                    method_upper,
                    endpoint,
                    query_string,
                    body_str,
                )

                url = f"{self._base_url}{endpoint}"
                if method_upper == "GET" and query_string:
                    url = f"{url}?{query_string}"

                request = Request(
                    url,
                    data=None if method_upper == "GET" else body_str.encode("utf-8"),
                    method=method_upper,
                )

                headers = {
                    "Content-Type": "application/json",
                    "locale": self._config.locale,
                }
                if signed:
                    headers.update(
                        {
                            "ACCESS-KEY": self._config.api_key,
                            "ACCESS-SIGN": signature,
                            "ACCESS-TIMESTAMP": timestamp_ms,
                        }
                    )
                    if self._config.passphrase:
                        headers["ACCESS-PASSPHRASE"] = self._config.passphrase

                for k, v in headers.items():
                    request.add_header(k, v)

                started = time.perf_counter()
                with self._opener.open(
                    request, timeout=self._config.timeout
                ) as response:
                    body = response.read()
                    duration = time.perf_counter() - started
                    if self._log.isEnabledFor(logging.DEBUG):
                        self._log.debug(
                            f"Weex {method_upper} {endpoint} completed in {duration:.3f}s "
                            f"status={getattr(response, 'status', 'unknown')} bytes={len(body)}"
                        )
                    result = json.loads(body)

                    if isinstance(result, dict) and "code" in result:
                        code = str(result.get("code"))
                        if code in (None, "", "200", "0"):
                            return result
                        error_msg = result.get("msg", "Unknown error")
                        message = f"Weex API error ({code}): {error_msg}"
                        if code in ("429", "408", "500", "503"):
                            should_retry = True
                            last_exc = RuntimeError(message)
                        else:
                            raise RuntimeError(message)
                    else:
                        return result

            except HTTPError as exc:
                last_exc = exc
                error_body = exc.read().decode("utf-8") if exc.fp else ""
                should_retry = self._should_retry_status(exc.code)
                message = f"Weex HTTP {exc.code}: {error_body or exc.reason}"
            except (URLError, TimeoutError, socket.timeout) as exc:
                last_exc = exc
                should_retry = True
                message = f"Weex connection error: {getattr(exc, 'reason', exc)}"
            except json.JSONDecodeError as exc:
                last_exc = exc
                should_retry = True
                message = "Weex returned invalid JSON"
            except Exception as exc:
                last_exc = exc
                message = f"Unexpected Weex error: {exc}"

            if not should_retry or attempt == max_attempts:
                self._log.error(
                    "Request failed (%s/%s): %s", attempt, max_attempts, message
                )
                raise RuntimeError(message) from last_exc

            backoff = min(
                self._RETRY_MAX_DELAY, self._RETRY_BASE_DELAY * (2 ** (attempt - 1))
            )
            jitter = random.uniform(0, backoff * 0.3)
            sleep_for = backoff + jitter
            self._log.warning(
                "Request failed (%s/%s): %s. Retrying in %.2fs...",
                attempt,
                max_attempts,
                message,
                sleep_for,
            )
            time.sleep(sleep_for)

        raise RuntimeError("Weex request failed after retries") from last_exc

    def _should_retry_status(self, status: Optional[int]) -> bool:
        if status is None:
            return True
        if status >= 500:
            return True
        return status in (408, 429)

    def _margin_mode_code(self, margin_mode: MarginMode) -> int:
        """Map internal margin mode to Weex integer code."""
        return 3 if margin_mode == MarginMode.ISOLATED else 1

    def _signed_timestamp_ms(self) -> int:
        """Return a timestamp adjusted with server offset."""
        if self._time_offset_ms is None:
            self._time_offset_ms = self._fetch_time_offset_ms()
        return int(time.time() * 1000 + (self._time_offset_ms or 0))

    def _fetch_time_offset_ms(self) -> int:
        """Fetch server time and compute local offset (server - local)."""
        try:
            url = f"{self._base_url}/capi/v2/market/time"
            request = Request(url, method="GET")
            with self._opener.open(request, timeout=self._config.timeout) as response:
                payload = response.read()
                data = json.loads(payload)
                server_ts = int(data.get("timestamp", 0))
                local_ts = int(time.time() * 1000)
                return server_ts - local_ts
        except Exception as exc:
            self._log.warning(f"Failed to sync server time: {exc}")
            return 0

    def _set_leverage(
        self, symbol: str, leverage: int, margin_mode: MarginMode
    ) -> None:
        """Internal leverage setter honoring desired margin mode."""
        endpoint = "/capi/v2/account/leverage"
        params = {
            "symbol": symbol,
            "marginMode": self._margin_mode_code(margin_mode),
            "longLeverage": str(leverage),
            "shortLeverage": str(leverage),
        }
        self._request("POST", endpoint, params, signed=True)
        self._log.info(
            "Set leverage for %s: %sx (mode=%s)",
            symbol,
            leverage,
            margin_mode.value,
        )

    def get_account_balance(self) -> float:
        """Get available account balance in USDT."""
        assets = self.get_account_assets()
        return assets.get("USDT", {}).get("available", 0.0)

    def get_account_assets(self) -> Dict[str, Dict[str, float]]:
        """Fetch account assets and return a symbol->asset info map."""
        try:
            endpoint = "/capi/v2/account/assets"
            result = self._request("GET", endpoint, signed=True)
            assets_list = result if isinstance(result, list) else result.get("data", [])

            assets: Dict[str, Dict[str, float]] = {}
            for asset in assets_list or []:
                try:
                    coin = asset.get("coinName")
                    if not coin:
                        continue
                    assets[coin.upper()] = {
                        "available": float(asset.get("available", 0) or 0),
                        "frozen": float(asset.get("frozen", 0) or 0),
                        "equity": float(asset.get("equity", 0) or 0),
                        "unrealize_pnl": float(asset.get("unrealizePnl", 0) or 0),
                    }
                except (ValueError, TypeError):
                    continue

            return assets
        except Exception as e:
            self._log.error(f"Failed to get account assets: {e}")
            raise RuntimeError(f"Failed to get account assets: {e}") from e

    def get_24h_tickers(self) -> List[Dict]:
        """Get 24-hour ticker data for all symbols."""
        # Not documented in provided futures API; return empty list to avoid failures.
        self._log.warning(
            "24h tickers endpoint not documented for Weex futures; returning empty list"
        )
        return []

    def get_current_positions(self) -> List[Position]:
        """Get all currently open positions."""
        try:
            if self._trading_mode == WeexTradingMode.SPOT:
                return []

            endpoint = "/capi/v2/account/position/allPosition"
            result = self._request("GET", endpoint, signed=True)
            position_data = (
                result if isinstance(result, list) else result.get("data", [])
            )

            positions: List[Position] = []
            for pos in position_data or []:
                try:
                    size = float(pos.get("size", 0))
                except (TypeError, ValueError):
                    continue
                if size == 0:
                    continue

                side_str = str(pos.get("side", "LONG")).upper()
                side = PositionSide.SHORT if side_str == "SHORT" else PositionSide.LONG

                margin_mode_str = str(pos.get("margin_mode", "")).upper()
                margin_mode = (
                    MarginMode.ISOLATED
                    if margin_mode_str == "ISOLATED"
                    else MarginMode.CROSS
                )

                entry_price = 0.0
                try:
                    open_value = float(pos.get("open_value", 0) or 0)
                    if size:
                        entry_price = abs(open_value / size)
                except (TypeError, ValueError, ZeroDivisionError):
                    entry_price = 0.0

                positions.append(
                    Position(
                        symbol=pos.get("symbol"),
                        side=side,
                        size=abs(size),
                        entry_price=entry_price,
                        leverage=float(pos.get("leverage", 1) or 1),
                        margin_mode=margin_mode,
                        unrealized_pnl=float(pos.get("unrealizePnl", 0) or 0),
                        liquidation_price=float(pos.get("liquidatePrice", 0) or 0)
                        if pos.get("liquidatePrice") not in (None, "0", 0)
                        else None,
                        position_id=str(pos.get("id") or ""),
                    )
                )

            return positions

        except Exception as e:
            self._log.error(f"Failed to get positions: {e}")
            raise RuntimeError(f"Failed to get positions: {e}") from e

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        positions = self.get_current_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    def get_current_orders(
        self, symbol: Optional[str] = None, limit: int = 100, page: int = 0
    ) -> List[Dict]:
        """Fetch current (open/pending) orders."""
        try:
            if self._trading_mode == WeexTradingMode.SPOT:
                return []

            endpoint = "/capi/v2/order/current"
            params: Dict[str, str | int] = {
                "limit": min(max(limit, 1), 100),
                "page": max(page, 0),
            }
            if symbol:
                params["symbol"] = symbol

            result = self._request("GET", endpoint, params, signed=True)
            orders = result if isinstance(result, list) else result.get("data", [])

            normalized: List[Dict] = []
            for order in orders or []:
                try:
                    normalized.append(
                        {
                            "symbol": order.get("symbol"),
                            "size": float(order.get("size", 0) or 0),
                            "client_oid": order.get("client_oid"),
                            "create_time": int(order.get("createTime", 0) or 0),
                            "filled_qty": float(order.get("filled_qty", 0) or 0),
                            "fee": float(order.get("fee", 0) or 0),
                            "order_id": order.get("order_id"),
                            "price": float(order.get("price", 0) or 0),
                            "price_avg": float(order.get("price_avg", 0) or 0),
                            "status": order.get("status"),
                            "type": order.get("type"),
                            "order_type": order.get("order_type"),
                            "total_profits": float(order.get("totalProfits", 0) or 0),
                            "contracts": int(order.get("contracts", 0) or 0),
                            "filled_qty_contracts": int(
                                order.get("filledQtyContracts", 0) or 0
                            ),
                            "preset_tp": order.get("presetTakeProfitPrice"),
                            "preset_sl": order.get("presetStopLossPrice"),
                        }
                    )
                except (TypeError, ValueError):
                    continue

            return normalized
        except Exception as e:
            self._log.error(f"Failed to get current orders: {e}")
            raise RuntimeError(f"Failed to get current orders: {e}") from e

    def get_history_orders(
        self,
        symbol: Optional[str] = None,
        page_size: int = 100,
        create_date_ms: Optional[int] = None,
    ) -> List[Dict]:
        """Fetch historical orders."""
        try:
            if self._trading_mode == WeexTradingMode.SPOT:
                return []

            endpoint = "/capi/v2/order/history"
            params: Dict[str, str | int] = {
                "pageSize": min(max(page_size, 1), 100),
            }
            if symbol:
                params["symbol"] = symbol
            if create_date_ms:
                params["createDate"] = int(create_date_ms)

            result = self._request("GET", endpoint, params, signed=True)
            orders = result if isinstance(result, list) else result.get("data", [])

            normalized: List[Dict] = []
            for order in orders or []:
                try:
                    normalized.append(
                        {
                            "symbol": order.get("symbol"),
                            "size": float(order.get("size", 0) or 0),
                            "client_oid": order.get("client_oid"),
                            "create_time": int(order.get("createTime", 0) or 0),
                            "filled_qty": float(order.get("filled_qty", 0) or 0),
                            "fee": float(order.get("fee", 0) or 0),
                            "order_id": order.get("order_id"),
                            "price": float(order.get("price", 0) or 0),
                            "price_avg": float(order.get("price_avg", 0) or 0),
                            "status": order.get("status"),
                            "type": order.get("type"),
                            "order_type": order.get("order_type"),
                            "total_profits": float(order.get("totalProfits", 0) or 0),
                            "contracts": int(order.get("contracts", 0) or 0),
                            "filled_qty_contracts": int(
                                order.get("filledQtyContracts", 0) or 0
                            ),
                            "preset_tp": order.get("presetTakeProfitPrice"),
                            "preset_sl": order.get("presetStopLossPrice"),
                        }
                    )
                except (TypeError, ValueError):
                    continue

            return normalized
        except Exception as e:
            self._log.error(f"Failed to get history orders: {e}")
            raise RuntimeError(f"Failed to get history orders: {e}") from e

    def get_single_account(self, coin: str) -> Dict:
        """Get account information for a specific coin."""
        try:
            endpoint = "/capi/v2/account/getAccount"
            params = {"coin": coin}
            result = self._request("GET", endpoint, params, signed=True)
            return result if isinstance(result, dict) else {}
        except Exception as e:
            self._log.error(f"Failed to get single account for {coin}: {e}")
            raise RuntimeError(f"Failed to get single account for {coin}: {e}") from e

    def modify_user_account_mode(
        self,
        symbol: str,
        margin_mode: MarginMode,
        separated_mode: int = 1,
    ) -> None:
        """Modify user account mode (margin and separation)."""
        try:
            endpoint = "/capi/v2/account/position/changeHoldModel"
            params = {
                "symbol": symbol,
                "marginMode": self._margin_mode_code(margin_mode),
                "separatedMode": separated_mode,
            }
            self._request("POST", endpoint, params, signed=True)
            self._log.info(
                "Modified account mode for %s: marginMode=%s separatedMode=%s",
                symbol,
                margin_mode.value,
                separated_mode,
            )
        except Exception as e:
            self._log.error(f"Failed to modify account mode for {symbol}: {e}")
            raise RuntimeError(
                f"Failed to modify account mode for {symbol}: {e}"
            ) from e

    def set_leverage(self, symbol: str, leverage: int) -> None:
        """Set leverage for a symbol."""
        try:
            if self._trading_mode == WeexTradingMode.SPOT:
                self._log.warning("Leverage not applicable for spot trading")
                return
            self._set_leverage(symbol, leverage, MarginMode.CROSS)

        except Exception as e:
            self._log.error(f"Failed to set leverage for {symbol}: {e}")
            raise RuntimeError(f"Failed to set leverage: {e}") from e

    def set_margin_mode(self, symbol: str, margin_mode: MarginMode) -> None:
        """Set margin mode for a symbol."""
        try:
            if self._trading_mode == WeexTradingMode.SPOT:
                self._log.warning("Margin mode not applicable for spot trading")
                return

            endpoint = "/capi/v2/account/position/changeHoldModel"
            params = {
                "symbol": symbol,
                "marginMode": self._margin_mode_code(margin_mode),
                "separatedMode": 1,  # use combined as default
            }

            self._request("POST", endpoint, params, signed=True)
            self._log.info(f"Set margin mode for {symbol}: {margin_mode.value}")

        except Exception as e:
            # Margin mode might already be set, log as warning
            self._log.warning(f"Failed to set margin mode for {symbol}: {e}")

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
        """Open a market position with optional TP/SL."""
        try:
            if self._trading_mode == WeexTradingMode.SPOT:
                raise RuntimeError("Spot trading not implemented for Weex")

            # Ensure account settings match desired mode/leverage
            self.set_margin_mode(symbol, margin_mode)
            self._set_leverage(symbol, leverage, margin_mode)

            endpoint = "/capi/v2/order/placeOrder"
            order_type = "1" if side == PositionSide.LONG else "2"
            params: Dict[str, str | int | float] = {
                "symbol": symbol,
                "client_oid": str(uuid.uuid4().int)[:18],
                "size": str(quantity),
                "type": order_type,  # 1=open long, 2=open short, 3=close long, 4=close short
                "order_type": "3",  # IOC per doc: 0 normal,1 post-only,2 FOK,3 IOC
                "match_price": "1",  # 0 limit, 1 market
                "marginMode": self._margin_mode_code(margin_mode),
            }

            if take_profit is not None:
                params["presetTakeProfitPrice"] = str(take_profit)
            if stop_loss is not None:
                params["presetStopLossPrice"] = str(stop_loss)

            result = self._request("POST", endpoint, params, signed=True)
            order_data = result if isinstance(result, dict) else {}
            order_id = order_data.get("order_id") or order_data.get("orderId")

            self._log.info(
                f"Opened {side.value} position for {symbol}: qty={quantity} leverage={leverage} margin_mode={margin_mode.value}"
            )

            return OrderResult(
                order_id=str(order_id or uuid.uuid4()),
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                price=0.0,
                quantity=quantity,
                status="FILLED",
                timestamp=datetime.now(timezone.utc),
            )

        except Exception as e:
            self._log.error(f"Failed to open position for {symbol}: {e}")
            raise RuntimeError(f"Failed to open position: {e}") from e

    def close_position(
        self,
        symbol: str,
        side: Optional[PositionSide] = None,
    ) -> OrderResult:
        """Close an open position."""
        try:
            # Get current position to determine size
            position = self.get_position(symbol)
            if position is None:
                raise RuntimeError(f"No open position found for {symbol}")

            if side is not None and position.side != side:
                raise RuntimeError(
                    f"Position side mismatch: expected {side.value}, got {position.side.value}"
                )

            endpoint = "/capi/v2/order/placeOrder"
            order_type = "3" if position.side == PositionSide.LONG else "4"
            params = {
                "symbol": symbol,
                "client_oid": str(uuid.uuid4().int)[:18],
                "size": str(position.size),
                "type": order_type,  # 3=close long, 4=close short
                "order_type": "3",  # IOC
                "match_price": "1",
                "marginMode": self._margin_mode_code(position.margin_mode),
            }

            result = self._request("POST", endpoint, params, signed=True)
            order_data = result if isinstance(result, dict) else {}

            self._log.info(f"Closed position for {symbol}")

            return OrderResult(
                order_id=str(
                    order_data.get("order_id")
                    or order_data.get("orderId")
                    or uuid.uuid4()
                ),
                symbol=symbol,
                side=position.side,
                order_type=OrderType.MARKET,
                price=0.0,
                quantity=position.size,
                status="FILLED",
                timestamp=datetime.now(timezone.utc),
            )

        except Exception as e:
            self._log.error(f"Failed to close position for {symbol}: {e}")
            raise RuntimeError(f"Failed to close position: {e}") from e

    def cancel_all_orders(self, symbol: str) -> None:
        """Cancel all open orders for a symbol."""
        self._log.warning(
            "Cancel all orders not documented for Weex futures; skipping cancellation for %s",
            symbol,
        )

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[List]:
        """Get kline/candlestick data."""
        self._log.warning(
            "Klines endpoint not documented for Weex futures; returning empty list"
        )
        return []

    def test_connection(self) -> bool:
        """Test connectivity to the exchange."""
        try:
            endpoint = "/capi/v2/market/time"
            result = self._request("GET", endpoint)
            if isinstance(result, dict) and result.get("timestamp"):
                self._log.info("Weex connection test successful")
                return True
            self._log.warning("Weex connection test returned unexpected response")
            return False

        except Exception as e:
            self._log.error(f"Weex connection test failed: {e}")
            raise RuntimeError(f"Connection test failed: {e}") from e

    def close(self) -> None:
        """Close the exchange client and cleanup resources."""
        self._log.info("Closed Weex exchange client")


def _generate_signature(
    secret_key, timestamp, method, request_path, query_string, body
):
    message = (
        timestamp + method.upper() + request_path + query_string + str(body)
        if body
        else ""
    )
    signature = hmac.new(secret_key.encode(), message.encode(), hashlib.sha256).digest()
    return base64.b64encode(signature).decode()
