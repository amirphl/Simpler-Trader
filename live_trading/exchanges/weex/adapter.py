"""CCXT-backed Weex adapter for spot and USDT-margined perpetual trading."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from candle_downloader.binance import interval_to_milliseconds

from ...exchange import (
    Exchange,
    ExchangeConfig,
    MarginMode,
    OrderResult,
    OrderType,
    Position,
    PositionSide,
)


class WeexExchange(Exchange):
    """Adapt CCXT's Weex unified API to the project's exchange interface.

    ``trading_mode="futures"`` selects Weex USDT perpetual swaps; ``"spot"``
    selects spot markets.  Public callers continue to use compact symbols such
    as ``ZECUSDT`` while CCXT receives its canonical symbols (for example,
    ``ZEC/USDT:USDT`` for a perpetual).
    """

    _FUTURES = "futures"
    _SPOT = "spot"

    def __init__(
        self,
        config: ExchangeConfig,
        logger: Optional[logging.Logger] = None,
        *,
        symbols: Iterable[str] = (),
        client: Any = None,
    ) -> None:
        self._config = config
        self._log = logger or logging.getLogger(__name__)
        self._mode = self._normalize_mode(config.trading_mode)
        self._market_cache: Dict[str, Dict[str, Any]] = {}
        self._tracked_symbols = {
            self._normalize_symbol(symbol) for symbol in symbols if str(symbol).strip()
        }
        self._stop_order_ids: Dict[str, str] = {}
        self._margin_mode_by_symbol: Dict[str, str] = {}

        if client is not None:
            self._client = client
        else:
            self._client = self._create_ccxt_client(config)
        self._load_markets()

    @classmethod
    def _normalize_mode(cls, mode: str) -> str:
        normalized = str(mode or cls._FUTURES).strip().lower()
        if normalized in {"future", "futures", "swap", "perpetual"}:
            return cls._FUTURES
        if normalized == cls._SPOT:
            return cls._SPOT
        raise ValueError("Weex trading_mode must be 'futures' or 'spot'")

    def _create_ccxt_client(self, config: ExchangeConfig) -> Any:
        if config.testnet and self._mode == self._SPOT:
            raise ValueError(
                "Weex CCXT sandbox supports swap markets only; Weex spot trading "
                "requires mainnet (--no-testnet or --live)"
            )
        try:
            import ccxt
        except ImportError as exc:  # pragma: no cover - exercised at deployment
            raise RuntimeError(
                "Weex support requires the 'ccxt' package. Install project "
                "dependencies with: pip install -r requirements.txt"
            ) from exc

        if not config.passphrase:
            raise ValueError(
                "Weex requires API_PASSPHRASE (CCXT sends it as the API password)"
            )
        options: Dict[str, Any] = {
            "apiKey": config.api_key,
            "secret": config.api_secret,
            "password": config.passphrase,
            "enableRateLimit": True,
            "timeout": int(max(config.timeout, 0.1) * 1000),
            "options": {
                "defaultType": "swap" if self._mode == self._FUTURES else "spot",
            },
        }
        if config.proxies:
            proxy = config.proxies.get("https") or config.proxies.get("http")
            if proxy:
                # CCXT's synchronous Python exchange clients use this unified
                # proxy setting for REST requests.
                options["httpProxy"] = proxy

        exchange = ccxt.weex(options)
        if config.testnet:
            exchange.set_sandbox_mode(True)
        return exchange

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        return "".join(char for char in str(symbol).upper() if char.isalnum())

    @staticmethod
    def _market_is_futures(market: Dict[str, Any]) -> bool:
        return bool(market.get("swap") or market.get("contract"))

    @staticmethod
    def _market_is_spot(market: Dict[str, Any]) -> bool:
        return bool(market.get("spot")) and not bool(market.get("contract"))

    def _market_matches_mode(self, market: Dict[str, Any]) -> bool:
        if self._mode == self._FUTURES:
            return self._market_is_futures(market)
        return self._market_is_spot(market)

    def _load_markets(self) -> Dict[str, Dict[str, Any]]:
        markets = self._client.load_markets()
        if not isinstance(markets, dict):
            raise RuntimeError("Weex CCXT load_markets returned an invalid response")
        return markets

    def _market(self, symbol: str) -> Dict[str, Any]:
        compact = self._normalize_symbol(symbol)
        cached = self._market_cache.get(compact)
        if cached is not None:
            return cached

        markets = getattr(self._client, "markets", None) or self._load_markets()
        if not isinstance(markets, dict):
            raise RuntimeError("Weex CCXT market cache is invalid")

        candidates: List[Dict[str, Any]] = []
        supplied = markets.get(symbol)
        if isinstance(supplied, dict):
            candidates.append(supplied)
        for key, market in markets.items():
            if not isinstance(market, dict):
                continue
            identifiers = (key, market.get("symbol"), market.get("id"))
            if any(self._normalize_symbol(value) == compact for value in identifiers):
                candidates.append(market)

        for market in candidates:
            if self._market_matches_mode(market):
                canonical = str(market.get("symbol") or "").strip()
                if not canonical:
                    raise RuntimeError(f"Weex market metadata is missing a symbol for {symbol}")
                self._market_cache[compact] = market
                self._tracked_symbols.add(compact)
                return market

        market_kind = "perpetual futures" if self._mode == self._FUTURES else "spot"
        raise RuntimeError(f"Weex {market_kind} market is unavailable for {symbol}")

    def _ccxt_symbol(self, symbol: str) -> str:
        return str(self._market(symbol)["symbol"])

    @staticmethod
    def _legacy_symbol(market: Dict[str, Any]) -> str:
        market_id = str(market.get("id") or "").strip()
        if market_id:
            return market_id.upper()
        return WeexExchange._normalize_symbol(str(market.get("symbol") or ""))

    def _legacy_symbol_for_ccxt_symbol(self, symbol: str) -> str:
        markets = getattr(self._client, "markets", {})
        market = markets.get(symbol) if isinstance(markets, dict) else None
        if isinstance(market, dict):
            return self._legacy_symbol(market)
        return self._normalize_symbol(symbol)

    def _params_for_mode(self) -> Dict[str, str]:
        return {"type": "swap" if self._mode == self._FUTURES else "spot"}

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _order_id(order: Dict[str, Any]) -> str:
        info = order.get("info") if isinstance(order.get("info"), dict) else {}
        return str(
            order.get("id")
            or order.get("orderId")
            or order.get("algoId")
            or info.get("orderId")
            or info.get("algoId")
            or ""
        ).strip()

    @staticmethod
    def _project_order_status(status: Any, default: str = "NEW") -> str:
        """Translate CCXT's unified status values to this project's vocabulary."""
        normalized = str(status or default).strip().lower()
        return {
            "open": "NEW",
            "closed": "FILLED",
            "canceled": "CANCELED",
            "cancelled": "CANCELED",
            "expired": "EXPIRED",
            "rejected": "REJECTED",
            "partially_filled": "PARTIALLY_FILLED",
            "partiallyfilled": "PARTIALLY_FILLED",
        }.get(normalized, normalized.upper())

    @staticmethod
    def _matches_client_id(order: Dict[str, Any], client_id: str) -> bool:
        info = order.get("info") if isinstance(order.get("info"), dict) else {}
        return any(
            str(value or "").strip() == client_id
            for value in (
                order.get("clientOrderId"),
                order.get("client_id"),
                info.get("clientOrderId"),
                info.get("origClientOrderId"),
            )
        )

    def _quantity(self, symbol: str, quantity: float) -> float:
        quantity = float(quantity)
        if quantity <= 0:
            raise RuntimeError(f"Invalid order quantity for {symbol}: {quantity}")
        ccxt_symbol = self._ccxt_symbol(symbol)
        normalized = self._as_float(self._client.amount_to_precision(ccxt_symbol, quantity))
        if normalized <= 0:
            raise RuntimeError(f"Weex normalized quantity is invalid for {symbol}")
        market = self._market(symbol)
        min_amount = self._as_float(
            ((market.get("limits") or {}).get("amount") or {}).get("min")
        )
        if min_amount > 0 and normalized < min_amount:
            raise RuntimeError(
                f"Quantity {normalized} is below Weex minimum {min_amount} for {symbol}"
            )
        return normalized

    def validate_order_quantity(self, symbol: str, quantity: float) -> float:
        """Return a CCXT precision-normalized, valid order quantity."""
        return self._quantity(symbol, quantity)

    def _price(self, symbol: str, price: float) -> float:
        price = float(price)
        if price <= 0:
            raise RuntimeError(f"Invalid order price for {symbol}: {price}")
        normalized = self._as_float(
            self._client.price_to_precision(self._ccxt_symbol(symbol), price)
        )
        if normalized <= 0:
            raise RuntimeError(f"Weex normalized price is invalid for {symbol}")
        return normalized

    def _order_result(
        self,
        order: Dict[str, Any],
        *,
        symbol: str,
        side: PositionSide,
        order_type: OrderType,
        quantity: float,
        price: float,
    ) -> OrderResult:
        order_id = self._order_id(order)
        if not order_id:
            raise RuntimeError("Weex order response did not include an order id")
        timestamp = self._as_float(order.get("timestamp"))
        occurred_at = (
            datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
            if timestamp > 0
            else datetime.now(timezone.utc)
        )
        return OrderResult(
            order_id=order_id,
            symbol=self._normalize_symbol(symbol),
            side=side,
            order_type=order_type,
            price=self._as_float(order.get("price"), price),
            quantity=self._as_float(order.get("amount"), quantity),
            status=self._project_order_status(order.get("status")),
            timestamp=occurred_at,
        )

    def get_account_balance_for_symbol(self, symbol: str) -> float:
        """Fetch the account balance that funds an entry for ``symbol``."""
        del symbol  # Accounts are venue-scoped; the argument enables routing callers.
        return self.get_account_balance()

    def get_account_balance(self) -> float:
        balance = self._client.fetch_balance(self._params_for_mode())
        usdt = balance.get("USDT", {}) if isinstance(balance, dict) else {}
        available = self._as_float(
            usdt.get("free") if isinstance(usdt, dict) else None,
            self._as_float(usdt.get("total") if isinstance(usdt, dict) else None),
        )
        if available < 0:
            raise RuntimeError("Weex returned a negative USDT balance")
        return available

    def fetch_price(self, symbol: str) -> Optional[float]:
        ticker = self._client.fetch_ticker(self._ccxt_symbol(symbol))
        for key in ("last", "mark", "close", "bid", "ask"):
            value = self._as_float(ticker.get(key) if isinstance(ticker, dict) else None)
            if value > 0:
                return value
        return None

    def get_24h_tickers(self) -> List[Dict[str, Any]]:
        # CCXT's first positional argument is ``symbols``. Passing the market
        # type positionally makes the request invalid on the current Weex API.
        tickers = self._client.fetch_tickers(params=self._params_for_mode())
        if not isinstance(tickers, dict):
            return []
        result: List[Dict[str, Any]] = []
        for ccxt_symbol, ticker in tickers.items():
            try:
                market = self._market(str(ccxt_symbol))
            except RuntimeError:
                continue
            result.append(
                {
                    "symbol": self._legacy_symbol(market),
                    "priceChange": self._as_float(ticker.get("change")),
                    "priceChangePercent": self._as_float(ticker.get("percentage")),
                    "lastPrice": self._as_float(ticker.get("last")),
                    "volume": self._as_float(ticker.get("baseVolume")),
                    "quoteVolume": self._as_float(ticker.get("quoteVolume")),
                }
            )
        return result

    def _position_from_ccxt(self, raw: Dict[str, Any]) -> Optional[Position]:
        contracts = self._as_float(raw.get("contracts"), self._as_float(raw.get("size")))
        if contracts == 0:
            return None
        side = str(raw.get("side") or "").strip().lower()
        if side in {"long", "buy"}:
            position_side = PositionSide.LONG
        elif side in {"short", "sell"}:
            position_side = PositionSide.SHORT
        else:
            raise RuntimeError(f"Weex returned an unknown position side: {side!r}")
        margin_mode = (
            MarginMode.ISOLATED
            if str(raw.get("marginMode") or "").lower() == "isolated"
            else MarginMode.CROSS
        )
        ccxt_symbol = str(raw.get("symbol") or "")
        if not ccxt_symbol:
            raise RuntimeError("Weex position is missing its symbol")
        return Position(
            symbol=self._legacy_symbol_for_ccxt_symbol(ccxt_symbol),
            side=position_side,
            size=abs(contracts),
            entry_price=self._as_float(raw.get("entryPrice")),
            leverage=self._as_float(raw.get("leverage"), 1.0),
            margin_mode=margin_mode,
            unrealized_pnl=self._as_float(raw.get("unrealizedPnl")),
            liquidation_price=(
                self._as_float(raw.get("liquidationPrice"))
                if self._as_float(raw.get("liquidationPrice")) > 0
                else None
            ),
            position_id=str(raw.get("id") or raw.get("positionId") or "") or None,
        )

    def _spot_position(self, symbol: str) -> Optional[Position]:
        market = self._market(symbol)
        base = str(market.get("base") or "").upper()
        if not base:
            raise RuntimeError(f"Weex spot market metadata has no base asset for {symbol}")
        balance = self._client.fetch_balance({"type": "spot"})
        asset = balance.get(base, {}) if isinstance(balance, dict) else {}
        quantity = self._as_float(asset.get("total") if isinstance(asset, dict) else None)
        if quantity <= 0:
            return None
        return Position(
            symbol=self._legacy_symbol(market),
            side=PositionSide.LONG,
            size=quantity,
            entry_price=0.0,
            leverage=1.0,
            margin_mode=MarginMode.CROSS,
            unrealized_pnl=0.0,
            position_id=f"spot:{self._legacy_symbol(market)}",
        )

    def get_current_positions(self) -> List[Position]:
        if self._mode == self._FUTURES:
            rows = self._client.fetch_positions(params={"type": "swap"})
            positions = [self._position_from_ccxt(row) for row in rows or []]
            return [position for position in positions if position is not None]

        # A spot balance is an asset holding, not a futures position. Restrict
        # strategy-created adapters to their configured symbols so unrelated
        # wallet assets can never be claimed as a bot position.
        tracked = sorted(self._tracked_symbols)
        if not tracked:
            return []
        positions: List[Position] = []
        for compact_symbol in tracked:
            position = self._spot_position(compact_symbol)
            if position is not None:
                positions.append(position)
        return positions

    def get_position(self, symbol: str) -> Optional[Position]:
        if self._mode == self._SPOT:
            return self._spot_position(symbol)
        ccxt_symbol = self._ccxt_symbol(symbol)
        rows = self._client.fetch_positions_for_symbol(ccxt_symbol)
        positions = [self._position_from_ccxt(row) for row in rows or []]
        live_positions = [position for position in positions if position is not None]
        if len(live_positions) > 1:
            raise RuntimeError(f"Weex returned multiple positions for {symbol}")
        return live_positions[0] if live_positions else None

    def set_leverage(self, symbol: str, leverage: int) -> None:
        if self._mode == self._SPOT:
            if leverage != 1:
                raise RuntimeError("Weex spot trading does not support leverage")
            return
        if leverage <= 0:
            raise RuntimeError("Leverage must be positive")
        margin_mode = self._margin_mode_by_symbol.get(self._normalize_symbol(symbol))
        params = {"marginMode": margin_mode} if margin_mode else {}
        self._client.set_leverage(int(leverage), self._ccxt_symbol(symbol), params)

    def set_margin_mode(self, symbol: str, margin_mode: MarginMode) -> None:
        if self._mode == self._SPOT:
            return
        ccxt_mode = "isolated" if margin_mode is MarginMode.ISOLATED else "cross"
        self._client.set_margin_mode(ccxt_mode, self._ccxt_symbol(symbol))
        self._margin_mode_by_symbol[self._normalize_symbol(symbol)] = ccxt_mode

    def _entry_params(
        self,
        *,
        client_id: Optional[str],
        stop_loss: Optional[float],
        take_profit: Optional[float],
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if client_id:
            params["clientOrderId"] = client_id
        if self._mode == self._FUTURES:
            if stop_loss is not None:
                params["stopLoss"] = {
                    "triggerPrice": float(stop_loss),
                    "triggerPriceType": "mark",
                }
            if take_profit is not None:
                params["takeProfit"] = {
                    "triggerPrice": float(take_profit),
                    "triggerPriceType": "mark",
                }
        elif stop_loss is not None or take_profit is not None:
            raise RuntimeError(
                "Weex spot orders do not provide a native TP/SL API through CCXT; "
                "refusing to submit an unprotected order"
            )
        return params

    def _validate_spot_entry(self, side: PositionSide) -> None:
        if self._mode == self._SPOT and side is PositionSide.SHORT:
            raise RuntimeError("Weex spot trading cannot open short positions")

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
        del leverage, margin_mode
        self._validate_spot_entry(side)
        normalized_quantity = self._quantity(symbol, quantity)
        order = self._client.create_order(
            self._ccxt_symbol(symbol),
            "market",
            "buy" if side is PositionSide.LONG else "sell",
            normalized_quantity,
            None,
            self._entry_params(
                client_id=None, stop_loss=stop_loss, take_profit=take_profit
            ),
        )
        return self._order_result(
            order,
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=normalized_quantity,
            price=0.0,
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
        client_id: Optional[str] = None,
    ) -> OrderResult:
        del leverage, margin_mode
        self._validate_spot_entry(side)
        normalized_quantity = self._quantity(symbol, quantity)
        normalized_price = self._price(symbol, price)
        order = self._client.create_order(
            self._ccxt_symbol(symbol),
            "limit",
            "buy" if side is PositionSide.LONG else "sell",
            normalized_quantity,
            normalized_price,
            self._entry_params(
                client_id=client_id, stop_loss=stop_loss, take_profit=take_profit
            ),
        )
        return self._order_result(
            order,
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=normalized_quantity,
            price=normalized_price,
        )

    def close_position(
        self, symbol: str, side: Optional[PositionSide] = None
    ) -> OrderResult:
        position = self.get_position(symbol)
        if position is None:
            raise RuntimeError(f"Weex close position failed: no open position for {symbol}")
        if side is not None and position.side is not side:
            raise RuntimeError(
                f"Weex position side changed for {symbol}: expected {side.value}, "
                f"found {position.side.value}"
            )

        if self._mode == self._FUTURES:
            order = self._client.close_position(
                self._ccxt_symbol(symbol),
                "sell" if position.side is PositionSide.LONG else "buy",
            )
            result_side = (
                PositionSide.SHORT
                if position.side is PositionSide.LONG
                else PositionSide.LONG
            )
        else:
            order = self._client.create_order(
                self._ccxt_symbol(symbol), "market", "sell", position.size
            )
            result_side = PositionSide.SHORT
        return self._order_result(
            order,
            symbol=symbol,
            side=result_side,
            order_type=OrderType.MARKET,
            quantity=position.size,
            price=0.0,
        )

    def cancel_all_orders(self, symbol: str) -> None:
        """Cancel regular and, for swaps, standalone trigger orders."""
        ccxt_symbol = self._ccxt_symbol(symbol)
        errors: List[Exception] = []
        for params in ({}, {"trigger": True} if self._mode == self._FUTURES else None):
            if params is None:
                continue
            try:
                self._client.cancel_all_orders(ccxt_symbol, params)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(f"Weex failed to cancel all orders for {symbol}") from errors[0]

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        if not str(order_id).strip():
            return False
        ccxt_symbol = self._ccxt_symbol(symbol)
        trigger_params: Dict[str, bool] = {}
        if self._mode == self._FUTURES:
            try:
                trigger_orders = self._client.fetch_open_orders(
                    ccxt_symbol, params={"trigger": True}
                )
                if any(
                    self._order_id(order) == str(order_id)
                    for order in trigger_orders or []
                    if isinstance(order, dict)
                ):
                    trigger_params = {"trigger": True}
            except Exception as exc:
                self._log.warning(
                    "Weex: unable to identify trigger order %s/%s: %s",
                    symbol,
                    order_id,
                    exc,
                )
        try:
            self._client.cancel_order(str(order_id), ccxt_symbol, trigger_params)
            return True
        except Exception as exc:
            if self._mode == self._FUTURES and not trigger_params:
                try:
                    self._client.cancel_order(
                        str(order_id), ccxt_symbol, {"trigger": True}
                    )
                    return True
                except Exception:
                    pass
            self._log.warning("Weex: cancel order failed %s/%s: %s", symbol, order_id, exc)
            return False

    def _stop_orders(self, position: Position) -> List[Dict[str, Any]]:
        if self._mode != self._FUTURES:
            return []
        rows = self._client.fetch_open_orders(
            self._ccxt_symbol(position.symbol), params={"trigger": True}
        )
        result: List[Dict[str, Any]] = []
        expected_side = "sell" if position.side is PositionSide.LONG else "buy"
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            if not bool(row.get("reduceOnly")):
                continue
            if str(row.get("side") or "").lower() != expected_side:
                continue
            trigger = self._as_float(row.get("triggerPrice"), self._as_float(row.get("stopLossPrice")))
            if trigger > 0:
                result.append(row)
        return result

    @staticmethod
    def _is_protective(
        side: PositionSide, existing: float, requested: float
    ) -> bool:
        return existing >= requested if side is PositionSide.LONG else existing <= requested

    def ensure_position_stop_loss(
        self, position: Position, stop_price: float
    ) -> Optional[float]:
        """Install and read back a standalone reduce-only futures stop order."""
        if self._mode != self._FUTURES:
            self._log.warning("Weex spot does not expose a native protective-stop API")
            return None
        normalized_stop = self._price(position.symbol, stop_price)
        existing_orders = self._stop_orders(position)
        for order in existing_orders:
            existing = self._as_float(order.get("triggerPrice"), self._as_float(order.get("stopLossPrice")))
            if self._is_protective(position.side, existing, normalized_stop):
                order_id = self._order_id(order)
                if position.position_id and order_id:
                    self._stop_order_ids[position.position_id] = order_id
                return existing

        close_side = "sell" if position.side is PositionSide.LONG else "buy"
        created = self._client.create_order(
            self._ccxt_symbol(position.symbol),
            "market",
            close_side,
            self._quantity(position.symbol, position.size),
            None,
            {
                "stopLossPrice": normalized_stop,
                "stopLossPriceType": "mark",
                "reduceOnly": True,
            },
        )
        created_id = self._order_id(created)
        if not created_id:
            self._log.warning("Weex stop-loss response for %s did not include an id", position.symbol)
            return None

        # An acknowledgement is insufficient for EMA+AVWAP. Require the stop
        # to be observable through CCXT's trigger-order endpoint.
        for order in self._stop_orders(position):
            order_id = self._order_id(order)
            actual = self._as_float(order.get("triggerPrice"), self._as_float(order.get("stopLossPrice")))
            if order_id == created_id and actual > 0:
                if position.position_id:
                    self._stop_order_ids[position.position_id] = created_id
                # Install and verify the replacement before removing older
                # stops; cancelling first would create an unprotected gap.
                for existing_order in existing_orders:
                    existing_id = self._order_id(existing_order)
                    if not existing_id or existing_id == created_id:
                        continue
                    try:
                        self._client.cancel_order(
                            existing_id,
                            self._ccxt_symbol(position.symbol),
                            {"trigger": True},
                        )
                    except Exception as exc:
                        # Retaining two reduce-only stops is safer than losing
                        # protection if the old-stop cancellation is transient.
                        self._log.warning(
                            "Weex: retained old protective stop %s for %s after "
                            "replacement confirmation failed to cancel it: %s",
                            existing_id,
                            position.symbol,
                            exc,
                        )
                return actual
        return None

    def place_stop_loss_order(self, position: Position, stop_price: float) -> Optional[str]:
        actual = self.ensure_position_stop_loss(position, stop_price)
        if actual is None:
            return None
        return self._stop_order_ids.get(position.position_id or "")

    def update_stop_loss_order(
        self, position: Position, order_id: str, stop_price: float
    ) -> bool:
        del order_id
        return self.ensure_position_stop_loss(position, stop_price) is not None

    def update_position_stop_loss(self, position: Position, stop_price: float) -> bool:
        return self.ensure_position_stop_loss(position, stop_price) is not None

    def get_order_status(
        self,
        *,
        symbol: str,
        order_id: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not order_id and not client_id:
            return None
        ccxt_symbol = self._ccxt_symbol(symbol)
        if order_id:
            # The Weex futures endpoint accepts its server order id, not a
            # client id. Supplying both makes CCXT prefer clientOrderId.
            order = self._client.fetch_order(order_id, ccxt_symbol)
        elif self._mode == self._SPOT:
            order = self._client.fetch_order(
                None, ccxt_symbol, {"clientOrderId": client_id}
            )
        else:
            order = self._find_futures_order_by_client_id(ccxt_symbol, client_id or "")
        if not isinstance(order, dict):
            return None
        normalized = dict(order)
        normalized["status"] = self._project_order_status(order.get("status"))
        return normalized

    def _find_futures_order_by_client_id(
        self, ccxt_symbol: str, client_id: str
    ) -> Optional[Dict[str, Any]]:
        """Find an uncertain futures submission without re-submitting it."""
        for fetcher, kwargs in (
            (self._client.fetch_open_orders, {}),
            (getattr(self._client, "fetch_canceled_and_closed_orders", None), {"limit": 100}),
        ):
            if not callable(fetcher):
                continue
            rows = fetcher(ccxt_symbol, **kwargs)
            for row in rows or []:
                if isinstance(row, dict) and self._matches_client_id(row, client_id):
                    return row
        return None

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[List]:
        rows = self._client.fetch_ohlcv(
            self._ccxt_symbol(symbol), interval, start_time, max(int(limit), 1)
        )
        interval_ms = interval_to_milliseconds(interval)
        result: List[List] = []
        for row in rows or []:
            if not isinstance(row, (list, tuple)) or len(row) < 6:
                continue
            open_time = int(row[0])
            if end_time is not None and open_time > end_time:
                continue
            close_time = open_time + interval_ms - 1
            result.append(
                [
                    open_time,
                    str(row[1]),
                    str(row[2]),
                    str(row[3]),
                    str(row[4]),
                    str(row[5]),
                    close_time,
                    "0",
                    0,
                    "0",
                    "0",
                    "0",
                ]
            )
        return result

    def get_trading_pairs(self, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        requested = symbols or []
        if requested:
            markets = [self._market(symbol) for symbol in requested]
        else:
            raw_markets = getattr(self._client, "markets", {}) or self._load_markets()
            markets = [
                market
                for market in raw_markets.values()
                if isinstance(market, dict) and self._market_matches_mode(market)
            ]
        return [
            {
                "symbol": self._legacy_symbol(market),
                "base": market.get("base"),
                "quote": market.get("quote"),
                "active": market.get("active"),
                "precision": market.get("precision", {}),
                "limits": market.get("limits", {}),
            }
            for market in markets
        ]

    def get_depth(self, symbol: str, limit: Optional[str | int] = None) -> Dict[str, Any]:
        requested_limit = int(limit) if limit is not None else None
        return self._client.fetch_order_book(self._ccxt_symbol(symbol), requested_limit)

    def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        if self._mode != self._FUTURES:
            raise RuntimeError("Funding rates are only available for Weex futures")
        return self._client.fetch_funding_rate(self._ccxt_symbol(symbol))

    def validate_ema_avwap_execution(self) -> None:
        if self._mode != self._FUTURES:
            raise RuntimeError(
                "EMA+AVWAP requires Weex futures: Weex spot has no CCXT-native "
                "protective-stop endpoint for this strategy's fail-closed safety model"
            )
        for symbol in sorted(self._tracked_symbols):
            mode = self._client.fetch_position_mode(self._ccxt_symbol(symbol))
            if isinstance(mode, dict) and mode.get("hedged") is True:
                raise RuntimeError(
                    "EMA+AVWAP requires Weex one-way position mode; hedge mode is unsupported"
                )

    def test_connection(self) -> bool:
        try:
            self._client.fetch_time()
            return True
        except Exception as exc:
            raise RuntimeError(f"Weex connection test failed: {exc}") from exc

    def close(self) -> None:
        closer = getattr(self._client, "close", None)
        if callable(closer):
            closer()
