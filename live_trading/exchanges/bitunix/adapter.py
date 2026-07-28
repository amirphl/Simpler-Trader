"""Bitunix exchange adapter implementing the shared Exchange interface."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ...exchange import (
    Exchange,
    ExchangeConfig,
    KlineUpdate,
    MarginMode,
    OrderResult,
    OrderType,
    Position,
    PositionSide,
)
from .client import BitunixClient
from .kline_stream import BitunixKlineStream
from .utils import infer_margin_coin_from_symbol, interval_to_milliseconds


class BitunixExchange(Exchange):
    """Bitunix futures exchange adapter."""

    _KLINE_PAGE_SIZE = 200
    # The exchange rate-limits public kline history by source IP.  EMA+AVWAP
    # starts several per-symbol fetches concurrently and the account launcher
    # can run three coordinators on the same host, so serialize each account's
    # page requests.  Across three accounts this caps the startup burst at six
    # history requests/second instead of dozens at once.
    _KLINE_REQUEST_MIN_INTERVAL_SECONDS = 0.5

    def __init__(
        self, config: ExchangeConfig, logger: Optional[logging.Logger] = None
    ) -> None:
        self._config = config
        self._log = logger or logging.getLogger(__name__)
        self._client = BitunixClient(config, self._log)
        self._default_margin_coin = "USDT"
        self._pair_meta_cache: Dict[str, Dict[str, Any]] = {}
        self._kline_request_lock = threading.Lock()
        self._next_kline_request_at = 0.0

    def _wait_for_kline_request_slot(self) -> None:
        """Reserve a paced slot for one public Bitunix kline request."""
        with self._kline_request_lock:
            now = time.monotonic()
            wait_seconds = max(0.0, self._next_kline_request_at - now)
            # Reserve before sleeping so concurrent symbol fetches cannot all
            # wake up and issue their requests together.
            self._next_kline_request_at = max(
                now, self._next_kline_request_at
            ) + self._KLINE_REQUEST_MIN_INTERVAL_SECONDS
        if wait_seconds > 0:
            time.sleep(wait_seconds)

    @staticmethod
    def _quantize(
        value: float, decimals: int, *, rounding_mode: str = ROUND_DOWN
    ) -> float:
        decimals = max(int(decimals), 0)
        quantum = Decimal("1").scaleb(-decimals)
        quantized = Decimal(str(value)).quantize(quantum, rounding=rounding_mode)
        return float(quantized)

    def _get_symbol_meta(self, symbol: str) -> Dict[str, Any]:
        normalized_symbol = str(symbol).strip().upper()
        cached = self._pair_meta_cache.get(normalized_symbol)
        if cached is not None:
            return cached

        for item in self.get_trading_pairs(symbols=[normalized_symbol]):
            if str(item.get("symbol", "")).strip().upper() == normalized_symbol:
                self._pair_meta_cache[normalized_symbol] = item
                return item
        return {}

    def _normalize_quantity(self, symbol: str, quantity: float) -> float:
        qty = float(quantity)
        if qty <= 0:
            raise RuntimeError(f"Invalid order quantity for {symbol}: {quantity}")

        meta = self._get_symbol_meta(symbol)
        base_precision = int(meta.get("basePrecision", 8) or 8)
        min_trade_volume = float(meta.get("minTradeVolume", 0) or 0)

        normalized_qty = self._quantize(qty, base_precision, rounding_mode=ROUND_DOWN)
        if min_trade_volume > 0 and normalized_qty < min_trade_volume:
            raise RuntimeError(
                f"Quantity {normalized_qty} below minTradeVolume {min_trade_volume} for {symbol}"
            )
        return normalized_qty

    def _normalize_limit_price(
        self, symbol: str, side: PositionSide, price: float
    ) -> float:
        px = float(price)
        if px <= 0:
            raise RuntimeError(f"Invalid order price for {symbol}: {price}")

        meta = self._get_symbol_meta(symbol)
        quote_precision = int(meta.get("quotePrecision", 8) or 8)
        price_protect_scope = float(meta.get("priceProtectScope", 0) or 0)
        tick_size = 10 ** (-max(quote_precision, 0))
        mark_price = self.fetch_price(symbol)
        adjusted = px

        # Keep limit price within exchange price-protection band to avoid 30014/30015.
        if mark_price is not None and mark_price > 0 and price_protect_scope > 0:
            if side == PositionSide.LONG:
                max_buy = mark_price * (1.0 + price_protect_scope)
                adjusted = min(adjusted, max_buy - tick_size)
            else:
                min_sell = mark_price * (1.0 - price_protect_scope)
                adjusted = max(adjusted, min_sell + tick_size)

        rounding = ROUND_DOWN if side == PositionSide.LONG else ROUND_UP
        normalized_price = self._quantize(
            adjusted, quote_precision, rounding_mode=rounding
        )

        if normalized_price <= 0:
            raise RuntimeError(
                f"Normalized order price is invalid for {symbol}: raw={price} normalized={normalized_price}"
            )
        if abs(normalized_price - px) > 1e-12:
            self._log.info(
                "Bitunix: normalized %s limit price for %s from %.10f to %.10f",
                side.value,
                symbol,
                px,
                normalized_price,
            )
        return normalized_price

    def _normalize_stop_loss_price(
        self, position: Position, stop_price: float
    ) -> float:
        """Normalize SL to symbol precision and keep it on valid side of last price."""
        return self._normalize_directional_stop_loss_price(
            position.symbol, position.side, stop_price
        )

    def _normalize_directional_stop_loss_price(
        self, symbol: str, side: PositionSide, stop_price: float
    ) -> float:
        """Normalize an SL for a position side before sending it to Bitunix."""
        meta = self._get_symbol_meta(symbol)
        quote_precision = int(meta.get("quotePrecision", 8) or 8)
        tick_size = 10 ** (-max(quote_precision, 0))
        mark_price = self.fetch_price(symbol)
        safety_offset = tick_size * 2  # keep a small buffer away from trigger boundary

        adjusted = float(stop_price)
        if mark_price is not None and mark_price > 0:
            if side == PositionSide.LONG:
                adjusted = min(adjusted, mark_price - safety_offset)
            else:
                adjusted = max(adjusted, mark_price + safety_offset)

        # Never widen a protective stop during tick normalization: a long stop
        # rounds up and a short stop rounds down.  If the market has already
        # crossed the requested level, the safety-offset adjustment above will
        # make verification fail and the strategy will emergency-close.
        rounding = ROUND_UP if side == PositionSide.LONG else ROUND_DOWN
        normalized = self._quantize(adjusted, quote_precision, rounding_mode=rounding)
        if normalized <= 0:
            raise RuntimeError(
                f"Normalized stop loss is invalid for {symbol}: raw={stop_price} normalized={normalized}"
            )
        return normalized

    @staticmethod
    def _parse_position_side(raw_side: Any) -> PositionSide:
        """Translate the documented position payload without guessing."""
        side = str(raw_side or "").strip().upper()
        if side in {"SHORT", "SELL"}:
            return PositionSide.SHORT
        if side in {"LONG", "BUY"}:
            return PositionSide.LONG
        raise RuntimeError(f"Unknown Bitunix position side: {raw_side!r}")

    def get_position_mode(self) -> str:
        """Return the account's actual Bitunix position mode, or fail closed."""
        try:
            account = self._client.get_single_account(self._default_margin_coin)
            mode = str(account.get("positionMode", "")).strip().upper()
        except Exception as exc:
            raise RuntimeError("Failed to fetch Bitunix position mode") from exc
        if mode not in {"ONE_WAY", "HEDGE"}:
            raise RuntimeError(f"Unknown Bitunix position mode: {mode!r}")
        return mode

    def validate_ema_avwap_execution(self) -> None:
        """EMA/AVWAP owns one net position per symbol, never a hedge pair."""
        mode = self.get_position_mode()
        if mode != "ONE_WAY":
            raise RuntimeError(
                "EMA+AVWAP requires Bitunix ONE_WAY position mode; refusing to "
                "trade a HEDGE account because the strategy tracks one net "
                "position per symbol"
            )

    def _positions_for_symbol(self, symbol: str) -> List[Position]:
        normalized = str(symbol).strip().upper()
        return [
            position
            for position in self.get_current_positions()
            if position.symbol.upper() == normalized
        ]

    def fetch_price(self, symbol: str) -> Optional[float]:
        """Fetch last price for a symbol (public endpoint)."""
        return self._client.fetch_price(symbol)

    def start_kline_stream(
        self,
        *,
        symbols: tuple[str, ...],
        interval: str,
        on_kline: Callable[[KlineUpdate], None],
    ) -> BitunixKlineStream:
        """Start a public stream of forming candles for live indicator updates."""
        stream = BitunixKlineStream(
            symbols=symbols,
            interval=interval,
            on_kline=on_kline,
            logger=self._log,
            proxies=self._config.proxies,
        )
        stream.start()
        return stream

    def get_account_balance(self) -> float:
        margin_coin = self._default_margin_coin
        balance = self._client.get_available_balance(margin_coin)
        if balance is None:
            raise RuntimeError("Failed to fetch Bitunix balance")
        return balance

    def get_24h_tickers(self) -> List[Dict[str, Any]]:
        ticks = self._client.fetch_tickers()
        return list(ticks.values())

    def get_trading_pairs(
        self, symbols: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Fetch trading pair metadata."""
        pairs = self._client.get_trading_pairs(symbols=symbols)
        return list(pairs.values())

    def get_depth(
        self, symbol: str, limit: Optional[str | int] = None
    ) -> Dict[str, Any]:
        """Fetch order book depth for a symbol."""
        return self._client.get_depth(symbol=symbol, limit=limit)

    def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """Fetch current funding rate for a symbol."""
        return self._client.get_funding_rate(symbol=symbol)

    def get_position_tiers(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch position tier settings for a symbol."""
        return self._client.get_position_tiers(symbol=symbol)

    def place_position_tpsl_order(
        self,
        symbol: str,
        position_id: str,
        tp_price: Optional[float] = None,
        tp_stop_type: Optional[str] = None,
        sl_price: Optional[float] = None,
        sl_stop_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Place TP/SL order bound to an existing position."""
        return self._client.place_position_tpsl_order(
            symbol=symbol,
            position_id=position_id,
            tp_price=tp_price,
            tp_stop_type=tp_stop_type,
            sl_price=sl_price,
            sl_stop_type=sl_stop_type,
        )

    def get_current_positions(self) -> List[Position]:
        try:
            positions = self._client.get_pending_positions()
        except Exception as exc:
            self._log.error("Bitunix: failed to fetch pending positions: %s", exc)
            raise RuntimeError("Bitunix failed to fetch current positions") from exc

        normalized: List[Position] = []
        for pos in positions or []:
            try:
                size = float(pos.get("qty", 0) or 0)
                if size == 0:
                    continue
                side = self._parse_position_side(pos.get("side"))
                margin_mode_str = str(pos.get("marginMode", "")).upper()
                if margin_mode_str == "ISOLATION":
                    margin_mode = MarginMode.ISOLATED
                elif margin_mode_str == "CROSS":
                    margin_mode = MarginMode.CROSS
                else:
                    raise RuntimeError(
                        f"Unknown Bitunix margin mode: {margin_mode_str!r}"
                    )
                symbol = str(pos.get("symbol") or "").strip().upper()
                if not symbol:
                    raise RuntimeError("Bitunix position payload is missing symbol")
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
                        symbol=symbol,
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
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid Bitunix position payload: {pos!r}"
                ) from exc

        return normalized

    def get_position(self, symbol: str) -> Optional[Position]:
        positions = self._positions_for_symbol(symbol)
        if not positions:
            return None
        if len(positions) > 1:
            raise RuntimeError(
                f"Ambiguous Bitunix position lookup for {symbol}: "
                "multiple hedge positions are open"
            )
        return positions[0]

    def set_leverage(self, symbol: str, leverage: int) -> None:
        margin_coin = infer_margin_coin_from_symbol(symbol)
        if not self._client.change_leverage(margin_coin, symbol, leverage):
            raise RuntimeError(f"Failed to set leverage for {symbol}")

    def set_margin_mode(self, symbol: str, margin_mode: MarginMode) -> None:
        margin_coin = infer_margin_coin_from_symbol(symbol)
        mode_str = "ISOLATION" if margin_mode == MarginMode.ISOLATED else "CROSS"
        if not self._client.change_margin_mode(margin_coin, symbol, mode_str):
            raise RuntimeError(f"Failed to set margin mode {mode_str} for {symbol}")

    def adjust_position_margin(
        self,
        symbol: str,
        amount: float,
        side: Optional[PositionSide] = None,
        position_id: Optional[str] = None,
    ) -> None:
        """Adjust margin for an isolated position."""
        margin_coin = infer_margin_coin_from_symbol(symbol)
        side_str = side.value if side is not None else None
        if not self._client.adjust_position_margin(
            margin_coin=margin_coin,
            symbol=symbol,
            amount=amount,
            side=side_str,
            position_id=position_id,
        ):
            raise RuntimeError(f"Failed to adjust position margin for {symbol}")

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
        normalized_qty = self._normalize_quantity(symbol, quantity)
        normalized_stop_loss = (
            self._normalize_directional_stop_loss_price(symbol, side, stop_loss)
            if stop_loss is not None
            else None
        )
        response = self._client.place_order(
            symbol=symbol,
            side="BUY" if side == PositionSide.LONG else "SELL",
            qty=normalized_qty,
            order_type=OrderType.MARKET.value,
            trade_side="OPEN",
            reduce_only=False,
            tp_price=take_profit,
            sl_price=normalized_stop_loss,
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
            quantity=normalized_qty,
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
        client_id: Optional[str] = None,
    ) -> OrderResult:
        # self.set_leverage(symbol, leverage)
        normalized_qty = self._normalize_quantity(symbol, quantity)
        normalized_price = self._normalize_limit_price(symbol, side, price)
        normalized_stop_loss = (
            self._normalize_directional_stop_loss_price(symbol, side, stop_loss)
            if stop_loss is not None
            else None
        )
        response = self._client.place_order(
            symbol=symbol,
            side="BUY" if side == PositionSide.LONG else "SELL",
            qty=normalized_qty,
            order_type=OrderType.LIMIT.value,
            price=normalized_price,
            effect="GTC",
            # EMA+AVWAP validates ONE_WAY mode before running.  In that mode
            # Bitunix expects normal buy/sell semantics, not hedge tradeSide.
            trade_side=None,
            reduce_only=False,
            tp_price=take_profit,
            sl_price=normalized_stop_loss,
            sl_stop_type="MARK_PRICE" if normalized_stop_loss is not None else None,
            sl_order_type="MARKET" if normalized_stop_loss is not None else None,
            client_id=client_id,
        )
        if not response or not response.get("orderId"):
            raise RuntimeError(
                "Bitunix open limit position failed: no order id returned"
            )
        order_id = str(response.get("orderId") or "")
        return OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            price=normalized_price,
            quantity=normalized_qty,
            status="NEW",
            timestamp=datetime.now(timezone.utc),
        )

    def close_position(
        self,
        symbol: str,
        side: Optional[PositionSide] = None,
    ) -> OrderResult:
        position_mode = self.get_position_mode()
        positions = self._positions_for_symbol(symbol)
        if position_mode == "HEDGE":
            if side is None:
                raise RuntimeError("Bitunix hedge close requires an explicit side")
            positions = [position for position in positions if position.side == side]
        elif len(positions) == 1 and side is not None and positions[0].side != side:
            raise RuntimeError(
                f"Bitunix position side changed for {symbol}: expected {side.value}, "
                f"found {positions[0].side.value}"
            )
        if not positions:
            raise RuntimeError(
                f"Bitunix close position failed: no open position found for {symbol}"
            )
        if len(positions) != 1:
            raise RuntimeError(
                f"Bitunix close position is ambiguous for {symbol}; "
                "specify/resolve the hedge side first"
            )
        position = positions[0]

        close_qty = position.size
        if position_mode == "HEDGE":
            # Bitunix's hedge API uses the position direction together with
            # tradeSide=CLOSE (BUY closes long, SELL closes short).
            side_to_send = position.side
            trade_side: Optional[str] = "CLOSE"
            position_id: Optional[str] = position.position_id or None
        else:
            # In ONE_WAY mode use ordinary opposite-side reduce-only semantics.
            side_to_send = (
                PositionSide.SHORT
                if position.side == PositionSide.LONG
                else PositionSide.LONG
            )
            trade_side = None
            position_id = None

        response = self._client.place_order(
            symbol=symbol,
            side="BUY" if side_to_send == PositionSide.LONG else "SELL",
            qty=close_qty,
            order_type=OrderType.MARKET.value,
            trade_side=trade_side,
            position_id=position_id,
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

    def get_order_status(
        self,
        *,
        symbol: str,
        order_id: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the venue's authoritative state for one order."""
        del symbol  # Bitunix's detail endpoint is keyed by order/client id.
        payload = self._client.get_order_detail(
            order_id=order_id,
            client_id=client_id,
        )
        return payload or None

    @staticmethod
    def _is_at_least_as_protective(
        side: PositionSide, actual_stop: float, required_stop: float
    ) -> bool:
        if side == PositionSide.LONG:
            return actual_stop >= required_stop
        return actual_stop <= required_stop

    def _get_position_stop_loss(self, position: Position) -> Optional[float]:
        if not position.position_id:
            return None
        orders = self._client.get_pending_tpsl_orders(
            symbol=position.symbol,
            position_id=position.position_id,
            limit=100,
        )
        candidates: List[float] = []
        for order in orders:
            if str(order.get("positionId") or "") != position.position_id:
                continue
            try:
                stop = float(order.get("slPrice"))
            except (TypeError, ValueError):
                continue
            if stop > 0:
                candidates.append(stop)
        if not candidates:
            return None
        return max(candidates) if position.side == PositionSide.LONG else min(candidates)

    def ensure_position_stop_loss(
        self, position: Position, stop_price: float
    ) -> Optional[float]:
        """Install and read back a native stop; never report an assumed stop."""
        if not position.position_id:
            self._log.warning(
                "Bitunix: cannot verify protective stop for %s without position_id",
                position.symbol,
            )
            return None

        normalized_stop = self._normalize_stop_loss_price(position, stop_price)
        existing = self._get_position_stop_loss(position)
        if existing is not None and self._is_at_least_as_protective(
            position.side, existing, normalized_stop
        ):
            return existing

        if existing is None:
            result = self._client.place_position_tpsl_order(
                symbol=position.symbol,
                position_id=position.position_id,
                sl_price=normalized_stop,
                sl_stop_type="MARK_PRICE",
            )
        else:
            result = self._client.modify_position_tpsl_order(
                symbol=position.symbol,
                position_id=position.position_id,
                sl_price=normalized_stop,
                sl_stop_type="MARK_PRICE",
            )
        if not result:
            return None

        # The acknowledgement is not enough: wait briefly for the order to be
        # visible and verify the exact protection before claiming the fill.
        for attempt in range(3):
            actual = self._get_position_stop_loss(position)
            if actual is not None and self._is_at_least_as_protective(
                position.side, actual, normalized_stop
            ):
                return actual
            if attempt < 2:
                time.sleep(0.2)
        return None

    def cancel_all_orders(self, symbol: str) -> None:
        try:
            self._client.cancel_all_orders(symbol=symbol)
        except Exception as exc:
            raise RuntimeError(f"Failed to cancel all orders for {symbol}") from exc

    def close_all_position(self, symbol: Optional[str] = None) -> None:
        """Close all positions, optionally scoped to a symbol."""
        if not self._client.close_all_position(symbol=symbol):
            raise RuntimeError(f"Failed to close all positions (symbol={symbol})")

    def flash_close_position(self, position_id: str) -> bool:
        """Close an open position immediately by position id."""
        result = self._client.flash_close_position(position_id=position_id)
        return str(result.get("positionId", "")).strip() != ""

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
        """Fetch historical trade orders."""
        return self._client.get_history_orders(
            symbol=symbol,
            order_id=order_id,
            client_id=client_id,
            status=status,
            order_type=order_type,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            skip=skip,
            limit=limit,
        )

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
        """Fetch historical trade fills."""
        return self._client.get_history_trades(
            symbol=symbol,
            order_id=order_id,
            position_id=position_id,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            skip=skip,
            limit=limit,
        )

    def get_order_detail(
        self,
        order_id: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch a single order detail by order id or client id."""
        return self._client.get_order_detail(order_id=order_id, client_id=client_id)

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
        """Fetch pending trade orders."""
        return self._client.get_pending_orders(
            symbol=symbol,
            order_id=order_id,
            client_id=client_id,
            status=status,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            skip=skip,
            limit=limit,
        )

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
        """Modify an existing pending order."""
        return self._client.modify_order(
            qty=qty,
            price=price,
            order_id=order_id,
            client_id=client_id,
            tp_price=tp_price,
            tp_stop_type=tp_stop_type,
            tp_order_type=tp_order_type,
            tp_order_price=tp_order_price,
            sl_price=sl_price,
            sl_stop_type=sl_stop_type,
            sl_order_type=sl_order_type,
            sl_order_price=sl_order_price,
        )

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[List]:
        def row_time(row: Dict[str, Any]) -> int:
            try:
                return int(row.get("time", 0) or 0)
            except (AttributeError, TypeError, ValueError):
                return 0

        requested_limit = max(int(limit), 1)
        remaining = requested_limit
        page_end_time = end_time
        rows_by_time: Dict[int, Dict[str, Any]] = {}

        # Bitunix limits each REST request to 200 klines.  Page backwards so
        # strategy warm-up can safely request more history than one response.
        while remaining > 0:
            page_limit = min(remaining, self._KLINE_PAGE_SIZE)
            self._wait_for_kline_request_slot()
            page = self._client.get_kline_history(
                symbol=symbol,
                interval=interval,
                limit=page_limit,
                start_time=start_time,
                end_time=page_end_time,
                kline_type="LAST_PRICE",
            )
            if not page:
                break

            page_times = {row_time(row) for row in page if row_time(row) > 0}
            new_times = page_times.difference(rows_by_time)
            for row in page:
                timestamp = row_time(row)
                if timestamp > 0:
                    rows_by_time[timestamp] = row
            remaining = requested_limit - len(rows_by_time)
            if remaining <= 0 or not new_times:
                break

            earliest = min(page_times, default=0)
            if earliest <= 0 or (start_time is not None and earliest <= start_time):
                break
            next_page_end = earliest - 1
            if page_end_time is not None and next_page_end >= page_end_time:
                break
            page_end_time = next_page_end

        rows = list(rows_by_time.values())
        if not rows:
            return []

        rows = sorted(rows, key=row_time)
        if limit > 0 and len(rows) > limit:
            rows = rows[-limit:]

        interval_ms = interval_to_milliseconds(interval)
        out: List[List] = []
        for row in rows:
            try:
                open_time = int(row.get("time", 0) or 0)
            except (TypeError, ValueError):
                continue
            close_time = (
                open_time + interval_ms - 1
                if interval_ms is not None and open_time > 0
                else open_time
            )
            out.append(
                [
                    open_time,
                    str(row.get("open", "")),
                    str(row.get("high", "")),
                    str(row.get("low", "")),
                    str(row.get("close", "")),
                    str(row.get("baseVol", "0")),
                    close_time,
                    str(row.get("quoteVol", "0")),
                    0,
                    "0",
                    "0",
                    "0",
                ]
            )
        return out

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

    def place_stop_loss_order(
        self, position: Position, stop_price: float
    ) -> Optional[str]:
        """Place an order-level TP/SL for the full position size and return its id."""
        if not position.position_id:
            self._log.warning(
                "Bitunix: cannot place stop loss for %s without position_id",
                position.symbol,
            )
            return None

        try:
            normalized_sl = self._normalize_stop_loss_price(position, stop_price)
            sl_qty = abs(float(position.size))
            result = self._client.place_tpsl_order(
                symbol=position.symbol,
                position_id=position.position_id,
                sl_price=normalized_sl,
                sl_stop_type="MARK_PRICE",
                sl_order_type="LIMIT",
                sl_order_price=normalized_sl,
                sl_qty=sl_qty,
            )
            order_id = str(result.get("orderId") or "").strip() if result else ""
            if not order_id:
                self._log.warning(
                    "Bitunix: stop loss placement returned empty response for %s",
                    position.symbol,
                )
                return None
            self._log.info(
                "Bitunix: placed stop loss for %s (position %s) to %.6f order=%s",
                position.symbol,
                position.position_id,
                stop_price,
                order_id,
            )
            return order_id
        except Exception as exc:
            self._log.warning(
                "Bitunix: failed to place stop loss for %s: %s", position.symbol, exc
            )
            return None

    def update_stop_loss_order(
        self, position: Position, order_id: str, stop_price: float
    ) -> bool:
        """Update an existing order-level TP/SL by order id."""
        if not order_id:
            self._log.warning("Bitunix: cannot update stop loss without order_id")
            return False

        try:
            normalized_sl = self._normalize_stop_loss_price(position, stop_price)
            sl_qty = abs(float(position.size))
            result = self._client.modify_tpsl_order(
                order_id=order_id,
                sl_price=normalized_sl,
                sl_stop_type="MARK_PRICE",
                sl_order_type="LIMIT",
                sl_order_price=normalized_sl,
                sl_qty=sl_qty,
            )
            if not result:
                self._log.warning(
                    "Bitunix: stop loss update returned empty response for %s",
                    position.symbol,
                )
                return False
            self._log.info(
                "Bitunix: updated stop loss for %s (order %s) to %.6f",
                position.symbol,
                order_id,
                stop_price,
            )
            return True
        except Exception as exc:
            self._log.warning(
                "Bitunix: failed to update stop loss for %s: %s", position.symbol, exc
            )
            return False

    def update_position_stop_loss(self, position: Position, stop_price: float) -> bool:
        """Update and read back a position-level TP/SL using positionId."""
        try:
            actual = self.ensure_position_stop_loss(position, stop_price)
            if actual is None:
                self._log.warning(
                    "Bitunix: could not confirm position stop loss for %s",
                    position.symbol,
                )
                return False
            self._log.info(
                "Bitunix: confirmed position stop loss for %s (position %s) at %.6f",
                position.symbol,
                position.position_id,
                actual,
            )
            return True
        except Exception as exc:
            self._log.warning(
                "Bitunix: failed to update position stop loss for %s: %s",
                position.symbol,
                exc,
            )
            return False

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel a regular trade order by order id."""
        try:
            normalized_order_id = str(order_id).strip()
            if not normalized_order_id:
                self._log.warning(
                    "Bitunix: cancel_order rejected: order_id is required"
                )
                return False

            result = self._client.cancel_orders(
                symbol=symbol, order_list=[{"orderId": normalized_order_id}]
            )
            if not result:
                self._log.warning(
                    "Bitunix: cancel_order failed symbol=%s order_id=%s",
                    symbol,
                    normalized_order_id,
                )
                return False

            success_list = result.get("successList")
            if isinstance(success_list, list):
                for item in success_list:
                    if not isinstance(item, dict):
                        continue
                    response_order_id = str(
                        item.get("orderId") or item.get("id") or ""
                    ).strip()
                    if response_order_id == normalized_order_id:
                        return True

            failure_list = result.get("failureList")
            if isinstance(failure_list, list):
                for item in failure_list:
                    if not isinstance(item, dict):
                        continue
                    response_order_id = str(
                        item.get("orderId") or item.get("id") or ""
                    ).strip()
                    if response_order_id == normalized_order_id:
                        return False

            # Acknowledging the request is not proof that this particular order
            # was cancelled.  The caller must retain and reconcile it rather than
            # lose track of a live GTC order.
            self._log.warning(
                "Bitunix: cancel_order response lacked a definitive result "
                "symbol=%s order_id=%s",
                symbol,
                normalized_order_id,
            )
            return False
        except Exception as exc:
            self._log.warning(
                "Bitunix: cancel_order error symbol=%s order_id=%s error=%s",
                symbol,
                order_id,
                exc,
            )
            return False

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return currently open (pending) orders, optionally filtered by symbol."""
        result = self._client.get_pending_orders(symbol=symbol)
        order_list = result.get("orderList")
        if isinstance(order_list, list):
            return [o for o in order_list if isinstance(o, dict)]
        return []
