"""Bitunix exchange adapter implementing the shared Exchange interface."""

from __future__ import annotations

import logging
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ...exchange import (
    Exchange,
    ExchangeConfig,
    MarginMode,
    OrderResult,
    OrderType,
    Position,
    PositionSide,
)
from .client import BitunixClient
from .utils import infer_margin_coin_from_symbol, interval_to_milliseconds

class BitunixExchange(Exchange):
    """Bitunix futures exchange adapter."""

    def __init__(self, config: ExchangeConfig, logger: Optional[logging.Logger] = None) -> None:
        self._config = config
        self._log = logger or logging.getLogger(__name__)
        self._client = BitunixClient(config, self._log)
        self._default_margin_coin = "USDT"
        self._pair_meta_cache: Dict[str, Dict[str, Any]] = {}

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

    def _normalize_limit_price(self, symbol: str, side: PositionSide, price: float) -> float:
        px = float(price)
        if px <= 0:
            raise RuntimeError(f"Invalid order price for {symbol}: {price}")

        meta = self._get_symbol_meta(symbol)
        quote_precision = int(meta.get("quotePrecision", 8) or 8)
        price_protect_scope = float(meta.get("priceProtectScope", 0) or 0)
        mark_price = self.fetch_price(symbol)
        adjusted = px

        # Keep limit price within exchange price-protection band to avoid 30014/30015.
        if mark_price is not None and mark_price > 0 and price_protect_scope > 0:
            if side == PositionSide.LONG:
                max_buy = mark_price * (1.0 + price_protect_scope)
                adjusted = min(adjusted, max_buy)
            else:
                min_sell = mark_price * (1.0 - price_protect_scope)
                adjusted = max(adjusted, min_sell)

        rounding = ROUND_DOWN if side == PositionSide.LONG else ROUND_UP
        normalized_price = self._quantize(adjusted, quote_precision, rounding_mode=rounding)

        if normalized_price <= 0:
            raise RuntimeError(
                f"Normalized order price is invalid for {symbol}: raw={price} normalized={normalized_price}"
            )
        if abs(normalized_price - px) > 0:
            self._log.info(
                "Bitunix: normalized %s limit price for %s from %.10f to %.10f",
                side.value,
                symbol,
                px,
                normalized_price,
            )
        return normalized_price

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

    def get_trading_pairs(self, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Fetch trading pair metadata."""
        pairs = self._client.get_trading_pairs(symbols=symbols)
        return list(pairs.values())

    def get_depth(self, symbol: str, limit: Optional[str | int] = None) -> Dict[str, Any]:
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
        response = self._client.place_order(
            symbol=symbol,
            side="BUY" if side == PositionSide.LONG else "SELL",
            qty=normalized_qty,
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
    ) -> OrderResult:
        # self.set_leverage(symbol, leverage)
        normalized_qty = self._normalize_quantity(symbol, quantity)
        normalized_price = self._normalize_limit_price(symbol, side, price)
        response = self._client.place_order(
            symbol=symbol,
            side="BUY" if side == PositionSide.LONG else "SELL",
            qty=normalized_qty,
            order_type=OrderType.LIMIT.value,
            price=normalized_price,
            effect="GTC",
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
        result = self._client.cancel_all_orders(symbol=symbol)
        if not result:
            raise RuntimeError(f"Failed to cancel all orders for {symbol}")

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
        rows = self._client.get_kline_history(
            symbol=symbol,
            interval=interval,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            kline_type="LAST_PRICE",
        )
        if not rows:
            return []

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

    # ------------------------------------------------------------------
    # PinBarMagic placeholders (to be replaced with official Bitunix docs)
    # ------------------------------------------------------------------
    def place_stop_entry_order(
        self,
        symbol: str,
        side: PositionSide,
        quantity: float,
        stop_price: float,
        leverage: int,
        margin_mode: MarginMode,
        stop_loss: Optional[float] = None,
    ) -> OrderResult:
        """Placeholder stop-entry implementation.

        NOTE: Bitunix trigger-order API wiring is pending. This fallback submits a
        limit order at the stop trigger price so higher-level logic remains runnable.
        """
        self._log.warning(
            "Bitunix: place_stop_entry_order is placeholder; using LIMIT fallback "
            "(symbol=%s side=%s stop_price=%.6f)",
            symbol,
            side.value,
            stop_price,
        )
        return self.open_limit_position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=stop_price,
            leverage=leverage,
            margin_mode=margin_mode,
            take_profit=None,
            stop_loss=stop_loss,
        )

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel a regular trade order by order id."""
        try:
            normalized_order_id = str(order_id).strip()
            if not normalized_order_id:
                self._log.warning("Bitunix: cancel_order rejected: order_id is required")
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

            # If API accepted request but omitted per-order details, treat as submitted.
            return True
        except Exception as exc:
            self._log.warning(
                "Bitunix: cancel_order error symbol=%s order_id=%s error=%s",
                symbol,
                order_id,
                exc,
            )
            return False

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Placeholder for open-order retrieval."""
        self._log.warning(
            "Bitunix: get_open_orders placeholder not implemented (symbol=%s)",
            symbol,
        )
        return []
