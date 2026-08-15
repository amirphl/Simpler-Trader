"""Symbol-aware exchange routing for strategy-specific venue exceptions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..exchange import (
    Exchange,
    ExchangeConfig,
    MarginMode,
    OrderResult,
    Position,
    PositionSide,
)


class ZecWeexRoutedExchange(Exchange):
    """Route ZECUSDT to Weex and every other symbol to the primary exchange."""

    WEEX_SYMBOL = "ZECUSDT"

    def __init__(self, primary: Exchange, weex: Exchange) -> None:
        self._primary = primary
        self._weex = weex
        # The shared interface exposes config in several callers/tests. Primary
        # remains Bitunix for its non-ZEC symbols.
        self._config: ExchangeConfig = getattr(primary, "_config")

    @classmethod
    def _is_weex_symbol(cls, symbol: str) -> bool:
        return "".join(char for char in str(symbol).upper() if char.isalnum()) == cls.WEEX_SYMBOL

    def _exchange_for(self, symbol: str) -> Exchange:
        return self._weex if self._is_weex_symbol(symbol) else self._primary

    def get_account_balance(self) -> float:
        return self._primary.get_account_balance()

    def get_account_balance_for_symbol(self, symbol: str) -> float:
        exchange = self._exchange_for(symbol)
        getter = getattr(exchange, "get_account_balance_for_symbol", None)
        return float(getter(symbol)) if callable(getter) else exchange.get_account_balance()

    def fetch_price(self, symbol: str) -> Optional[float]:
        return self._exchange_for(symbol).fetch_price(symbol)

    def get_24h_tickers(self) -> List[Dict[str, Any]]:
        primary = [
            row
            for row in self._primary.get_24h_tickers()
            if not self._is_weex_symbol(str(row.get("symbol", "")))
        ]
        weex = [
            row
            for row in self._weex.get_24h_tickers()
            if self._is_weex_symbol(str(row.get("symbol", "")))
        ]
        return primary + weex

    def get_current_positions(self) -> List[Position]:
        primary = [
            position
            for position in self._primary.get_current_positions()
            if not self._is_weex_symbol(position.symbol)
        ]
        weex = [
            position
            for position in self._weex.get_current_positions()
            if self._is_weex_symbol(position.symbol)
        ]
        return primary + weex

    def get_position(self, symbol: str) -> Optional[Position]:
        return self._exchange_for(symbol).get_position(symbol)

    def set_leverage(self, symbol: str, leverage: int) -> None:
        self._exchange_for(symbol).set_leverage(symbol, leverage)

    def set_margin_mode(self, symbol: str, margin_mode: MarginMode) -> None:
        self._exchange_for(symbol).set_margin_mode(symbol, margin_mode)

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
        return self._exchange_for(symbol).open_market_position(
            symbol, side, quantity, leverage, margin_mode, take_profit, stop_loss
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
        return self._exchange_for(symbol).open_limit_position(
            symbol,
            side,
            quantity,
            price,
            leverage,
            margin_mode,
            take_profit,
            stop_loss,
            client_id,
        )

    def close_position(
        self, symbol: str, side: Optional[PositionSide] = None
    ) -> OrderResult:
        return self._exchange_for(symbol).close_position(symbol, side)

    def cancel_all_orders(self, symbol: str) -> None:
        self._exchange_for(symbol).cancel_all_orders(symbol)

    def place_stop_loss_order(self, position: Position, stop_price: float) -> Optional[str]:
        return self._exchange_for(position.symbol).place_stop_loss_order(position, stop_price)

    def update_stop_loss_order(
        self, position: Position, order_id: str, stop_price: float
    ) -> bool:
        return self._exchange_for(position.symbol).update_stop_loss_order(
            position, order_id, stop_price
        )

    def update_position_stop_loss(self, position: Position, stop_price: float) -> bool:
        return self._exchange_for(position.symbol).update_position_stop_loss(position, stop_price)

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        return self._exchange_for(symbol).cancel_order(symbol, order_id)

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[List]:
        return self._exchange_for(symbol).get_klines(
            symbol, interval, limit, start_time, end_time
        )

    def test_connection(self) -> bool:
        return self._primary.test_connection() and self._weex.test_connection()

    def close(self) -> None:
        self._primary.close()
        self._weex.close()

    # Strategy extensions -------------------------------------------------
    def validate_order_quantity(self, symbol: str, quantity: float) -> float:
        validator = getattr(self._exchange_for(symbol), "validate_order_quantity", None)
        return float(validator(symbol, quantity)) if callable(validator) else quantity

    def validate_ema_avwap_execution(self) -> None:
        for exchange in (self._primary, self._weex):
            validator = getattr(exchange, "validate_ema_avwap_execution", None)
            if callable(validator):
                validator()

    def ensure_position_stop_loss(
        self, position: Position, stop_price: float
    ) -> Optional[float]:
        confirmer = getattr(self._exchange_for(position.symbol), "ensure_position_stop_loss", None)
        return confirmer(position, stop_price) if callable(confirmer) else None

    def get_order_status(
        self,
        *,
        symbol: str,
        order_id: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        getter = getattr(self._exchange_for(symbol), "get_order_status", None)
        if not callable(getter):
            return None
        return getter(symbol=symbol, order_id=order_id, client_id=client_id)

    def get_trading_pairs(self, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if symbols:
            rows: List[Dict[str, Any]] = []
            for symbol in symbols:
                getter = getattr(self._exchange_for(symbol), "get_trading_pairs")
                rows.extend(getter([symbol]))
            return rows
        return self._primary.get_trading_pairs() + self._weex.get_trading_pairs(
            [self.WEEX_SYMBOL]
        )

    def get_depth(self, symbol: str, limit: Optional[str | int] = None) -> Dict[str, Any]:
        return self._exchange_for(symbol).get_depth(symbol, limit)

    def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        return self._exchange_for(symbol).get_funding_rate(symbol)
