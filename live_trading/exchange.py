"""Exchange interface for live trading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class PositionSide(Enum):
    """Position direction."""

    LONG = "LONG"
    SHORT = "SHORT"


class PositionStatus(Enum):
    """Position status."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"


class OrderType(Enum):
    """Order type."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"


class MarginMode(Enum):
    """Margin mode for positions."""

    ISOLATED = "ISOLATED"
    CROSS = "CROSS"


@dataclass(frozen=True)
class ExchangeConfig:
    """Configuration for exchange client."""

    api_key: str
    api_secret: str
    testnet: bool = True
    timeout: float = 10.0
    max_retries: int = 3
    proxies: Optional[Dict[str, str]] = None
    passphrase: Optional[str] = None
    locale: str = "en-US"
    base_url: Optional[str] = None


@dataclass(frozen=True)
class Position:
    """Represents an open position on the exchange."""

    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    leverage: float
    margin_mode: MarginMode
    unrealized_pnl: float
    liquidation_price: Optional[float] = None
    position_id: Optional[str] = None


@dataclass(frozen=True)
class OrderResult:
    """Result of placing an order."""

    order_id: str
    symbol: str
    side: PositionSide
    order_type: OrderType
    price: float
    quantity: float
    status: str
    timestamp: datetime


class Exchange(ABC):
    """Abstract base class for exchange clients.

    This interface will be implemented for specific exchanges (Binance, etc.)
    based on their API documentation.
    """

    @abstractmethod
    def __init__(self, config: ExchangeConfig) -> None:
        """Initialize the exchange client with configuration."""
        pass

    @abstractmethod
    def get_account_balance(self) -> float:
        """Get available account balance in USDT.

        Returns:
            Available balance in USDT

        Raises:
            RuntimeError: If API call fails
        """
        pass

    @abstractmethod
    def fetch_price(self, symbol: str) -> Optional[float]:
        """Fetch the latest price for a given symbol.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")

        Returns:
            Latest price as a float

        Raises:
            RuntimeError: If API call fails
        """
        pass

    @abstractmethod
    def get_24h_tickers(self) -> List[Dict[str, Any]]:
        """Get 24-hour ticker data for all symbols.

        Returns:
            List of ticker dictionaries containing:
                - symbol: str
                - priceChange: float
                - priceChangePercent: float
                - lastPrice: float
                - volume: float (base asset volume)
                - quoteVolume: float (quote asset volume)

        Raises:
            RuntimeError: If API call fails
        """
        pass

    @abstractmethod
    def get_current_positions(self) -> List[Position]:
        """Get all currently open positions.

        Returns:
            List of Position objects

        Raises:
            RuntimeError: If API call fails
        """
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")

        Returns:
            Position object if position exists, None otherwise

        Raises:
            RuntimeError: If API call fails
        """
        pass

    @abstractmethod
    def set_leverage(self, symbol: str, leverage: int) -> None:
        """Set leverage for a symbol.

        Args:
            symbol: Trading symbol
            leverage: Leverage multiplier (e.g., 10 for 10x)

        Raises:
            RuntimeError: If API call fails
        """
        pass

    @abstractmethod
    def set_margin_mode(self, symbol: str, margin_mode: MarginMode) -> None:
        """Set margin mode for a symbol.

        Args:
            symbol: Trading symbol
            margin_mode: ISOLATED or CROSS

        Raises:
            RuntimeError: If API call fails
        """
        pass

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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
        """Open a limit position with optional TP/SL."""
        raise NotImplementedError

    @abstractmethod
    def close_position(
        self,
        symbol: str,
        side: Optional[PositionSide] = None,
    ) -> OrderResult:
        """Close an open position.

        Args:
            symbol: Trading symbol
            side: Position side to close (if None, closes all positions for symbol)

        Returns:
            OrderResult with execution details

        Raises:
            RuntimeError: If position closing fails
        """
        pass

    @abstractmethod
    def cancel_all_orders(self, symbol: str) -> None:
        """Cancel all open orders for a symbol.

        Args:
            symbol: Trading symbol

        Raises:
            RuntimeError: If API call fails
        """
        pass

    @abstractmethod
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
        """Placeholder stop-entry implementation."""
        pass

    @abstractmethod
    def update_stop_loss(self, position: Position, stop_price: float) -> bool:
        pass

    @abstractmethod
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        pass

    @abstractmethod
    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[List]:
        """Get kline/candlestick data.

        Args:
            symbol: Trading symbol
            interval: Kline interval (e.g., "1m", "5m", "15m", "1h")
            limit: Number of klines to return (max 1500)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds

        Returns:
            List of klines in Binance format:
            [
                [
                    open_time,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    close_time,
                    quote_asset_volume,
                    ...
                ]
            ]

        Raises:
            RuntimeError: If API call fails
        """
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test connectivity to the exchange.

        Returns:
            True if connection successful

        Raises:
            RuntimeError: If connection test fails
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the exchange client and cleanup resources."""
        pass

    # Optional exchange extensions -------------------------------------
    # Default implementations keep the base interface compatible with
    # exchanges that do not expose these advanced endpoints.

    def get_trading_pairs(
        self, symbols: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_trading_pairs"
        )

    def get_depth(
        self, symbol: str, limit: Optional[str | int] = None
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_depth"
        )

    def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_funding_rate"
        )

    def get_position_tiers(self, symbol: str) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_position_tiers"
        )

    def place_position_tpsl_order(
        self,
        symbol: str,
        position_id: str,
        tp_price: Optional[float] = None,
        tp_stop_type: Optional[str] = None,
        sl_price: Optional[float] = None,
        sl_stop_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement place_position_tpsl_order"
        )

    def close_all_position(self, symbol: Optional[str] = None) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement close_all_position"
        )

    def flash_close_position(self, position_id: str) -> bool:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement flash_close_position"
        )

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
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_history_orders"
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
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_history_trades"
        )

    def get_order_detail(
        self,
        order_id: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_order_detail"
        )

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
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_pending_orders"
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
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement modify_order"
        )

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_open_orders"
        )
