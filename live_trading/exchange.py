"""Exchange interface for live trading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


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
    def get_24h_tickers(self) -> List[Dict[str, any]]:
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
        """Open a market position with optional TP/SL.
        
        Args:
            symbol: Trading symbol
            side: LONG or SHORT
            quantity: Position size in quote currency (USDT)
            leverage: Leverage multiplier
            margin_mode: ISOLATED or CROSS
            take_profit: Optional take profit price
            stop_loss: Optional stop loss price
            
        Returns:
            OrderResult with execution details
            
        Raises:
            RuntimeError: If order placement fails
        """
        pass

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
