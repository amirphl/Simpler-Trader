"""Live trading module for automated strategy execution."""

from .exchange import (
    Exchange,
    ExchangeConfig,
    Position,
    PositionSide,
    PositionStatus,
    OrderType,
    MarginMode,
)
from .heiken_ashi_strategy import HeikenAshiLiveStrategy, LiveTradingStrategy
from .models import (
    LiveTradingConfig,
    PendingEntryRecord,
    PinBarMagicSnapshot,
    SymbolInfo,
    TradingSignal,
    TradingState,
    PositionRecord,
)
from .pinbar_magic_strategy_v3 import PinBarMagicLiveStrategyV3
from .strategy_shared import BaseLiveTradingStrategy

__all__ = [
    "Exchange",
    "ExchangeConfig",
    "Position",
    "PositionSide",
    "PositionStatus",
    "OrderType",
    "MarginMode",
    "BaseLiveTradingStrategy",
    "HeikenAshiLiveStrategy",
    "LiveTradingStrategy",
    "PinBarMagicLiveStrategyV3",
    "LiveTradingConfig",
    "SymbolInfo",
    "TradingSignal",
    "TradingState",
    "PositionRecord",
    "PendingEntryRecord",
    "PinBarMagicSnapshot",
]
