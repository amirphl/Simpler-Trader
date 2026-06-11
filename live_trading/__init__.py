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
from .ema_avwap_pullback_strategy import (
    EmaAvwapPullbackLiveConfig,
    EmaAvwapPullbackLiveCoordinator,
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
    "EmaAvwapPullbackLiveConfig",
    "EmaAvwapPullbackLiveCoordinator",
    "LiveTradingConfig",
    "SymbolInfo",
    "TradingSignal",
    "TradingState",
    "PositionRecord",
    "PendingEntryRecord",
    "PinBarMagicSnapshot",
]
