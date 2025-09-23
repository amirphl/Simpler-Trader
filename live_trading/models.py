"""Data models for live trading."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

from candle_downloader.models import Candle
from .exchange import MarginMode, PositionSide

ALLOWED_TIMEFRAMES = (
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "12h",
    "1d",
)
ALLOWED_STRATEGIES = ("heiken_ashi", "pinbar_magic_v2")


class CandlePattern(Enum):
    """Candle pattern type."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


@dataclass(frozen=True)
class LiveTradingConfig:
    """Configuration for live trading strategy."""

    # Exchange settings
    exchange_name: str  # e.g., "binance", "bybit"
    api_key: str
    api_secret: str
    testnet: bool = True
    strategy_name: Literal["heiken_ashi", "pinbar_magic_v2"] = "pinbar_magic_v2"

    # Trading parameters
    timeframe: str = "1h"  # e.g., "15m", "1h"
    top_m_symbols: int = 100  # Number of symbols to scan
    top_n_signals: int = 5  # Number of symbols to trade (movers)
    price_change_threshold_pct: float = 2.0  # Minimum % change
    pinbar_symbols: Tuple[str, ...] = field(default_factory=tuple)

    # Signal parameters
    heiken_ashi_candles_before: int = 3  # candles before reversal
    leverage: int = 10  # leverage multiplier
    take_profit_pct: float = 1.0  # take profit percentage
    equity_risk_pct: float = 3.0
    atr_multiple: float = 0.5
    trail_points: float = 1.0
    trail_offset: float = 1.0
    slow_sma_period: int = 50
    medium_ema_period: int = 18
    fast_ema_period: int = 6
    atr_period: int = 14
    entry_cancel_bars: int = 3
    entry_activation_mode: Literal["next_bar", "same_bar"] = "next_bar"
    trailing_tick_timeframe: str = "15m"
    use_trailing_tick_emulation: bool = False
    use_stop_fill_open_gap: bool = True
    enable_friday_close: bool = True
    friday_close_hour_utc: int = 16
    enable_ema_cross_close: bool = True
    risk_equity_include_unrealized: bool = True
    risk_equity_mark_source: Literal["close", "open", "hl2", "ohlc4"] = "close"

    # Position management
    margin_mode: MarginMode = MarginMode.ISOLATED
    disable_symbol_hours: float = 24  # hours to disable after trade (0 = no disable)
    position_size_usdt: float = 100.0  # Position size in USDT

    # Risk management
    max_concurrent_positions: int = 5
    max_position_size_pct: float = 10.0  # Max position size as % of balance

    # Data persistence
    state_file: Path = Path("./data/live_trading_state.json")
    positions_db: Path = Path("./data/live_trading_positions.db")
    klines_db: Path = Path("./configs/live_trading.env")
    log_file: Path = Path("./logs/live_trading.log")

    # Scheduling
    candle_ready_delay_seconds: int = (
        30  # Delay after timeframe close to ensure candle availability
    )
    execution_interval_minutes: int = 5  # How often to run the strategy loop

    # Notifications
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_proxy: Optional[str] = None
    telegram_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.timeframe not in ALLOWED_TIMEFRAMES:
            raise ValueError(f"Invalid timeframe: {self.timeframe}")
        if self.trailing_tick_timeframe not in ALLOWED_TIMEFRAMES:
            raise ValueError(f"Invalid trailing_tick_timeframe: {self.trailing_tick_timeframe}")
        if self.strategy_name not in ALLOWED_STRATEGIES:
            raise ValueError("strategy_name must be one of: heiken_ashi, pinbar_magic_v2")

        # Scanner parameters are only required by Heiken Ashi flow.
        if self.strategy_name == "heiken_ashi":
            if self.top_m_symbols <= 0:
                raise ValueError("top_m_symbols must be positive")
            if self.top_n_signals <= 0:
                raise ValueError("top_n_signals must be positive")
            if self.top_n_signals > self.top_m_symbols:
                raise ValueError("top_n_signals cannot exceed top_m_symbols")
            if self.price_change_threshold_pct <= 0:
                raise ValueError("price_change_threshold_pct must be positive")

        if self.heiken_ashi_candles_before <= 0:
            raise ValueError("heiken_ashi_candles_before must be positive")
        if self.leverage <= 0:
            raise ValueError("leverage must be positive")
        if self.take_profit_pct <= 0:
            raise ValueError("take_profit_pct must be positive")
        if self.equity_risk_pct <= 0:
            raise ValueError("equity_risk_pct must be positive")
        if self.atr_multiple <= 0:
            raise ValueError("atr_multiple must be positive")
        if self.trail_points <= 0:
            raise ValueError("trail_points must be positive")
        if self.trail_offset < 0:
            raise ValueError("trail_offset must be non-negative")
        if min(
            self.slow_sma_period,
            self.medium_ema_period,
            self.fast_ema_period,
            self.atr_period,
            self.entry_cancel_bars,
        ) <= 0:
            raise ValueError("period-based values must be positive")
        if self.entry_activation_mode not in {"next_bar", "same_bar"}:
            raise ValueError("entry_activation_mode must be one of: next_bar, same_bar")
        if self.risk_equity_mark_source not in {"close", "open", "hl2", "ohlc4"}:
            raise ValueError(
                "risk_equity_mark_source must be one of: close, open, hl2, ohlc4"
            )
        if self.strategy_name == "pinbar_magic_v2":
            for symbol in self.pinbar_symbols:
                if not symbol or not str(symbol).strip():
                    raise ValueError("pinbar_symbols must not contain empty values")
        if not (0 <= self.friday_close_hour_utc <= 23):
            raise ValueError("friday_close_hour_utc must be in 0..23")
        if self.disable_symbol_hours < 0:
            raise ValueError("disable_symbol_hours must be non-negative")
        if self.position_size_usdt <= 0:
            raise ValueError("position_size_usdt must be positive")
        if self.max_concurrent_positions <= 0:
            raise ValueError("max_concurrent_positions must be positive")
        if self.max_position_size_pct <= 0 or self.max_position_size_pct > 100:
            raise ValueError("max_position_size_pct must be between 0 and 100")
        if self.candle_ready_delay_seconds < 0:
            raise ValueError("candle_ready_delay_seconds must be non-negative")
        if self.execution_interval_minutes <= 0:
            raise ValueError("execution_interval_minutes must be positive")
        if self.telegram_enabled:
            if not self.telegram_bot_token:
                raise ValueError(
                    "telegram_bot_token is required when telegram_enabled is True"
                )
            if not self.telegram_chat_id:
                raise ValueError(
                    "telegram_chat_id is required when telegram_enabled is True"
                )
            if self.telegram_timeout_seconds <= 0:
                raise ValueError("telegram_timeout_seconds must be positive")


@dataclass
class SymbolInfo:
    """Information about a symbol for scanning."""

    symbol: str
    current_price: float
    price_change_pct: float
    volume: float
    quote_volume: float


@dataclass
class HekenAshiCandle:
    """Heiken Ashi candle."""

    open_time: datetime
    close_time: datetime
    ha_open: float
    ha_high: float
    ha_low: float
    ha_close: float
    # Keep original OHLC for reference
    orig_open: float
    orig_high: float
    orig_low: float
    orig_close: float
    volume: float

    def is_bullish(self) -> bool:
        """Check if candle is bullish (HA close > HA open)."""
        return self.ha_close > self.ha_open

    def is_bearish(self) -> bool:
        """Check if candle is bearish (HA close < HA open)."""
        return self.ha_close < self.ha_open


@dataclass
class TradingSignal:
    """Trading signal generated by the strategy."""

    timestamp: datetime
    symbol: str
    side: PositionSide
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    leverage: int
    margin_mode: MarginMode
    reason: str
    strategy: str = "heiken_ashi"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionRecord:
    """Record of a submitted position."""

    position_id: str
    symbol: str
    side: PositionSide
    entry_time: datetime
    entry_price: float
    quantity: float
    leverage: int
    margin_mode: MarginMode
    take_profit: Optional[float]
    stop_loss: Optional[float]
    risk_amount: Optional[float] = None
    trailing_active: bool = False
    trailing_stop: Optional[float] = None
    extreme_since_activation: Optional[float] = None
    strategy: str = "heiken_ashi"
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    status: str = "OPEN"  # OPEN, CLOSED, ERROR
    notes: str = ""


@dataclass
class TradingState:
    """State tracking for live trading."""

    disabled_symbols: Dict[str, datetime] = field(default_factory=dict)
    active_positions: Dict[str, PositionRecord] = field(default_factory=dict)
    pending_entries: Dict[str, "PendingEntryRecord"] = field(default_factory=dict)
    last_pinbar_signal_times: Dict[str, datetime] = field(default_factory=dict)
    last_execution_time: Optional[datetime] = None
    total_trades: int = 0
    successful_trades: int = 0
    failed_trades: int = 0

    def is_symbol_disabled(
        self, symbol: str, current_time: datetime, disable_hours: float
    ) -> bool:
        """Check if a symbol is currently disabled."""
        if disable_hours == 0:
            return False
        if symbol not in self.disabled_symbols:
            return False
        disabled_until = self._ensure_aware(self.disabled_symbols[symbol])
        now = self._ensure_aware(current_time)
        return now < disabled_until

    def disable_symbol(self, symbol: str, current_time: datetime, hours: float) -> None:
        """Disable a symbol for the specified number of hours."""
        if hours > 0:
            from datetime import timedelta

            now = self._ensure_aware(current_time)
            self.disabled_symbols[symbol] = now + timedelta(hours=hours)

    def cleanup_disabled_symbols(self, current_time: datetime) -> None:
        """Remove expired disabled symbols."""
        now = self._ensure_aware(current_time)
        expired = [
            symbol
            for symbol, until in self.disabled_symbols.items()
            if now >= self._ensure_aware(until)
        ]
        for symbol in expired:
            del self.disabled_symbols[symbol]

    @staticmethod
    def _ensure_aware(moment: datetime) -> datetime:
        if moment.tzinfo is None:
            return moment.replace(tzinfo=timezone.utc)
        return moment


@dataclass
class PendingEntryRecord:
    """State for a pending Pine-like stop-entry order."""

    order_key: str
    symbol: str
    side: PositionSide
    entry_price: float
    quantity: float
    leverage: int
    margin_mode: MarginMode
    risk_amount: float
    stop_for_risk: float
    created_time: datetime
    signal_time: datetime
    activate_time: datetime
    order_id: Optional[str] = None
    status: str = "PENDING"  # PENDING, PLACED, FILLED, CANCELLED, ERROR
    notes: str = ""


@dataclass(frozen=True)
class PinBarMagicSnapshot:
    """Computed per-symbol bar snapshot used for live Pin Bar Magic management."""

    symbol: str
    timeframe: str
    timeframe_minutes: int
    bar: Candle
    previous_bar: Candle
    fast: float
    med: float
    slow: float
    atr_prev: float
    fast_prev: float
    med_prev: float
    long_entry: bool
    short_entry: bool
    crossunder: bool
    crossover: bool
    friday_close: bool
