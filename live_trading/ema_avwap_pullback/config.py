"""Configuration for the EMA + AVWAP pullback live strategy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from candle_downloader.binance import interval_to_milliseconds

from ..exchange import MarginMode

Direction = Literal["long", "short"]


class EntryMode(str, Enum):
    CLOSE = "close"
    LIVE = "live"


class ExitMode(str, Enum):
    CLOSE = "close"
    LIVE = "live"


class ExitBand(str, Enum):
    BAND_1 = "band_1"
    BAND_2 = "band_2"

    @property
    def number(self) -> int:
        return 1 if self is ExitBand.BAND_1 else 2

EmaValidationMode = Literal["body", "wick"]
SetupWaitingReplacementMode = Literal["keep_waiting", "replace_waiting"]
# EMA/AVWAP sizing is deliberately a notional-budget calculation.  It is not
# stop-loss risk sizing, even though the retained CLI setting has the historic
# ``risk_amount_per_price`` name.
PositionSizingMode = Literal["risk_amount_per_price"]

@dataclass(frozen=True)
class EmaAvwapPullbackLiveConfig:
    symbols: tuple[str, ...] = ("ETHUSDT",)
    timeframe: str = "1h"
    trailing_tick_timeframe: str = "1m"
    use_trailing_tick_emulation: bool = False
    poll_interval_seconds: float = 5.0
    trailing_check_interval_seconds: float = 5.0
    live_kline_stale_seconds: float = 10.0

    leverage: int = 10
    margin_mode: MarginMode = MarginMode.ISOLATED
    max_concurrent_positions: int = 1
    max_entry_notional_usdt: float = 15.0
    max_position_size_pct: float = 10.0
    position_notional_pct: float = 1.0

    ema_length: int = 55
    consecutive_count: int = 4
    ema_validation_mode: EmaValidationMode = "body"
    setup_waiting_replacement_mode: SetupWaitingReplacementMode = "keep_waiting"
    max_setup_age_bars: int = 3
    max_entry_deviation_pct: float = 1.0
    position_sizing_mode: PositionSizingMode = "risk_amount_per_price"
    entry_mode: EntryMode = EntryMode.LIVE
    exit_mode: ExitMode = ExitMode.LIVE
    exit_band: ExitBand = ExitBand.BAND_1

    avwap_multiplier_1: float = 1.0
    avwap_multiplier_2: float = 2.0
    avwap_multiplier_3: float = 3.0

    rigid_stop_loss_pct: float = 0.0
    trailing_activation_threshold_pct: float = 0.0
    trailing_gap_pct: float = 1.0

    maker_fee_pct: float = 0.0002
    taker_fee_pct: float = 0.0006
    entry_slippage_pct: float = 0.0
    exit_slippage_pct: float = 0.0
    use_gap_cross_detection: bool = True

    entry_cancel_bars: int = 1
    max_history_bars: int = 512
    minimum_balance_usdt: float = 0.0
    candle_ready_delay_seconds: float = 0.0
    execution_interval_minutes: int = 5
    api_retries: int = 3
    api_retry_delay_seconds: float = 1.0
    emergency_close_on_stop_failure: bool = True
    allow_dynamic_stop_widening: bool = True
    min_stop_update_pct: float = 0.0
    disable_symbol_hours: float = 0.0

    state_file: Path = Path("./data/ema_avwap_pullback_live_trading_state.json")
    account_lock_file: Path | None = None
    positions_db: Path = Path("./data/ema_avwap_pullback_live_trading_positions.db")
    klines_db: Path = Path("./configs/live_trading.ema_avwap_pullback.env")
    log_file: Path = Path("./logs/ema_avwap_pullback_live_trading.log")

    def __post_init__(self) -> None:
        symbols = tuple(
            symbol.strip().upper() for symbol in self.symbols if symbol.strip()
        )
        if not symbols:
            raise ValueError("symbols must contain at least one symbol")
        if not self.timeframe.strip():
            raise ValueError("timeframe must not be empty")
        interval_to_milliseconds(self.timeframe)
        interval_to_milliseconds(self.trailing_tick_timeframe)
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if self.trailing_check_interval_seconds <= 0:
            raise ValueError("trailing_check_interval_seconds must be positive")
        if self.live_kline_stale_seconds <= 0:
            raise ValueError("live_kline_stale_seconds must be positive")
        if self.leverage <= 0:
            raise ValueError("leverage must be positive")
        if self.max_concurrent_positions <= 0:
            raise ValueError("max_concurrent_positions must be positive")
        if self.max_entry_notional_usdt <= 0:
            raise ValueError("max_entry_notional_usdt must be positive")
        if not 0 < self.max_position_size_pct <= 100:
            raise ValueError("max_position_size_pct must be in (0, 100]")
        if not 0 < self.position_notional_pct <= 100:
            raise ValueError("position_notional_pct must be in (0, 100]")
        if self.ema_length <= 0:
            raise ValueError("ema_length must be positive")
        if self.consecutive_count <= 0:
            raise ValueError("consecutive_count must be positive")
        if self.max_setup_age_bars <= 0:
            raise ValueError("max_setup_age_bars must be positive")
        if self.max_entry_deviation_pct < 0:
            raise ValueError("max_entry_deviation_pct must be non-negative")
        if self.ema_validation_mode not in {"body", "wick"}:
            raise ValueError("ema_validation_mode must be one of: body, wick")
        if self.setup_waiting_replacement_mode not in {
            "keep_waiting",
            "replace_waiting",
        }:
            raise ValueError(
                "setup_waiting_replacement_mode must be one of: "
                "keep_waiting, replace_waiting"
            )
        if self.position_sizing_mode != "risk_amount_per_price":
            raise ValueError(
                "position_sizing_mode must be risk_amount_per_price; EMA+AVWAP "
                "uses a position-notional budget, not stop-loss risk sizing"
            )
        try:
            entry_mode = EntryMode(self.entry_mode)
        except ValueError as exc:
            allowed = ", ".join(mode.value for mode in EntryMode)
            raise ValueError(f"entry_mode must be one of: {allowed}") from exc
        try:
            exit_mode = ExitMode(self.exit_mode)
        except ValueError as exc:
            allowed = ", ".join(mode.value for mode in ExitMode)
            raise ValueError(f"exit_mode must be one of: {allowed}") from exc
        try:
            exit_band = ExitBand(self.exit_band)
        except ValueError as exc:
            allowed = ", ".join(band.value for band in ExitBand)
            raise ValueError(f"exit_band must be one of: {allowed}") from exc
        if min(
            self.avwap_multiplier_1,
            self.avwap_multiplier_2,
            self.avwap_multiplier_3,
        ) <= 0:
            raise ValueError("AVWAP multipliers must be positive")
        if self.rigid_stop_loss_pct <= 0:
            raise ValueError(
                "rigid_stop_loss_pct must be positive; live EMA+AVWAP entries "
                "require an exchange-confirmed protective stop"
            )
        if self.trailing_activation_threshold_pct < 0:
            raise ValueError("trailing_activation_threshold_pct must be non-negative")
        if self.trailing_gap_pct < 0:
            raise ValueError("trailing_gap_pct must be non-negative")
        if min(self.maker_fee_pct, self.taker_fee_pct) < 0:
            raise ValueError("fee values must be non-negative")
        if min(self.entry_slippage_pct, self.exit_slippage_pct) < 0:
            raise ValueError("slippage values must be non-negative")
        if self.entry_cancel_bars <= 0:
            raise ValueError("entry_cancel_bars must be positive")
        min_history = max(self.ema_length, self.consecutive_count) + 2
        if self.max_history_bars < min_history:
            raise ValueError(
                f"max_history_bars must be at least {min_history} for this config"
            )
        if self.minimum_balance_usdt < 0:
            raise ValueError("minimum_balance_usdt must be non-negative")
        if self.candle_ready_delay_seconds < 0:
            raise ValueError("candle_ready_delay_seconds must be non-negative")
        if self.execution_interval_minutes <= 0:
            raise ValueError("execution_interval_minutes must be positive")
        if self.api_retries <= 0:
            raise ValueError("api_retries must be positive")
        if self.api_retry_delay_seconds < 0:
            raise ValueError("api_retry_delay_seconds must be non-negative")
        if self.min_stop_update_pct < 0:
            raise ValueError("min_stop_update_pct must be non-negative")
        if self.disable_symbol_hours < 0:
            raise ValueError("disable_symbol_hours must be non-negative")
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "entry_mode", entry_mode)
        object.__setattr__(self, "exit_mode", exit_mode)
        object.__setattr__(self, "exit_band", exit_band)
        object.__setattr__(self, "timeframe", self.timeframe.strip())
        object.__setattr__(
            self, "trailing_tick_timeframe", self.trailing_tick_timeframe.strip()
        )
