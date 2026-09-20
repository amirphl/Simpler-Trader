"""Configuration and public types for the EMA/AVWAP pullback strategy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from candle_downloader.binance import interval_to_milliseconds
from candle_downloader.models import normalize_usdm_perpetual_symbol

Direction = Literal["long", "short"]
EmaValidationMode = Literal["body", "wick"]
SetupWaitingReplacementMode = Literal["keep_waiting", "replace_waiting"]
# EMA + AVWAP live entries are always sized from a capped notional budget.
# Retaining a stop-distance alternative in the backtest makes its results
# incomparable to the live coordinator, which deliberately rejects it.
PositionSizingMode = Literal["risk_amount_per_price"]


class EntryMode(str, Enum):
    """When a qualifying pullback is allowed to open a position."""

    CLOSE = "close"
    LIVE = "live"


class ExitMode(str, Enum):
    """When the AVWAP profit target is allowed to close a position."""

    CLOSE = "close"
    LIVE = "live"


class ExitBand(str, Enum):
    """AVWAP standard-deviation band used as the profit target."""

    BAND_1 = "band_1"
    BAND_2 = "band_2"

    @property
    def number(self) -> int:
        return 1 if self is ExitBand.BAND_1 else 2


class FundingMode(str, Enum):
    """Whether settled perpetual funding is included in trade PnL."""

    HISTORICAL = "historical"
    NONE = "none"


@dataclass(frozen=True)
class EmaAvwapPullbackStrategyConfig:
    """Validated parameters for an EMA/AVWAP pullback backtest."""

    symbol: str
    timeframe: str

    initial_equity: float = 100.0
    leverage: float = 10.0
    max_entry_notional_usdt: float = 15.0
    max_position_size_pct: float = 10.0
    position_notional_pct: float = 1.0
    minimum_balance_usdt: float = 0.0

    ema_length: int = 55
    consecutive_count: int = 4
    ema_validation_mode: EmaValidationMode = "body"
    setup_waiting_replacement_mode: SetupWaitingReplacementMode = "keep_waiting"
    max_setup_age_bars: int = 3
    max_entry_deviation_pct: float = 1.0
    position_sizing_mode: PositionSizingMode = "risk_amount_per_price"
    # Historical OHLCV contains only completed candles.  Make the unbiased,
    # closed-candle execution contract the default for a backtest.  ``live``
    # remains available as an explicitly selected OHLC approximation of the
    # production tick/forming-candle path; it is not a like-for-like replay.
    entry_mode: EntryMode | str = EntryMode.CLOSE
    exit_mode: ExitMode | str = ExitMode.CLOSE
    exit_band: ExitBand | str = ExitBand.BAND_1

    avwap_multiplier_1: float = 1.0
    avwap_multiplier_2: float = 2.0
    avwap_multiplier_3: float = 3.0

    rigid_stop_loss_pct: float = 3.0
    trailing_activation_threshold_pct: float = 0.0
    trailing_gap_pct: float = 1.0

    maker_fee_pct: float = 0.0002
    taker_fee_pct: float = 0.0006
    entry_slippage_pct: float = 0.0
    exit_slippage_pct: float = 0.0
    # Historical funding is a market datum, not an optimizable daily-rate
    # assumption.  It uses Binance's settled funding rate and mark price.
    funding_mode: FundingMode | str = FundingMode.HISTORICAL
    use_gap_cross_detection: bool = True
    max_decision_log_entries: int = 20000

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper()
        timeframe = self.timeframe.strip()
        if not symbol:
            raise ValueError("symbol must not be empty")
        if not timeframe:
            raise ValueError("timeframe must not be empty")
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        if self.leverage <= 0:
            raise ValueError("leverage must be positive")
        if self.max_entry_notional_usdt <= 0:
            raise ValueError("max_entry_notional_usdt must be positive")
        if not 0 < self.max_position_size_pct <= 100:
            raise ValueError("max_position_size_pct must be in (0, 100]")
        if not 0 < self.position_notional_pct <= 100:
            raise ValueError("position_notional_pct must be in (0, 100]")
        if self.minimum_balance_usdt < 0:
            raise ValueError("minimum_balance_usdt must be non-negative")
        if self.ema_length <= 0:
            raise ValueError("ema_length must be positive")
        if self.consecutive_count <= 0:
            raise ValueError("consecutive_count must be positive")
        if self.ema_validation_mode not in {"body", "wick"}:
            raise ValueError("ema_validation_mode must be one of: body, wick")
        if self.setup_waiting_replacement_mode not in {
            "keep_waiting",
            "replace_waiting",
        }:
            raise ValueError(
                "setup_waiting_replacement_mode must be one of: keep_waiting, replace_waiting"
            )
        if self.max_setup_age_bars <= 0:
            raise ValueError("max_setup_age_bars must be positive")
        if self.max_entry_deviation_pct < 0:
            raise ValueError("max_entry_deviation_pct must be non-negative")
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
        try:
            funding_mode = FundingMode(self.funding_mode)
        except ValueError as exc:
            allowed = ", ".join(mode.value for mode in FundingMode)
            raise ValueError(f"funding_mode must be one of: {allowed}") from exc
        if (
            min(
                self.avwap_multiplier_1,
                self.avwap_multiplier_2,
                self.avwap_multiplier_3,
            )
            <= 0
        ):
            raise ValueError("AVWAP multipliers must be positive")
        if self.rigid_stop_loss_pct <= 0:
            raise ValueError(
                "rigid_stop_loss_pct must be positive; EMA+AVWAP entries "
                "require the live strategy's protective stop"
            )
        if self.trailing_activation_threshold_pct < 0:
            raise ValueError("trailing_activation_threshold_pct must be non-negative")
        if self.trailing_gap_pct < 0:
            raise ValueError("trailing_gap_pct must be non-negative")
        if min(self.maker_fee_pct, self.taker_fee_pct) < 0:
            raise ValueError("fee values must be non-negative")
        if min(self.entry_slippage_pct, self.exit_slippage_pct) < 0:
            raise ValueError("slippage values must be non-negative")
        if max(self.entry_slippage_pct, self.exit_slippage_pct) >= 1:
            raise ValueError("slippage values must be below 1.0 (100%)")
        if self.max_decision_log_entries <= 0:
            raise ValueError("max_decision_log_entries must be positive")
        interval_to_milliseconds(timeframe)
        object.__setattr__(self, "symbol", normalize_usdm_perpetual_symbol(symbol))
        object.__setattr__(self, "timeframe", timeframe)
        object.__setattr__(self, "entry_mode", entry_mode)
        object.__setattr__(self, "exit_mode", exit_mode)
        object.__setattr__(self, "exit_band", exit_band)
        object.__setattr__(self, "funding_mode", funding_mode)


__all__ = [
    "Direction",
    "EmaAvwapPullbackStrategyConfig",
    "EmaValidationMode",
    "EntryMode",
    "ExitMode",
    "ExitBand",
    "FundingMode",
    "PositionSizingMode",
    "SetupWaitingReplacementMode",
]
