from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from math import sqrt
from typing import Any, List, Mapping, Sequence, Tuple

from candle_downloader.models import Candle

from .base import BacktestContext, BacktestStrategy, TradePerformance
from .patterns import detect_candle_patterns


class StopLossMode(str, Enum):
    """Stop-loss placement strategy."""

    PERCENT = "percent"
    CLOSE = "close"
    LOW = "low"
    OPEN = "open"
    BODY = "body"


@dataclass(frozen=True)
class EngulfingStrategyConfig:
    """Configuration for the Bullish Engulfing + Stochastic strategy."""

    symbol: str
    timeframe: str
    window_size: int  # N: number of previous candles to check for bearish
    leverage: float
    # IMPORTANT: This is a FRACTION (e.g., 0.02 = 2%).
    take_profit_pct: float

    doji_size: float = 0.05  # For pattern detection

    # --- Volume / exhaustion detection ---
    volume_window: int = 20  # number of candles for volume & price window
    max_volume_pressure_score: float = 3.0  # threshold for exhaustion engulfing
    enable_volume_pressure_filter: bool = True

    # --- Stop-loss behaviour ---
    stop_loss_mode: StopLossMode = StopLossMode.PERCENT
    stop_loss_pct: float = 0.005  # 0.5% default, only used when mode == PERCENT
    exchange_fee_pct: float = 0.0004  # default 4 bps per side

    # --- Optional filters ---
    skip_large_upper_wick: bool = False
    skip_bollinger_cross: bool = False
    bollinger_period: int = 20
    bollinger_stddev: float = 2.0

    # --- Stochastic comparison ---
    enable_stochastic_filter: bool = True
    stochastic_first_line: str = "k"  # "k" or "d"
    stochastic_first_period: int = 20
    stochastic_first_threshold: float | None = None
    stochastic_second_line: str = "k"
    stochastic_second_period: int = 100
    stochastic_second_threshold: float | None = None
    stochastic_comparison: str = "gt"  # "gt" or "lt"
    stochastic_d_smoothing: int = 3
    starting_capital: float = 100.0

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if not self.timeframe:
            raise ValueError("timeframe must not be empty")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.leverage <= 0:
            raise ValueError("leverage must be positive")
        if self.take_profit_pct <= 0:
            raise ValueError("take_profit_pct must be positive")
        if self.doji_size <= 0 or self.doji_size > 1:
            raise ValueError("doji_size must be in the range (0, 1]")
        if self.volume_window < 2:
            raise ValueError("volume_window must be at least 2")
        if self.max_volume_pressure_score <= 0:
            raise ValueError("max_volume_pressure_score must be positive")
        if self.stop_loss_mode is StopLossMode.PERCENT and self.stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be positive for percent mode")
        if self.exchange_fee_pct < 0:
            raise ValueError("exchange_fee_pct must be non-negative")
        if self.starting_capital <= 0:
            raise ValueError("starting_capital must be positive")
        if self.bollinger_period < 2:
            raise ValueError("bollinger_period must be at least 2")
        if self.bollinger_stddev <= 0:
            raise ValueError("bollinger_stddev must be positive")
        if self.stochastic_first_period <= 1:
            raise ValueError("stochastic_first_period must be greater than 1")
        if self.stochastic_second_period <= 1:
            raise ValueError("stochastic_second_period must be greater than 1")
        if self.stochastic_d_smoothing <= 0:
            raise ValueError("stochastic_d_smoothing must be positive")
        comparison = self.stochastic_comparison.strip().lower()
        if comparison not in {"gt", "lt"}:
            raise ValueError("stochastic_comparison must be 'gt' or 'lt'")
        first_line = self.stochastic_first_line.strip().lower()
        if first_line not in {"k", "d"}:
            raise ValueError("stochastic_first_line must be 'k' or 'd'")
        second_line = self.stochastic_second_line.strip().lower()
        if second_line not in {"k", "d"}:
            raise ValueError("stochastic_second_line must be 'k' or 'd'")
        for value, label in (
            (self.stochastic_first_threshold, "stochastic_first_threshold"),
            (self.stochastic_second_threshold, "stochastic_second_threshold"),
        ):
            if value is not None and not 0.0 <= value <= 100.0:
                raise ValueError(f"{label} must be within 0..100 when provided")

        object.__setattr__(self, "stochastic_comparison", comparison)
        object.__setattr__(self, "stochastic_first_line", first_line)
        object.__setattr__(self, "stochastic_second_line", second_line)


@dataclass
class Position:
    """Represents an open long position."""

    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: float
    size: float  # Position size in base currency
    metadata: Mapping[str, float | int | str | None] = field(default_factory=dict)


def calculate_volume_pressure_score(
    candles: Sequence[Candle],
    engulf_idx: int,
    window: int,
) -> float | None:
    """
    Measure how 'exhaustive' the engulfing candle is:
      score = (engulf_volume / avg_recent_volume)
              * relative_position_in_recent_range

    - engulf_idx: index of the engulfing candle (the one with bullish_engulfing)
    - window: number of candles to look back INCLUDING the engulfing candle.
    """
    if engulf_idx <= 0 or not candles:
        return None

    start = max(0, engulf_idx - window + 1)
    window_candles = candles[start : engulf_idx + 1]
    if len(window_candles) < 2:
        return None

    # Separate recent history (without engulfing) and engulfing candle
    recent_candles = window_candles[:-1]
    engulf_candle = window_candles[-1]

    # Average volume of recent candles
    recent_volumes = [getattr(c, "volume", 0.0) or 0.0 for c in recent_candles]
    avg_recent_volume = (
        sum(recent_volumes) / len(recent_volumes) if recent_volumes else 0.0
    )
    if avg_recent_volume <= 0:
        return None

    engulf_volume = getattr(engulf_candle, "volume", 0.0) or 0.0
    volume_ratio = engulf_volume / avg_recent_volume

    # Relative position of engulfing close within recent price range
    highs = [c.high for c in window_candles]
    lows = [c.low for c in window_candles]
    window_high = max(highs)
    window_low = min(lows)
    price_range = window_high - window_low
    if price_range <= 0:
        relative_pos = 0.5  # neutral if no range
    else:
        relative_pos = (engulf_candle.close - window_low) / price_range
        # clamp to [0, 1] just in case of numerical issues
        relative_pos = max(0.0, min(1.0, relative_pos))

    score = volume_ratio * relative_pos
    return score


def calculate_bollinger_bands(
    candles: Sequence[Candle],
    *,
    period: int,
    stddev_multiplier: float,
    index: int,
) -> tuple[float, float, float] | None:
    """Return (lower, middle, upper) Bollinger bands for the provided index."""
    if index < period - 1 or period <= 1:
        return None
    window = candles[index - period + 1 : index + 1]
    if len(window) < period:
        return None

    closes = [c.close for c in window]
    middle = sum(closes) / period
    variance = sum((close - middle) ** 2 for close in closes) / period
    stddev = sqrt(variance)
    upper = middle + stddev_multiplier * stddev
    lower = middle - stddev_multiplier * stddev
    return lower, middle, upper


def calculate_stochastic_k(
    candles: Sequence[Candle], period: int, index: int
) -> float | None:
    """Calculate Stochastic %K for a given candle index.

    %K = ((Close - Lowest Low) / (Highest High - Lowest Low)) * 100
    """
    if index < period - 1:
        return None
    window = candles[index - period + 1 : index + 1]
    if len(window) < period:
        return None
    closes = [c.close for c in window]
    highs = [c.high for c in window]
    lows = [c.low for c in window]
    lowest_low = min(lows)
    highest_high = max(highs)
    current_close = closes[-1]
    if highest_high == lowest_low:
        return 50.0  # Neutral when range is zero
    return ((current_close - lowest_low) / (highest_high - lowest_low)) * 100.0


def calculate_stochastic_d(
    candles: Sequence[Candle],
    period: int,
    index: int,
    smoothing: int,
) -> float | None:
    """Calculate Stochastic %D as SMA of %K with the provided smoothing length."""
    if smoothing <= 0:
        return None
    if index - (smoothing - 1) < 0:
        return None
    values: list[float] = []
    for offset in range(smoothing):
        k_value = calculate_stochastic_k(candles, period, index - offset)
        if k_value is None:
            return None
        values.append(k_value)
    if not values:
        return None
    return sum(values) / len(values)


def calculate_stochastic_value(
    candles: Sequence[Candle],
    *,
    line: str,
    period: int,
    index: int,
    smoothing: int,
) -> float | None:
    """Return stochastic %K or %D based on configuration."""
    line = line.lower()
    if line == "k":
        return calculate_stochastic_k(candles, period, index)
    if line == "d":
        return calculate_stochastic_d(candles, period, index, smoothing=smoothing)
    return None


def are_bearish(candles: Sequence[Candle], start_idx: int, count: int) -> bool:
    """Check if N candles starting from start_idx are all bearish (close < open)."""
    if start_idx < 0 or start_idx + count > len(candles):
        return False
    for i in range(start_idx, start_idx + count):
        if candles[i].close >= candles[i].open:
            return False
    return True


class EngulfingStrategy(BacktestStrategy):
    """Long-only strategy based on Bullish Engulfing pattern + Stochastic confirmation."""

    def __init__(self, config: EngulfingStrategyConfig) -> None:
        self._config = config
        self._log = logging.getLogger(f"{self.__class__.__name__}")

    def name(self) -> str:
        return "EngulfingStrategy"

    def symbols(self) -> Sequence[str]:
        return [self._config.symbol]

    def timeframes(self) -> Sequence[str]:
        return [self._config.timeframe]

    def run(
        self, context: BacktestContext
    ) -> Tuple[Sequence[TradePerformance], Mapping[str, Any] | None]:
        cfg = self._config
        self._log.info(
            "Running strategy with configuration",
            extra={
                "symbol": cfg.symbol,
                "timeframe": cfg.timeframe,
                "window_size": cfg.window_size,
                "leverage": cfg.leverage,
                "take_profit_pct": cfg.take_profit_pct,
                "stop_loss_mode": cfg.stop_loss_mode.value,
                "stop_loss_pct": cfg.stop_loss_pct,
                "skip_large_upper_wick": cfg.skip_large_upper_wick,
                "skip_bollinger_cross": cfg.skip_bollinger_cross,
                "bollinger_period": cfg.bollinger_period,
                "bollinger_stddev": cfg.bollinger_stddev,
                "enable_volume_pressure_filter": cfg.enable_volume_pressure_filter,
                "volume_window": cfg.volume_window,
                "max_volume_pressure_score": cfg.max_volume_pressure_score,
                "enable_stochastic_filter": cfg.enable_stochastic_filter,
                "stochastic_first_line": cfg.stochastic_first_line,
                "stochastic_first_period": cfg.stochastic_first_period,
                "stochastic_first_threshold": cfg.stochastic_first_threshold,
                "stochastic_second_line": cfg.stochastic_second_line,
                "stochastic_second_period": cfg.stochastic_second_period,
                "stochastic_second_threshold": cfg.stochastic_second_threshold,
                "stochastic_comparison": cfg.stochastic_comparison,
                "stochastic_d_smoothing": cfg.stochastic_d_smoothing,
                "doji_size": cfg.doji_size,
                "exchange_fee_pct": cfg.exchange_fee_pct,
                "starting_capital": cfg.starting_capital,
            },
        )
        symbol = cfg.symbol
        timeframe = cfg.timeframe
        candles = context.data.get(symbol, {}).get(timeframe, [])
        ignore_count = context.ignore_candles.get(symbol, {}).get(timeframe, 0)
        start_index = max(ignore_count, self._required_history())
        self._log.info(
            "Candles loaded",
            extra={
                "symbol": symbol,
                "timeframe": timeframe,
                "length": len(candles),
                "ignore_count": ignore_count,
                "start_index": start_index,
            },
        )
        if len(candles) <= start_index:
            return [], {"note": "insufficient_data", "candles": len(candles)}

        patterns = detect_candle_patterns(candles, doji_size=cfg.doji_size)
        trades: List[TradePerformance] = []
        position: Position | None = None
        current_capital = cfg.starting_capital
        stats: dict[str, int | float] = {
            "signals_detected": 0,
            "entries_opened": 0,
            "entries_skipped_stop_above_entry": 0,
            "entries_skipped_upper_wick": 0,
            "entries_skipped_bollinger": 0,
            "entries_skipped_volume": 0,
            "entries_skipped_stochastic": 0,
            "open_positions_force_closed": 0,
        }

        for idx in range(start_index, len(candles)):
            candle = candles[idx]
            prev_candle = candles[idx - 1]

            # Check if existing position hits TP or SL
            if position is not None:
                exit_price, exit_reason = self._check_exit(candle, position)
                if exit_price is not None and exit_reason is not None:
                    pnl = self._record_exit(
                        candle=candle,
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        position=position,
                        trades=trades,
                    )
                    current_capital += pnl
                    position = None

            # Entry logic: only if no position is open
            if position is None:
                prev_pattern = patterns[idx - 1] if idx - 1 < len(patterns) else None

                # Check if previous candle is Bullish Engulfing
                if prev_pattern and prev_pattern.bullish_engulfing:
                    stats["signals_detected"] += 1
                    # Check if N candles before previous candle are bearish
                    # Previous candle is at idx-1, so we check from idx-1-N to idx-2 (inclusive)
                    check_start = idx - 1 - cfg.window_size
                    if check_start >= 0 and are_bearish(candles, check_start, cfg.window_size):
                        stochastic_values = self._resolve_stochastic_values(
                            candles,
                            idx - 1,
                        )
                        if not self._passes_stochastic_filter(stochastic_values):
                            stats["entries_skipped_stochastic"] += 1
                            continue

                        # --- Volume-pressure filter (optional) ---
                        score: float | None = None
                        if cfg.enable_volume_pressure_filter:
                            score = calculate_volume_pressure_score(
                                candles,
                                engulf_idx=idx - 1,
                                window=cfg.volume_window,
                            )
                            if score is not None and score > cfg.max_volume_pressure_score:
                                stats["entries_skipped_volume"] += 1
                                continue

                        engulf_candle = prev_candle

                        if cfg.skip_large_upper_wick:
                            upper_wick = engulf_candle.high - engulf_candle.close
                            body = engulf_candle.close - engulf_candle.open
                            if body <= 0 or upper_wick > body:
                                stats["entries_skipped_upper_wick"] += 1
                                continue

                        if cfg.skip_bollinger_cross:
                            bands = calculate_bollinger_bands(
                                candles,
                                period=cfg.bollinger_period,
                                stddev_multiplier=cfg.bollinger_stddev,
                                index=idx - 1,
                            )
                            if bands is not None:
                                _, _, upper_band = bands
                                if engulf_candle.high >= upper_band:
                                    stats["entries_skipped_bollinger"] += 1
                                    continue

                        # Entry signal triggered
                        entry_price = candle.open
                        stop_loss = self._compute_stop_loss(entry_price, engulf_candle)

                        # Basic sanity: if SL >= entry, risk/reward is broken → skip.
                        if stop_loss >= entry_price:
                            stats["entries_skipped_stop_above_entry"] += 1
                            continue

                        take_profit = entry_price * (1.0 + cfg.take_profit_pct)

                        position = Position(
                            entry_time=candle.open_time,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            leverage=cfg.leverage,
                            size=current_capital,
                            metadata={
                                "signal_index": idx - 1,
                                "signal_open_time": engulf_candle.open_time.isoformat(),
                                "volume_pressure_score": score,
                                "stochastic_first": stochastic_values[0],
                                "stochastic_second": stochastic_values[1],
                            },
                        )
                        stats["entries_opened"] += 1
                        exit_price, exit_reason = self._check_exit(candle, position)
                        if exit_price is not None and exit_reason is not None:
                            pnl = self._record_exit(
                                candle=candle,
                                exit_price=exit_price,
                                exit_reason=exit_reason,
                                position=position,
                                trades=trades,
                            )
                            current_capital += pnl
                            position = None

        # Close any remaining position at the end
        if position is not None and candles:
            last_candle = candles[-1]
            pnl = self._record_exit(
                candle=last_candle,
                exit_price=last_candle.close,
                exit_reason="End of backtest",
                position=position,
                trades=trades,
                exit_time=last_candle.close_time,
                note_override="End of backtest",
            )
            current_capital += pnl
            stats["open_positions_force_closed"] += 1
            position = None

        stats["ending_capital"] = current_capital
        return trades, stats

    def _required_history(self) -> int:
        cfg = self._config
        stochastic_history = max(
            cfg.stochastic_first_period,
            cfg.stochastic_second_period,
        ) + max(cfg.stochastic_d_smoothing - 1, 0)
        return max(
            2,
            cfg.window_size + 1,
            cfg.volume_window,
            cfg.bollinger_period,
            stochastic_history,
        )

    def _check_exit(
        self, candle: Candle, position: Position
    ) -> tuple[float | None, str | None]:
        if candle.low <= position.stop_loss:
            return position.stop_loss, "Stop Loss"
        if candle.high >= position.take_profit:
            return position.take_profit, "Take Profit"
        return None, None

    def _record_exit(
        self,
        *,
        candle: Candle,
        exit_price: float,
        exit_reason: str,
        position: Position,
        trades: List[TradePerformance],
        exit_time: datetime | None = None,
        note_override: str | None = None,
    ) -> float:
        pnl, fees_paid, return_pct = self._compute_trade_financials(
            exit_price, position
        )
        note = note_override or f"{exit_reason} at {exit_price:.2f}"

        trades.append(
            TradePerformance(
                entry_time=position.entry_time,
                exit_time=exit_time or candle.close_time,
                pnl=pnl,
                return_pct=return_pct,
                notes=note,
                metadata={
                    "entry_price": position.entry_price,
                    "exit_price": exit_price,
                    "stop_loss": position.stop_loss,
                    "take_profit": position.take_profit,
                    "leverage": position.leverage,
                    "fees": fees_paid,
                    **position.metadata,
                },
            )
        )
        return pnl

    def _compute_trade_financials(
        self, exit_price: float, position: Position
    ) -> tuple[float, float, float]:
        price_change = exit_price - position.entry_price
        gross_pnl = (
            (price_change / position.entry_price) * position.size * position.leverage
        )
        fees_paid = self._calculate_fees(exit_price, position)
        pnl = gross_pnl - fees_paid
        return_pct = (pnl / position.size) * 100.0 if position.size else 0.0
        return pnl, fees_paid, return_pct

    def _calculate_fees(self, exit_price: float, position: Position) -> float:
        fee_pct = self._config.exchange_fee_pct
        if fee_pct <= 0:
            return 0.0
        notional_entry = position.size * position.leverage
        if notional_entry <= 0 or position.entry_price <= 0:
            return 0.0
        quantity = notional_entry / position.entry_price
        entry_fee = notional_entry * fee_pct
        exit_fee = quantity * exit_price * fee_pct
        return entry_fee + exit_fee

    def _compute_stop_loss(self, entry_price: float, engulf_candle: Candle) -> float:
        mode = self._config.stop_loss_mode
        if mode is StopLossMode.PERCENT:
            return entry_price * (1.0 - self._config.stop_loss_pct)
        if mode is StopLossMode.CLOSE:
            return engulf_candle.close
        if mode is StopLossMode.LOW:
            return engulf_candle.low
        if mode is StopLossMode.OPEN:
            return engulf_candle.open
        if mode is StopLossMode.BODY:
            body = engulf_candle.close - engulf_candle.open
            if body <= 0:
                return engulf_candle.open
            return engulf_candle.close - (self._config.stop_loss_pct * body)
        raise ValueError(f"Unsupported stop-loss mode: {mode}")

    def _resolve_stochastic_values(
        self, candles: Sequence[Candle], index: int
    ) -> tuple[float | None, float | None]:
        cfg = self._config
        first = calculate_stochastic_value(
            candles,
            line=cfg.stochastic_first_line,
            period=cfg.stochastic_first_period,
            index=index,
            smoothing=cfg.stochastic_d_smoothing,
        )
        second = calculate_stochastic_value(
            candles,
            line=cfg.stochastic_second_line,
            period=cfg.stochastic_second_period,
            index=index,
            smoothing=cfg.stochastic_d_smoothing,
        )
        return first, second

    def _passes_stochastic_filter(
        self, values: tuple[float | None, float | None]
    ) -> bool:
        if not self._config.enable_stochastic_filter:
            return True

        cfg = self._config
        first, second = values
        if first is None or second is None:
            return False

        if not self._passes_threshold(first, cfg.stochastic_first_threshold):
            return False
        if not self._passes_threshold(second, cfg.stochastic_second_threshold):
            return False

        if cfg.stochastic_comparison == "gt":
            return first > second
        return first < second

    def _passes_threshold(self, value: float, threshold: float | None) -> bool:
        if threshold is None:
            return True
        if self._config.stochastic_comparison == "gt":
            return value >= threshold
        return value <= threshold
