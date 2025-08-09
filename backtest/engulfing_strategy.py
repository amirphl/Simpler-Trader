from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Sequence

from candle_downloader.models import Candle

from .base import BacktestContext, BacktestStrategy, TradePerformance
from .patterns import detect_candle_patterns


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

    # --- New knobs for volume / exhaustion detection ---
    volume_window: int = 20             # number of candles for volume & price window
    max_volume_pressure_score: float = 3.0  # threshold for exhaustion engulfing

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.leverage <= 0:
            raise ValueError("leverage must be positive")
        if self.take_profit_pct <= 0:
            raise ValueError("take_profit_pct must be positive")
        if self.volume_window < 2:
            raise ValueError("volume_window must be at least 2")
        if self.max_volume_pressure_score <= 0:
            raise ValueError("max_volume_pressure_score must be positive")



@dataclass
class Position:
    """Represents an open long position."""

    entry_time: datetime
    entry_price: float
    stop_loss: float  # OPEN price of previous candle (the one that triggered signal)
    take_profit: float
    leverage: float
    size: float  # Position size in base currency


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
    avg_recent_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 0.0
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


def calculate_stochastic_k(candles: Sequence[Candle], period: int, index: int) -> float | None:
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

    def name(self) -> str:
        return "EngulfingStrategy"

    def symbols(self) -> Sequence[str]:
        return [self._config.symbol]

    def timeframes(self) -> Sequence[str]:
        return [self._config.timeframe]

    def run(self, context: BacktestContext) -> Sequence[TradePerformance]:
        symbol = self._config.symbol
        timeframe = self._config.timeframe
        candles = context.data.get(symbol, {}).get(timeframe, [])
        if len(candles) < max(100, self._config.window_size + 2):
            return []

        patterns = detect_candle_patterns(candles, doji_size=self._config.doji_size)
        trades: List[TradePerformance] = []
        position: Position | None = None
        initial_capital = context.config.initial_capital
        current_capital = initial_capital

        for idx in range(1, len(candles)):
            candle = candles[idx]
            prev_candle = candles[idx - 1]

            # Check if existing position hits TP or SL
            if position is not None:
                exit_reason: str | None = None
                exit_price: float | None = None

                # Check stop loss (price went below SL)
                if candle.low <= position.stop_loss:
                    exit_price = position.stop_loss
                    exit_reason = "Stop Loss"

                # Check take profit (price went above TP)
                elif candle.high >= position.take_profit:
                    exit_price = position.take_profit
                    exit_reason = "Take Profit"

                if exit_price is not None:
                    # Calculate PnL with leverage
                    price_change = exit_price - position.entry_price
                    pnl = (price_change / position.entry_price) * position.size * position.leverage
                    return_pct = (price_change / position.entry_price) * 100.0 * position.leverage

                    trades.append(
                        TradePerformance(
                            entry_time=position.entry_time,
                            exit_time=candle.open_time,
                            pnl=pnl,
                            return_pct=return_pct,
                            notes=f"{exit_reason} at {exit_price:.2f}",
                            metadata={
                                "entry_price": position.entry_price,
                                "exit_price": exit_price,
                                "stop_loss": position.stop_loss,
                                "take_profit": position.take_profit,
                                "leverage": position.leverage,
                            },
                        )
                    )
                    current_capital += pnl
                    position = None

            # Entry logic: only if no position is open
            if position is None and idx >= 2:
                prev_pattern = patterns[idx - 1] if idx - 1 < len(patterns) else None

                # Check if previous candle is Bullish Engulfing
                if prev_pattern and prev_pattern.bullish_engulfing:
                    # Check if N candles before previous candle are bearish
                    # Previous candle is at idx-1, so we check from idx-1-N to idx-2 (inclusive)
                    check_start = idx - 1 - self._config.window_size
                    if check_start >= 0 and are_bearish(candles, check_start, self._config.window_size):
                        # Check stochastic: K(20) > K(100)
                        stoch_k20 = calculate_stochastic_k(candles, 20, idx - 1)
                        stoch_k100 = calculate_stochastic_k(candles, 100, idx - 1)

                        if stoch_k20 is not None and stoch_k100 is not None and stoch_k20 > stoch_k100:
                            # --- Volume-pressure filter (new) ---
                            score = calculate_volume_pressure_score(
                                candles,
                                engulf_idx=idx - 1,
                                window=self._config.volume_window,
                            )
                            if score is not None and score > self._config.max_volume_pressure_score:
                                # Exhaustion engulfing → skip entry
                                continue

                            # Entry signal triggered
                            entry_price = candle.open
                            stop_loss = prev_candle.open

                            # Basic sanity: if SL >= entry, risk/reward is broken → skip.
                            if stop_loss >= entry_price:
                                continue

                            # take_profit_pct is FRACTION (0.02 = 2%)
                            take_profit = entry_price * (1.0 + self._config.take_profit_pct)

                            # Position size: use all available capital (your original choice)
                            position = Position(
                                entry_time=candle.open_time,
                                entry_price=entry_price,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                leverage=self._config.leverage,
                                size=current_capital,
                            )

        # Close any remaining position at the end
        if position is not None and candles:
            last_candle = candles[-1]
            exit_price = last_candle.close
            price_change = exit_price - position.entry_price
            pnl = (price_change / position.entry_price) * position.size * position.leverage
            return_pct = (price_change / position.entry_price) * 100.0 * position.leverage

            trades.append(
                TradePerformance(
                    entry_time=position.entry_time,
                    exit_time=last_candle.close_time,
                    pnl=pnl,
                    return_pct=return_pct,
                    notes="End of backtest",
                    metadata={
                        "entry_price": position.entry_price,
                        "exit_price": exit_price,
                        "stop_loss": position.stop_loss,
                        "take_profit": position.take_profit,
                        "leverage": position.leverage,
                    },
                )
            )

        return trades

