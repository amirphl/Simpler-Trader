from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence

from candle_downloader.models import Candle


@dataclass(frozen=True)
class EngulfingSignalConfig:
    """Configuration for transforming candle history into live engulfing signals."""

    timeframe: str
    window_size: int = 5
    leverage: float = 5.0
    take_profit_pct: float = 0.02  # e.g. 0.02 == 2%
    volume_window: int = 20
    max_volume_pressure_score: float = 3.0

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.leverage <= 0:
            raise ValueError("leverage must be positive")
        if not (0 < self.take_profit_pct < 1):
            raise ValueError("take_profit_pct must be between 0 and 1 (fractional)")
        if self.volume_window < 2:
            raise ValueError("volume_window must be at least 2")
        if self.max_volume_pressure_score <= 0:
            raise ValueError("max_volume_pressure_score must be positive")


@dataclass(frozen=True)
class EngulfingSignal:
    """Representation of a live trading signal."""

    symbol: str
    timeframe: str
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: float
    notes: Optional[str] = None


class EngulfingSignalDetector:
    """Reusable evaluator that turns candle sequences into live signals."""

    def __init__(self, config: EngulfingSignalConfig) -> None:
        self._config = config

    def evaluate(self, symbol: str, candles: Sequence[Candle]) -> Optional[EngulfingSignal]:
        """Return a signal if the newest candle completes the strategy conditions."""
        if len(candles) < max(self._config.window_size + 2, 101):
            return None

        idx = len(candles) - 1
        if idx < 2:
            return None

        entry_candle = candles[idx]
        engulf_candle = candles[idx - 1]
        reference_candle = candles[idx - 2]

        if not _is_bullish_engulfing(reference_candle, engulf_candle):
            return None

        start_idx = idx - 1 - self._config.window_size
        if start_idx < 0 or not are_bearish(candles, start_idx, self._config.window_size):
            return None

        stoch_k20 = calculate_stochastic_k(candles, 20, idx - 1)
        stoch_k100 = calculate_stochastic_k(candles, 100, idx - 1)
        if stoch_k20 is None or stoch_k100 is None or stoch_k20 <= stoch_k100:
            return None

        # score = calculate_volume_pressure_score(candles, idx - 1, self._config.volume_window)
        # if score is not None and score > self._config.max_volume_pressure_score:
        #     return None

        entry_price = entry_candle.open
        stop_loss = engulf_candle.open
        if stop_loss >= entry_price:
            return None

        take_profit = entry_price * (1.0 + self._config.take_profit_pct)
        notes = f"Stoch20 {stoch_k20:.2f} > Stoch100 {stoch_k100:.2f}"
        # if score is not None:
        #     notes = f"{notes}; vol-pressure {score:.2f}"

        return EngulfingSignal(
            symbol=symbol,
            timeframe=self._config.timeframe,
            entry_time=entry_candle.open_time,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=self._config.leverage,
            notes=notes,
        )


def are_bearish(candles: Sequence[Candle], start_idx: int, count: int) -> bool:
    """Return True if every candle in range [start_idx, start_idx+count) closed lower than it opened."""
    if start_idx < 0 or start_idx + count > len(candles):
        return False
    for idx in range(start_idx, start_idx + count):
        candle = candles[idx]
        if candle.close >= candle.open:
            return False
    return True


def calculate_stochastic_k(candles: Sequence[Candle], period: int, index: int) -> Optional[float]:
    """Compute Stochastic %K for a candle index."""
    if period <= 1 or index < 0:
        return None
    if index < period - 1:
        return None

    window = candles[index - period + 1 : index + 1]
    if len(window) < period:
        return None
    highs = [c.high for c in window]
    lows = [c.low for c in window]
    highest_high = max(highs)
    lowest_low = min(lows)
    if highest_high == lowest_low:
        return 50.0
    current_close = window[-1].close
    return ((current_close - lowest_low) / (highest_high - lowest_low)) * 100.0


def calculate_volume_pressure_score(
    candles: Sequence[Candle],
    engulf_idx: int,
    window: int,
) -> Optional[float]:
    """Quantify how explosive the engulfing candle is compared to recent history."""
    if engulf_idx <= 0 or window < 2 or not candles:
        return None

    start = max(0, engulf_idx - window + 1)
    window_candles = candles[start : engulf_idx + 1]
    if len(window_candles) < 2:
        return None

    recent_candles = window_candles[:-1]
    engulf_candle = window_candles[-1]

    recent_volumes = [getattr(candle, "volume", 0.0) or 0.0 for candle in recent_candles]
    avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 0.0
    if avg_volume <= 0:
        return None

    engulf_volume = getattr(engulf_candle, "volume", 0.0) or 0.0
    volume_ratio = engulf_volume / avg_volume

    highs = [c.high for c in window_candles]
    lows = [c.low for c in window_candles]
    price_range = max(highs) - min(lows)
    if price_range <= 0:
        relative_pos = 0.5
    else:
        relative_pos = (engulf_candle.close - min(lows)) / price_range
        relative_pos = max(0.0, min(1.0, relative_pos))

    return volume_ratio * relative_pos


def _is_bullish_engulfing(previous: Candle, current: Candle) -> bool:
    """Mirror the PineScript bullish engulfing rule-set."""
    return (
        previous.open > previous.close
        and current.close > current.open
        and current.close >= previous.open
        and previous.close >= current.open
        and (current.close - current.open) > (previous.open - previous.close)
    )


