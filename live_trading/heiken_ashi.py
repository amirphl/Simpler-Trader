"""Heiken Ashi candle calculations."""

from __future__ import annotations

from typing import List

from candle_downloader.models import Candle

from .models import HekenAshiCandle


def calculate_heiken_ashi(candles: List[Candle]) -> List[HekenAshiCandle]:
    """Convert regular candles to Heiken Ashi candles.

    Heiken Ashi formulas:
    - HA_Close = (Open + High + Low + Close) / 4
    - HA_Open = (Previous HA_Open + Previous HA_Close) / 2
    - HA_High = Max(High, HA_Open, HA_Close)
    - HA_Low = Min(Low, HA_Open, HA_Close)

    Args:
        candles: List of regular OHLC candles

    Returns:
        List of Heiken Ashi candles
    """
    if not candles:
        return []

    ha_candles: List[HekenAshiCandle] = []

    # First candle: HA_Open = (Open + Close) / 2
    first = candles[0]
    ha_open = (first.open + first.close) / 2
    ha_close = (first.open + first.high + first.low + first.close) / 4
    ha_high = max(first.high, ha_open, ha_close)
    ha_low = min(first.low, ha_open, ha_close)

    ha_candles.append(
        HekenAshiCandle(
            open_time=first.open_time,
            close_time=first.close_time,
            ha_open=ha_open,
            ha_high=ha_high,
            ha_low=ha_low,
            ha_close=ha_close,
            orig_open=first.open,
            orig_high=first.high,
            orig_low=first.low,
            orig_close=first.close,
            volume=first.volume,
        )
    )

    # Subsequent candles
    for i in range(1, len(candles)):
        candle = candles[i]
        prev_ha = ha_candles[-1]

        ha_close = (candle.open + candle.high + candle.low + candle.close) / 4
        ha_open = (prev_ha.ha_open + prev_ha.ha_close) / 2
        ha_high = max(candle.high, ha_open, ha_close)
        ha_low = min(candle.low, ha_open, ha_close)

        ha_candles.append(
            HekenAshiCandle(
                open_time=candle.open_time,
                close_time=candle.close_time,
                ha_open=ha_open,
                ha_high=ha_high,
                ha_low=ha_low,
                ha_close=ha_close,
                orig_open=candle.open,
                orig_high=candle.high,
                orig_low=candle.low,
                orig_close=candle.close,
                volume=candle.volume,
            )
        )

    return ha_candles


def check_consecutive_pattern(
    candles: List[HekenAshiCandle],
    pattern_length: int,
    bullish: bool,
) -> bool:
    """Check if the last N candles follow a pattern (all bullish or all bearish).

    Args:
        candles: List of Heiken Ashi candles
        pattern_length: Number of consecutive candles to check
        bullish: True to check for bullish pattern, False for bearish

    Returns:
        True if pattern is satisfied
    """
    if len(candles) < pattern_length:
        return False

    # Check last N candles
    for i in range(len(candles) - pattern_length, len(candles)):
        if bullish:
            if not candles[i].is_bullish():
                return False
        else:
            if not candles[i].is_bearish():
                return False

    return True


def detect_reversal_signal(
    ha_candles: List[HekenAshiCandle],
    lookback_candles: int,
) -> str | None:
    """Detect Heiken Ashi reversal pattern.

    Pattern for LONG signal:
    - Last W candles are bearish (decline)
    - Current candle (most recent) is bullish (reversal)

    Pattern for SHORT signal:
    - Last W candles are bullish (rally)
    - Current candle (most recent) is bearish (reversal)

    Args:
        candles: List of Heiken Ashi candles (must have at least lookback_candles + 1)
        lookback_candles: Number of candles before current to check (W parameter)

    Returns:
        "LONG" for bullish reversal, "SHORT" for bearish reversal, None for no signal
    """
    required_length = lookback_candles + 1
    if len(ha_candles) < required_length:
        return None

    # Current candle (most recent)
    last_ha_candle = ha_candles[-1]

    # Previous W candles (before last)
    previous_candles = ha_candles[-(lookback_candles + 1) : -1]

    # TODO:
    # Check for LONG signal: previous bearish, last bullish
    # if (
    #     last_candle.is_bearish()
    #     and last_ha_candle.is_bullish()
    #     and check_consecutive_pattern(previous_candles, lookback_candles, bullish=False)
    # ):
    #     return "LONG"
    if last_ha_candle.is_bullish() and check_consecutive_pattern(
        previous_candles, lookback_candles, bullish=False
    ):
        return "LONG"

    # TODO:
    # Check for SHORT signal: previous bullish, last bearish
    # if (
    #     last_candle.is_bullish()
    #     and last_ha_candle.is_bearish()
    #     and check_consecutive_pattern(previous_candles, lookback_candles, bullish=True)
    # ):
    #     return "SHORT"

    if last_ha_candle.is_bearish() and check_consecutive_pattern(
        previous_candles, lookback_candles, bullish=True
    ):
        return "SHORT"

    return None


def detect_reversal_signal_v2(
    ha_candles: List[HekenAshiCandle],
    lookback_candles: int,
) -> str | None:
    """Detect Heiken Ashi reversal pattern."""
    required_length = lookback_candles + 1
    if len(ha_candles) < required_length:
        return None

    # Current candle (most recent)
    last_ha_candle = ha_candles[-1]

    # Previous W ha candles (before last)
    previous_ha_candles = ha_candles[-(lookback_candles + 1) : -1]

    # TODO:
    # Check for LONG signal: previous bearish, last bullish
    # if (
    #     last_candle.is_bearish()
    #     and last_ha_candle.is_bullish()
    #     and check_consecutive_pattern(
    #         previous_ha_candles, lookback_candles, bullish=False
    #     )
    # ):
    #     return "LONG"
    if (
        last_ha_candle.is_bullish()
        and check_consecutive_pattern(
            previous_ha_candles, lookback_candles, bullish=False
        )
        # and last_ha_candle.ha_close > previous_ha_candles[-1].ha_high
    ):
        return "LONG"

    # TODO:
    # Check for SHORT signal: previous bullish, last bearish
    # if (
    #     last_candle.is_bullish()
    #     and last_ha_candle.is_bearish()
    #     and check_consecutive_pattern(
    #         previous_ha_candles, lookback_candles, bullish=True
    #     )
    # ):
    #     return "SHORT"
    if (
        last_ha_candle.is_bearish()
        and check_consecutive_pattern(
            previous_ha_candles, lookback_candles, bullish=True
        )
        # and last_ha_candle.ha_close < previous_ha_candles[-1].ha_low
    ):
        return "SHORT"

    return None
