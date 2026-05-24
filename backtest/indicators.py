from __future__ import annotations

from typing import List, Optional, Sequence

from candle_downloader.models import Candle

_EPSILON = 1e-9


def ema(values: Sequence[float], period: int) -> List[Optional[float]]:
    if period <= 0:
        raise ValueError("period must be positive")

    result: List[Optional[float]] = [None] * len(values)
    if len(values) < period:
        return result

    seed = sum(values[:period]) / period
    result[period - 1] = seed
    alpha = 2.0 / (period + 1)

    value = seed
    for idx in range(period, len(values)):
        value = values[idx] * alpha + value * (1.0 - alpha)
        result[idx] = value

    return result


def rsi(values: Sequence[float], period: int) -> List[Optional[float]]:
    if period <= 0:
        raise ValueError("period must be positive")

    result: List[Optional[float]] = [None] * len(values)
    if len(values) < period + 1:
        return result

    gains = [0.0] * len(values)
    losses = [0.0] * len(values)
    for idx in range(1, len(values)):
        change = values[idx] - values[idx - 1]
        gains[idx] = max(change, 0.0)
        losses[idx] = max(-change, 0.0)

    avg_gain = sum(gains[1 : period + 1]) / period
    avg_loss = sum(losses[1 : period + 1]) / period
    result[period] = _rsi_from_averages(avg_gain, avg_loss)

    for idx in range(period + 1, len(values)):
        avg_gain = ((period - 1) * avg_gain + gains[idx]) / period
        avg_loss = ((period - 1) * avg_loss + losses[idx]) / period
        result[idx] = _rsi_from_averages(avg_gain, avg_loss)

    return result


def atr(candles: Sequence[Candle], length: int) -> List[Optional[float]]:
    if length <= 0:
        raise ValueError("length must be positive")

    result: List[Optional[float]] = [None] * len(candles)
    if len(candles) < length:
        return result

    true_ranges: List[float] = [0.0] * len(candles)
    for idx, candle in enumerate(candles):
        if idx == 0:
            true_ranges[idx] = candle.high - candle.low
            continue
        prev_close = candles[idx - 1].close
        true_ranges[idx] = max(
            candle.high - candle.low,
            abs(candle.high - prev_close),
            abs(candle.low - prev_close),
        )

    seed = sum(true_ranges[:length]) / float(length)
    result[length - 1] = seed
    value = seed
    for idx in range(length, len(candles)):
        value = ((length - 1) * value + true_ranges[idx]) / float(length)
        result[idx] = value

    return result


def sma(values: Sequence[float], period: int) -> List[Optional[float]]:
    if period <= 0:
        raise ValueError("period must be positive")

    result: List[Optional[float]] = [None] * len(values)
    if len(values) < period:
        return result

    window_sum = sum(values[:period])
    result[period - 1] = window_sum / period
    for idx in range(period, len(values)):
        window_sum += values[idx] - values[idx - period]
        result[idx] = window_sum / period

    return result


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    if avg_loss <= _EPSILON:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


__all__ = ["atr", "ema", "rsi", "sma"]
