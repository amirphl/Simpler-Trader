from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Sequence

from candle_downloader.models import Candle

_PATTERN_LABELS: Dict[str, str] = {
    "doji": "Doji",
    "evening_star": "Evening Star",
    "morning_star": "Morning Star",
    "shooting_star": "Shooting Star",
    "hammer": "Hammer",
    "inverted_hammer": "Inverted Hammer",
    "bearish_harami": "Bearish Harami",
    "bullish_harami": "Bullish Harami",
    "bearish_engulfing": "Bearish Engulfing",
    "bullish_engulfing": "Bullish Engulfing",
    "piercing_line": "Piercing Line",
    "bullish_belt": "Bullish Belt",
    "bullish_kicker": "Bullish Kicker",
    "bearish_kicker": "Bearish Kicker",
    "hanging_man": "Hanging Man",
    "dark_cloud_cover": "Dark Cloud Cover",
}


@dataclass(frozen=True)
class CandlePatternSignals:
    """Boolean flags for each candlestick formation defined in the Pinescript study."""

    index: int
    timestamp: datetime
    doji: bool
    evening_star: bool
    morning_star: bool
    shooting_star: bool
    hammer: bool
    inverted_hammer: bool
    bearish_harami: bool
    bullish_harami: bool
    bearish_engulfing: bool
    bullish_engulfing: bool
    piercing_line: bool
    bullish_belt: bool
    bullish_kicker: bool
    bearish_kicker: bool
    hanging_man: bool
    dark_cloud_cover: bool

    def active_labels(self) -> List[str]:
        """Return human-friendly labels for all patterns that fired on the candle."""
        return [label for attr, label in _PATTERN_LABELS.items() if getattr(self, attr)]


def detect_candle_patterns(candles: Sequence[Candle], doji_size: float = 0.05) -> List[CandlePatternSignals]:
    """Replicate the TradingView Pinescript candle label logic for a candle sequence."""
    if not candles:
        return []
    if not 0.01 <= doji_size <= 1.0:
        raise ValueError("doji_size must be between 0.01 and 1.0")

    opens = [candle.open for candle in candles]
    highs = [candle.high for candle in candles]
    lows = [candle.low for candle in candles]
    closes = [candle.close for candle in candles]

    signals: List[CandlePatternSignals] = []

    def value(series: Sequence[float], idx: int, offset: int) -> float | None:
        target = idx - offset
        if target < 0 or target >= len(series):
            return None
        return series[target]

    def rolling_extreme(
        series: Sequence[float],
        idx: int,
        length: int,
        offset: int,
        extrema: Callable[[Sequence[float]], float],
    ) -> float | None:
        end = idx - offset
        if end < 0:
            return None
        start = max(0, end - length + 1)
        if start > end:
            return None
        return extrema(series[start : end + 1])

    for idx, candle in enumerate(candles):
        open_ = opens[idx]
        high = highs[idx]
        low = lows[idx]
        close = closes[idx]
        body = abs(open_ - close)
        range_ = max(high - low, 0.0)
        denom = 0.001 + range_

        open1 = value(opens, idx, 1)
        open2 = value(opens, idx, 2)
        high1 = value(highs, idx, 1)
        high2 = value(highs, idx, 2)
        low1 = value(lows, idx, 1)
        close1 = value(closes, idx, 1)
        close2 = value(closes, idx, 2)

        doji = range_ > 0 and body <= range_ * doji_size

        evening_star = False
        morning_star = False
        shooting_star = False
        hammer = False
        inverted_hammer = False
        bearish_harami = False
        bullish_harami = False
        bearish_engulfing = False
        bullish_engulfing = False
        piercing_line = False
        bullish_belt = False
        bullish_kicker = False
        bearish_kicker = False
        hanging_man = False
        dark_cloud_cover = False

        if close2 is not None and open2 is not None and open1 is not None and close1 is not None:
            min_prev = min(open1, close1)
            max_prev = max(open1, close1)
            evening_star = (
                close2 > open2 and min_prev > close2 and open_ < min_prev and close < open_
            )
            morning_star = (
                close2 < open2 and max_prev < close2 and open_ > max_prev and close > open_
            )

        if open1 is not None and close1 is not None:
            shooting_star = (
                open1 < close1
                and open_ > close1
                and high - max(open_, close) >= abs(open_ - close) * 3
                and min(close, open_) - low <= abs(open_ - close)
            )

            bearish_harami = (
                close1 > open1
                and open_ > close
                and open_ <= close1
                and open1 <= close
                and open_ - close < close1 - open1
            )
            bullish_harami = (
                open1 > close1
                and close > open_
                and close <= open1
                and close1 <= open_
                and close - open_ < open1 - close1
            )

            bearish_engulfing = (
                close1 > open1
                and open_ > close
                and open_ >= close1
                and open1 >= close
                and open_ - close > close1 - open1
            )
            bullish_engulfing = (
                open1 > close1
                and close > open_
                and close >= open1
                and close1 >= open_
                and close - open_ > open1 - close1
            )

            if low1 is not None:
                piercing_line = (
                    close1 < open1
                    and open_ < low1
                    and close > close1 + ((open1 - close1) / 2)
                    and close < open1
                )

            bullish_kicker = open1 > close1 and open_ >= open1 and close > open_
            bearish_kicker = open1 < close1 and open_ <= open1 and close <= open_

            dark_cloud_cover = (
                close1 > open1
                and ((close1 + open1) / 2) > close
                and open_ > close
                and open_ > close1
                and close > open1
                and (open_ - close) / denom > 0.6
            )

        hammer = (
            (range_ > 3 * (open_ - close))
            and ((close - low) / denom > 0.6)
            and ((open_ - low) / denom > 0.6)
        )
        inverted_hammer = (
            (range_ > 3 * (open_ - close))
            and ((high - close) / denom > 0.6)
            and ((high - open_) / denom > 0.6)
        )

        if high1 is not None and high2 is not None:
            hanging_man = (
                (range_ > 4 * (open_ - close))
                and ((close - low) / denom >= 0.75)
                and ((open_ - low) / denom >= 0.75)
                and high1 < open_
                and high2 < open_
            )

        lower = rolling_extreme(lows, idx, 10, 1, min)
        bullish_belt = False
        if (
            low == open_
            and lower is not None
            and open_ < lower
            and open_ < close
            and high1 is not None
            and low1 is not None
        ):
            bullish_belt = close > ((high1 - low1) / 2) + low1

        signals.append(
            CandlePatternSignals(
                index=idx,
                timestamp=candle.open_time,
                doji=doji,
                evening_star=evening_star,
                morning_star=morning_star,
                shooting_star=shooting_star,
                hammer=hammer,
                inverted_hammer=inverted_hammer,
                bearish_harami=bearish_harami,
                bullish_harami=bullish_harami,
                bearish_engulfing=bearish_engulfing,
                bullish_engulfing=bullish_engulfing,
                piercing_line=piercing_line,
                bullish_belt=bool(bullish_belt),
                bullish_kicker=bullish_kicker,
                bearish_kicker=bearish_kicker,
                hanging_man=hanging_man,
                dark_cloud_cover=dark_cloud_cover,
            )
        )

    return signals

