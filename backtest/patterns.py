from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Sequence

from candle_downloader.models import Candle

_EPSILON = 1e-9

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
    """Boolean flags for each supported candlestick formation."""

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
        return [label for attr, label in _PATTERN_LABELS.items() if getattr(self, attr)]


@dataclass(frozen=True)
class _CandleView:
    open: float
    high: float
    low: float
    close: float

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return max(self.high - self.low, 0.0)

    @property
    def body_low(self) -> float:
        return min(self.open, self.close)

    @property
    def body_high(self) -> float:
        return max(self.open, self.close)

    @property
    def upper_shadow(self) -> float:
        return max(self.high - self.body_high, 0.0)

    @property
    def lower_shadow(self) -> float:
        return max(self.body_low - self.low, 0.0)

    @property
    def midpoint(self) -> float:
        return (self.open + self.close) / 2.0

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    def body_ratio(self) -> float:
        return _ratio(self.body, self.range)

    def upper_shadow_ratio(self) -> float:
        return _ratio(self.upper_shadow, self.range)

    def lower_shadow_ratio(self) -> float:
        return _ratio(self.lower_shadow, self.range)


def detect_candle_patterns(
    candles: Sequence[Candle], doji_size: float = 0.05
) -> List[CandlePatternSignals]:
    """Detect candlestick patterns for each candle in the input sequence."""
    if not candles:
        return []
    if not 0.0 < doji_size <= 1.0:
        raise ValueError("doji_size must be within (0, 1]")

    views = [_CandleView(c.open, c.high, c.low, c.close) for c in candles]
    signals: List[CandlePatternSignals] = []

    for idx, candle in enumerate(candles):
        current = views[idx]
        prev1 = views[idx - 1] if idx >= 1 else None
        prev2 = views[idx - 2] if idx >= 2 else None
        prior_low = _rolling_low(views, end_idx=idx - 1, length=10)

        doji = _is_doji(current, doji_size)
        evening_star = _is_evening_star(prev2, prev1, current)
        morning_star = _is_morning_star(prev2, prev1, current)
        shooting_star = _is_shooting_star(prev1, current)
        hammer = _is_hammer(current)
        inverted_hammer = _is_inverted_hammer(current)
        bearish_harami = _is_bearish_harami(prev1, current)
        bullish_harami = _is_bullish_harami(prev1, current)
        bearish_engulfing = _is_bearish_engulfing(prev1, current)
        bullish_engulfing = _is_bullish_engulfing(prev1, current)
        piercing_line = _is_piercing_line(prev1, current)
        bullish_belt = _is_bullish_belt(prev1, current, prior_low)
        bullish_kicker = _is_bullish_kicker(prev1, current)
        bearish_kicker = _is_bearish_kicker(prev1, current)
        hanging_man = _is_hanging_man(
            prev2=prev2,
            prev1=prev1,
            current=current,
        )
        dark_cloud_cover = _is_dark_cloud_cover(prev1, current)

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
                bullish_belt=bullish_belt,
                bullish_kicker=bullish_kicker,
                bearish_kicker=bearish_kicker,
                hanging_man=hanging_man,
                dark_cloud_cover=dark_cloud_cover,
            )
        )

    return signals


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= _EPSILON:
        return 0.0
    return numerator / denominator


def _nearly_equal(left: float, right: float) -> bool:
    return abs(left - right) <= _EPSILON


def _rolling_low(
    candles: Sequence[_CandleView], *, end_idx: int, length: int
) -> float | None:
    if end_idx < 0 or length <= 0:
        return None
    start_idx = max(0, end_idx - length + 1)
    window = candles[start_idx : end_idx + 1]
    if not window:
        return None
    return min(c.low for c in window)


def _is_doji(candle: _CandleView, doji_size: float) -> bool:
    if candle.range <= _EPSILON:
        return candle.body <= _EPSILON
    return candle.body <= candle.range * doji_size


def _has_small_body(candle: _CandleView, threshold: float = 1 / 3) -> bool:
    return candle.body_ratio() <= threshold


def _is_evening_star(
    first: _CandleView | None,
    second: _CandleView | None,
    third: _CandleView,
) -> bool:
    if first is None or second is None:
        return False
    return (
        first.is_bullish
        and second.body_low > first.body_high
        and third.is_bearish
        and third.open < second.body_low
        and third.close < first.midpoint
    )


def _is_morning_star(
    first: _CandleView | None,
    second: _CandleView | None,
    third: _CandleView,
) -> bool:
    if first is None or second is None:
        return False
    return (
        first.is_bearish
        and second.body_high < first.body_low
        and third.is_bullish
        and third.open > second.body_high
        and third.close > first.midpoint
    )


def _is_shooting_star(previous: _CandleView | None, current: _CandleView) -> bool:
    if previous is None:
        return False
    return (
        previous.is_bullish
        and current.open > previous.close
        and current.upper_shadow >= current.body * 3.0
        and current.lower_shadow <= current.body
    )


def _is_hammer(current: _CandleView) -> bool:
    return (
        _has_small_body(current)
        and current.range > current.body * 3.0
        and current.lower_shadow_ratio() > 0.6
        and current.upper_shadow_ratio() < 0.2
    )


def _is_inverted_hammer(current: _CandleView) -> bool:
    return (
        _has_small_body(current)
        and current.range > current.body * 3.0
        and current.upper_shadow_ratio() > 0.6
        and current.lower_shadow_ratio() < 0.2
    )


def _is_bearish_harami(previous: _CandleView | None, current: _CandleView) -> bool:
    if previous is None:
        return False
    return (
        previous.is_bullish
        and current.is_bearish
        and current.open <= previous.close
        and previous.open <= current.close
        and current.body < previous.body
    )


def _is_bullish_harami(previous: _CandleView | None, current: _CandleView) -> bool:
    if previous is None:
        return False
    return (
        previous.is_bearish
        and current.is_bullish
        and current.close <= previous.open
        and previous.close <= current.open
        and current.body < previous.body
    )


def _is_bearish_engulfing(previous: _CandleView | None, current: _CandleView) -> bool:
    if previous is None:
        return False
    return (
        previous.is_bullish
        and current.is_bearish
        and current.open >= previous.close
        and previous.open >= current.close
        and current.body > previous.body
    )


def _is_bullish_engulfing(previous: _CandleView | None, current: _CandleView) -> bool:
    if previous is None:
        return False
    return (
        previous.is_bearish
        and current.is_bullish
        and current.close >= previous.open
        and previous.close >= current.open
        and current.body > previous.body
    )


def _is_piercing_line(previous: _CandleView | None, current: _CandleView) -> bool:
    if previous is None:
        return False
    return (
        previous.is_bearish
        and current.is_bullish
        and current.open < previous.low
        and current.close > previous.midpoint
        and current.close < previous.open
    )


def _is_bullish_belt(
    previous: _CandleView | None,
    current: _CandleView,
    prior_low: float | None,
) -> bool:
    if previous is None or prior_low is None:
        return False
    return (
        _nearly_equal(current.low, current.open)
        and current.open < prior_low
        and current.is_bullish
        and current.close > previous.midpoint
    )


def _is_bullish_kicker(previous: _CandleView | None, current: _CandleView) -> bool:
    if previous is None:
        return False
    return previous.is_bearish and current.open >= previous.open and current.is_bullish


def _is_bearish_kicker(previous: _CandleView | None, current: _CandleView) -> bool:
    if previous is None:
        return False
    return previous.is_bullish and current.open <= previous.open and current.is_bearish


def _is_hanging_man(
    *,
    prev2: _CandleView | None,
    prev1: _CandleView | None,
    current: _CandleView,
) -> bool:
    if prev1 is None or prev2 is None:
        return False
    return (
        _has_small_body(current, threshold=0.25)
        and current.range > current.body * 4.0
        and current.lower_shadow_ratio() >= 0.75
        and current.upper_shadow_ratio() < 0.2
        and prev1.high < current.open
        and prev2.high < current.open
    )


def _is_dark_cloud_cover(previous: _CandleView | None, current: _CandleView) -> bool:
    if previous is None:
        return False
    return (
        previous.is_bullish
        and current.is_bearish
        and current.open > previous.close
        and current.close < previous.midpoint
        and current.close > previous.open
        and current.body_ratio() > 0.6
    )
