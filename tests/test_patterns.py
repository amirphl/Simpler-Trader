from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from backtest.patterns import detect_candle_patterns
from candle_downloader.models import Candle


def _candle(
    *,
    offset: int,
    open: float,
    high: float,
    low: float,
    close: float,
    volume: float = 1.0,
) -> Candle:
    open_time = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=offset)
    close_time = open_time + timedelta(minutes=1)
    return Candle(
        symbol="BTCUSDT",
        interval="1m",
        open_time=open_time,
        close_time=close_time,
        open=open,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


class DetectCandlePatternsTests(unittest.TestCase):
    def test_accepts_small_positive_doji_size(self) -> None:
        signals = detect_candle_patterns(
            [_candle(offset=0, open=10.0, high=10.0, low=10.0, close=10.0)],
            doji_size=0.001,
        )

        self.assertEqual(len(signals), 1)
        self.assertTrue(signals[0].doji)

    def test_hammer_requires_small_real_body(self) -> None:
        signals = detect_candle_patterns(
            [_candle(offset=0, open=61.0, high=100.0, low=0.0, close=95.0)]
        )

        self.assertFalse(signals[0].hammer)

    def test_inverted_hammer_requires_small_real_body(self) -> None:
        signals = detect_candle_patterns(
            [_candle(offset=0, open=5.0, high=100.0, low=0.0, close=39.0)]
        )

        self.assertFalse(signals[0].inverted_hammer)

    def test_hanging_man_requires_small_real_body(self) -> None:
        signals = detect_candle_patterns(
            [
                _candle(offset=0, open=40.0, high=50.0, low=35.0, close=45.0),
                _candle(offset=1, open=45.0, high=55.0, low=40.0, close=50.0),
                _candle(offset=2, open=61.0, high=100.0, low=0.0, close=95.0),
            ]
        )

        self.assertFalse(signals[2].hanging_man)

    def test_bearish_kicker_requires_bearish_second_candle(self) -> None:
        signals = detect_candle_patterns(
            [
                _candle(offset=0, open=10.0, high=15.0, low=9.0, close=14.0),
                _candle(offset=1, open=9.0, high=10.0, low=8.0, close=9.0),
            ]
        )

        self.assertFalse(signals[1].bearish_kicker)

    def test_bullish_engulfing_matches_strategy_expectation(self) -> None:
        signals = detect_candle_patterns(
            [
                _candle(offset=0, open=10.0, high=10.5, low=7.5, close=8.0),
                _candle(offset=1, open=7.8, high=11.5, low=7.5, close=11.0),
            ]
        )

        self.assertTrue(signals[1].bullish_engulfing)


if __name__ == "__main__":
    unittest.main()
