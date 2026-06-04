from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from candle_downloader.models import Candle
from experiments.pivot_detection import detect_pivots, get_candles


def _candle(
    offset: int,
    *,
    open: float,
    high: float,
    low: float,
    close: float,
) -> Candle:
    open_time = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=offset)
    close_time = open_time + timedelta(hours=1)
    return Candle(
        symbol="ETHUSDT",
        interval="1h",
        open_time=open_time,
        close_time=close_time,
        open=open,
        high=high,
        low=low,
        close=close,
        volume=1.0,
    )


class PivotDetectionTests(unittest.TestCase):
    def test_rejects_non_positive_scan_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "scan_length"):
            detect_pivots([], 0)

    def test_rejects_empty_candle_time_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "start_ms"):
            get_candles(
                source="csv",
                symbol="ETHUSDT",
                interval="1h",
                start_ms=10,
                end_ms=10,
                csv_path="unused.csv",
            )

    def test_bearish_pivot_uses_pre_trigger_swing_high(self) -> None:
        candles = [
            _candle(0, open=9.4, high=10.0, low=9.2, close=9.4),
            _candle(1, open=9.8, high=10.0, low=9.0, close=9.4),
            _candle(2, open=9.4, high=10.2, low=9.3, close=9.9),
            _candle(3, open=9.9, high=10.0, low=8.8, close=9.0),
            _candle(4, open=9.0, high=10.4, low=8.9, close=10.1),
        ]

        pivots = detect_pivots(candles, 10)

        self.assertEqual(len(pivots), 1)
        self.assertEqual(pivots[0].type, "bearish")
        self.assertEqual(pivots[0].index, 2)
        self.assertEqual(pivots[0].reference_index, 1)
        self.assertEqual(pivots[0].trigger_index, 3)
        self.assertTrue(pivots[0].haunted)
        self.assertTrue(pivots[0].hunted)

    def test_bullish_pivot_uses_pre_trigger_swing_low(self) -> None:
        candles = [
            _candle(0, open=10.5, high=10.8, low=10.0, close=10.5),
            _candle(1, open=10.2, high=11.0, low=10.0, close=10.6),
            _candle(2, open=10.6, high=10.7, low=9.8, close=10.1),
            _candle(3, open=10.1, high=11.2, low=10.0, close=11.0),
            _candle(4, open=11.0, high=11.1, low=9.6, close=10.0),
        ]

        pivots = detect_pivots(candles, 10)

        self.assertEqual(len(pivots), 1)
        self.assertEqual(pivots[0].type, "bullish")
        self.assertEqual(pivots[0].index, 2)
        self.assertEqual(pivots[0].reference_index, 1)
        self.assertEqual(pivots[0].trigger_index, 3)
        self.assertTrue(pivots[0].haunted)

    def test_previous_same_color_lookup_stays_inside_scan_window(self) -> None:
        candles = [
            _candle(0, open=9.0, high=100.0, low=8.5, close=8.8),
            _candle(1, open=10.0, high=11.0, low=9.0, close=9.5),
            _candle(2, open=9.5, high=10.2, low=9.1, close=9.9),
            _candle(3, open=9.9, high=10.4, low=8.9, close=9.0),
        ]

        pivots = detect_pivots(candles, 3)

        self.assertEqual(len(pivots), 1)
        self.assertEqual(pivots[0].reference_index, 1)
        self.assertEqual(pivots[0].previous_bearish_index, None)


if __name__ == "__main__":
    unittest.main()
