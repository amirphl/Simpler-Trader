from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from candle_downloader.models import Candle
from experiments.pivot_detection import PivotConfig, detect_pivots, get_candles
from experiments.pivot_detection_v2 import detect_pivots_v2


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

    def test_v1_include_reference_candle_flag_controls_candidate_start(self) -> None:
        candles = [
            _candle(0, open=10.0, high=12.0, low=9.0, close=9.5),
            _candle(1, open=9.5, high=10.5, low=9.2, close=10.2),
            _candle(2, open=10.2, high=11.0, low=8.8, close=9.0),
        ]

        include_ref = detect_pivots(
            candles, 10, PivotConfig(include_reference_candle=True)
        )
        exclude_ref = detect_pivots(
            candles, 10, PivotConfig(include_reference_candle=False)
        )

        self.assertEqual(include_ref[0].index, 0)
        self.assertEqual(exclude_ref[0].index, 1)

    def test_v2_pivot_uses_full_formation_range(self) -> None:
        candles = [
            _candle(0, open=10.0, high=20.0, low=10.0, close=12.0),
            _candle(1, open=12.0, high=13.0, low=9.5, close=11.0),
            _candle(2, open=11.0, high=14.0, low=9.8, close=12.5),
            _candle(3, open=9.5, high=20.0, low=9.0, close=18.0),
        ]

        entries = detect_pivots_v2(candles, 10)
        pivots = [entry for entry in entries if entry.pivot_index is not None]

        self.assertEqual(len(pivots), 1)
        self.assertEqual(pivots[0].pivot_type, "bearish")
        self.assertEqual(pivots[0].pivot_index, 0)

    def test_v2_bullish_pivot_hunted_by_crossing_down_from_above(self) -> None:
        candles = [
            _candle(0, open=20.0, high=21.0, low=10.0, close=18.0),
            _candle(1, open=18.0, high=20.0, low=11.0, close=19.0),
            _candle(2, open=19.0, high=20.0, low=9.0, close=12.0),
            _candle(3, open=21.0, high=22.0, low=9.0, close=10.0),
        ]

        entries = detect_pivots_v2(candles, 10)
        pivots = [entry for entry in entries if entry.pivot_index is not None]

        self.assertEqual(len(pivots), 1)
        self.assertEqual(pivots[0].pivot_type, "bullish")
        self.assertEqual(pivots[0].pivot_index, 2)


if __name__ == "__main__":
    unittest.main()
