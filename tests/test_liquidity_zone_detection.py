from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from candle_downloader.models import Candle
from experiments.liquidity_zone_detection import (
    LiquidityZoneConfig,
    PriceRange,
    _build_zone,
    _v2_entries_to_pivots,
)
from experiments.pivot_detection import Pivot
from experiments.pivot_detection_v2 import PivotV2Entry


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


class LiquidityZoneDetectionTests(unittest.TestCase):
    def test_v2_adapter_marks_later_swept_pivots_as_hunted(self) -> None:
        candles = [
            _candle(0, open=10.0, high=12.0, low=9.5, close=11.0),
            _candle(1, open=11.0, high=11.5, low=9.0, close=10.0),
            _candle(2, open=10.0, high=12.5, low=9.8, close=11.5),
        ]
        entries = [
            PivotV2Entry(
                candle_index=1,
                pivot_index=0,
                pivot_type="bearish",
                hunt_index=1,
            )
        ]

        pivots = _v2_entries_to_pivots(entries, candles)

        self.assertEqual(len(pivots), 1)
        self.assertTrue(pivots[0].hunted)

    def test_zone_line_uses_fixed_direction_specific_methods(self) -> None:
        candles = [
            _candle(0, open=10.0, high=12.0, low=9.0, close=11.0),
            _candle(1, open=11.0, high=15.0, low=8.0, close=12.0),
            _candle(2, open=12.0, high=14.0, low=10.0, close=13.0),
            _candle(3, open=12.0, high=13.5, low=9.5, close=12.5),
        ]
        older = Pivot(
            index=0,
            type="bullish",
            high=12.0,
            low=9.0,
            reference_index=0,
            trigger_index=0,
            invalidation_index=None,
        )
        newer = Pivot(
            index=2,
            type="bullish",
            high=14.0,
            low=10.0,
            reference_index=2,
            trigger_index=2,
            invalidation_index=None,
        )
        config = LiquidityZoneConfig()

        bullish_zone = _build_zone(
            zone_id="bull",
            level=1,
            direction="UPWARD",
            older=older,
            newer=newer,
            overlap=PriceRange(low=10.0, high=11.0),
            candles=candles,
            config=config,
        )
        bearish_zone = _build_zone(
            zone_id="bear",
            level=1,
            direction="DOWNWARD",
            older=older,
            newer=newer,
            overlap=PriceRange(low=10.0, high=11.0),
            candles=candles,
            config=config,
        )

        self.assertEqual(bullish_zone.line_method, "pivot_low_min")
        self.assertEqual(bullish_zone.line_price, 9.0)
        self.assertEqual(bullish_zone.metadata["line_price"], 9.0)
        self.assertFalse(bullish_zone.is_hunted)
        self.assertEqual(bearish_zone.line_method, "pivot_high_max")
        self.assertEqual(bearish_zone.line_price, 14.0)
        self.assertFalse(bearish_zone.is_hunted)


if __name__ == "__main__":
    unittest.main()
