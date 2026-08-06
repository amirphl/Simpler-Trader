from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from candle_downloader.models import Candle
from experiments.bos_choch_detection import detect_bos_choch


def _candle(
    index: int, *, open: float, high: float, low: float, close: float
) -> Candle:
    open_time = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=index)
    return Candle(
        symbol="ETHUSDT",
        interval="1h",
        open_time=open_time,
        close_time=open_time + timedelta(hours=1),
        open=open,
        high=high,
        low=low,
        close=close,
        volume=1.0,
    )


class BOSCHoCHStorageTests(unittest.TestCase):
    def test_choch_retains_only_its_first_hunt_while_live_level_cascades(self) -> None:
        candles = [
            _candle(0, open=9.0, high=10.0, low=8.0, close=10.0),
            _candle(1, open=10.0, high=11.0, low=9.0, close=11.0),
            _candle(2, open=11.0, high=12.0, low=10.0, close=12.0),
            _candle(3, open=12.0, high=12.5, low=10.5, close=11.0),
            _candle(4, open=11.0, high=13.0, low=11.0, close=13.0),
            _candle(5, open=13.0, high=14.0, low=11.0, close=12.0),
            _candle(6, open=12.0, high=13.0, low=9.0, close=13.0),
            _candle(7, open=13.0, high=15.0, low=12.0, close=15.0),
            _candle(8, open=15.0, high=15.5, low=10.0, close=11.0),
            _candle(9, open=11.0, high=11.0, low=9.5, close=10.0),
        ]

        result = detect_bos_choch(candles)

        self.assertEqual(len(result.bos_records), 2)
        self.assertEqual(len(result.choch_records_by_bos), 2)

        first_choch = result.choch_records_by_bos[0]
        self.assertEqual(first_choch.candle_index, 3)
        self.assertEqual(first_choch.level, 10.5)
        self.assertEqual(first_choch.first_hunt_candle_index, 8)
        self.assertEqual(first_choch.current_level, 9.5)

        # The repeated hunt advances the live level, but it does not retain a
        # second hunt event or another per-BOS history entry.
        hunts = [event for event in result.events if event.event == "CHOCH_HUNT"]
        self.assertEqual([event.candle_index for event in hunts], [8])


if __name__ == "__main__":
    unittest.main()
