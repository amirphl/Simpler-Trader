from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from candle_downloader.models import Candle
from experiments.bos_choch_detection import BOSCHoCHResult, DirectionState
from experiments.liquidity_zone_detection import (
    LiquidityZone,
    LiquidityZoneResult,
    PriceRange,
)
from experiments.pivot_detection import Pivot
from experiments.scenario_detection import (
    ScenarioDetectionConfig,
    detect_scenarios,
)


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


def _pivot(index: int, pivot_type: str, *, high: float, low: float) -> Pivot:
    return Pivot(
        index=index,
        type=pivot_type,
        high=high,
        low=low,
        reference_index=index,
        trigger_index=index,
        invalidation_index=None,
    )


def _zone(direction: str) -> LiquidityZone:
    pivot_type = "bullish" if direction == "UPWARD" else "bearish"
    line_price = 94.0 if direction == "UPWARD" else 106.0
    price_range = (
        PriceRange(low=93.0, high=95.0)
        if direction == "UPWARD"
        else PriceRange(low=105.0, high=107.0)
    )
    return LiquidityZone(
        id=f"{direction}_ZONE",
        direction=direction,  # type: ignore[arg-type]
        level=1,
        left_pivot=_pivot(2, pivot_type, high=106.0, low=94.0),
        right_pivot=_pivot(3, pivot_type, high=106.0, low=94.0),
        price_range=price_range,
        line_price=line_price,
        line_method="pivot_low_min" if direction == "UPWARD" else "pivot_high_max",
        start_index=2,
        end_index=3,
        is_hunted=False,
        metadata={},
    )


def _liquidity_result(zone: LiquidityZone, pivots: list[Pivot]):
    all_pivots = [zone.left_pivot, zone.right_pivot, *pivots]
    return LiquidityZoneResult(
        pivots=all_pivots,
        bos_choch_result=BOSCHoCHResult(
            direction_state=DirectionState(direction="UPWARD", since_index=0),
        ),
        direction_segments=[],
        zones_by_level={
            1: {
                "UPWARD": [zone] if zone.direction == "UPWARD" else [],
                "DOWNWARD": [zone] if zone.direction == "DOWNWARD" else [],
            },
            2: {"UPWARD": [], "DOWNWARD": []},
        },
    )


class ScenarioDetectionTests(unittest.TestCase):
    def test_bullish_scenario_confirms_when_bearish_pivot_is_hunted_first(
        self,
    ) -> None:
        candles = [
            _candle(0, open=91.0, high=92.0, low=90.0, close=91.0),
            _candle(1, open=106.0, high=107.0, low=104.0, close=105.0),
            _candle(2, open=95.0, high=106.0, low=94.0, close=105.0),
            _candle(3, open=95.0, high=106.0, low=94.0, close=105.0),
            _candle(4, open=96.0, high=97.0, low=93.8, close=95.0),
            _candle(5, open=104.0, high=108.0, low=103.0, close=107.0),
        ]
        zone = _zone("UPWARD")
        pivots = [
            _pivot(0, "bullish", high=92.0, low=90.0),
            _pivot(1, "bearish", high=107.0, low=104.0),
        ]

        result = detect_scenarios(
            candles,
            ScenarioDetectionConfig(),
            liquidity_zone_result=_liquidity_result(zone, pivots),
            current_index=5,
        )

        self.assertEqual(len(result.scenarios), 1)
        scenario = result.scenarios[0]
        self.assertEqual(scenario.direction, "BULLISH")
        self.assertEqual(scenario.zone_hunt.candle_index, 4)
        self.assertEqual(scenario.zone_hunt.side, "from_above")
        self.assertEqual(scenario.confirming_pivot_hunt.pivot_type, "bearish")
        self.assertEqual(scenario.confirming_pivot_hunt.candle_index, 5)
        self.assertIsNone(scenario.cancelling_pivot_hunt)
        self.assertTrue(zone.is_hunted)
        self.assertEqual(zone.metadata["scenario_decision"], "approved")
        self.assertEqual(
            scenario.metadata["config"]["bullish_pivot_selection"],
            "non_hunted_only",
        )

    def test_bullish_scenario_cancels_when_bullish_pivot_is_hunted_first(
        self,
    ) -> None:
        candles = [
            _candle(0, open=91.0, high=92.0, low=90.0, close=91.0),
            _candle(1, open=106.0, high=107.0, low=104.0, close=105.0),
            _candle(2, open=95.0, high=106.0, low=94.0, close=105.0),
            _candle(3, open=95.0, high=106.0, low=94.0, close=105.0),
            _candle(4, open=96.0, high=97.0, low=93.8, close=95.0),
            _candle(5, open=91.0, high=104.0, low=89.0, close=90.0),
            _candle(6, open=104.0, high=108.0, low=103.0, close=107.0),
        ]
        zone = _zone("UPWARD")
        pivots = [
            _pivot(0, "bullish", high=92.0, low=90.0),
            _pivot(1, "bearish", high=107.0, low=104.0),
        ]

        result = detect_scenarios(
            candles,
            ScenarioDetectionConfig(),
            liquidity_zone_result=_liquidity_result(zone, pivots),
        )

        self.assertEqual(result.scenarios, [])
        self.assertTrue(zone.is_hunted)
        self.assertEqual(
            zone.metadata["scenario_decision"],
            "cancelled_by_relevant_pivot",
        )

    def test_pivot_selection_mode_is_applied_at_zone_hunt_time(self) -> None:
        candles = [
            _candle(0, open=91.0, high=92.0, low=90.0, close=91.0),
            _candle(1, open=106.0, high=107.0, low=104.0, close=105.0),
            _candle(2, open=89.0, high=91.0, low=88.0, close=90.0),
            _candle(3, open=95.0, high=106.0, low=94.0, close=105.0),
            _candle(4, open=96.0, high=97.0, low=93.8, close=95.0),
            _candle(5, open=104.0, high=108.0, low=103.0, close=107.0),
        ]
        zone_default = _zone("UPWARD")
        zone_ignore = _zone("UPWARD")
        pivots = [
            _pivot(0, "bullish", high=92.0, low=90.0),
            _pivot(1, "bearish", high=107.0, low=104.0),
        ]

        default_result = detect_scenarios(
            candles,
            ScenarioDetectionConfig(),
            liquidity_zone_result=_liquidity_result(zone_default, pivots),
        )
        ignore_result = detect_scenarios(
            candles,
            ScenarioDetectionConfig(
                bullish_pivot_selection="ignore_hunted_status",
            ),
            liquidity_zone_result=_liquidity_result(zone_ignore, pivots),
        )

        default_scenario = default_result.scenarios[0]
        ignore_scenario = ignore_result.scenarios[0]
        self.assertIsNone(default_scenario.relevant_bullish_pivot)
        self.assertIsNotNone(ignore_scenario.relevant_bullish_pivot)
        self.assertTrue(ignore_scenario.relevant_bullish_pivot.hunted_at_zone_hunt)

    def test_bearish_scenario_confirms_when_bullish_pivot_is_hunted_first(
        self,
    ) -> None:
        candles = [
            _candle(0, open=109.0, high=110.0, low=108.0, close=109.0),
            _candle(1, open=98.0, high=100.0, low=95.0, close=99.0),
            _candle(2, open=105.0, high=106.0, low=95.0, close=96.0),
            _candle(3, open=105.0, high=106.0, low=95.0, close=96.0),
            _candle(4, open=104.0, high=106.5, low=103.0, close=105.0),
            _candle(5, open=96.0, high=99.0, low=94.0, close=95.0),
        ]
        zone = _zone("DOWNWARD")
        pivots = [
            _pivot(0, "bearish", high=110.0, low=108.0),
            _pivot(1, "bullish", high=100.0, low=95.0),
        ]

        result = detect_scenarios(
            candles,
            ScenarioDetectionConfig(),
            liquidity_zone_result=_liquidity_result(zone, pivots),
            current_index=5,
        )

        self.assertEqual(len(result.scenarios), 1)
        scenario = result.scenarios[0]
        self.assertEqual(scenario.direction, "BEARISH")
        self.assertEqual(scenario.zone_hunt.candle_index, 4)
        self.assertEqual(scenario.zone_hunt.side, "from_below")
        self.assertEqual(scenario.confirming_pivot_hunt.pivot_type, "bullish")
        self.assertEqual(scenario.confirming_pivot_hunt.candle_index, 5)
        self.assertEqual(zone.metadata["scenario_decision"], "approved")


if __name__ == "__main__":
    unittest.main()
