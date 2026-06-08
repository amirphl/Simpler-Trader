from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from backtest.base import BacktestContext, BacktestRunConfig
from backtest.ema_avwap_pullback_strategy import (
    EmaAvwapPullbackStrategy,
    EmaAvwapPullbackStrategyConfig,
    _AvwapSnapshot,
    _PositionState,
)
from candle_downloader.models import Candle


def _candle(
    *,
    offset: int,
    open: float,
    high: float,
    low: float,
    close: float,
    volume: float = 100.0,
) -> Candle:
    open_time = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=offset)
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
        volume=volume,
    )


def _context(candles: list[Candle]) -> BacktestContext:
    return BacktestContext(
        config=BacktestRunConfig(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 3, tzinfo=timezone.utc),
        ),
        data={"ETHUSDT": {"1h": candles}},
        ignore_candles={"ETHUSDT": {"1h": 0}},
    )


class EmaAvwapPullbackStrategyTests(unittest.TestCase):
    def test_waiting_setup_ignores_new_same_direction_setup_by_default(self) -> None:
        candles = [
            _candle(offset=0, open=15.0, high=15.2, low=13.6, close=14.0),
            _candle(offset=1, open=12.0, high=12.1, low=10.8, close=11.0),
            _candle(offset=2, open=10.9, high=11.0, low=10.7, close=10.95),
            _candle(offset=3, open=10.4, high=10.5, low=9.8, close=10.0),
            _candle(offset=4, open=10.1, high=11.6, low=10.0, close=11.4),
        ]
        strategy = EmaAvwapPullbackStrategy(
            EmaAvwapPullbackStrategyConfig(
                symbol="ETHUSDT",
                timeframe="1h",
                initial_equity=10_000.0,
                leverage=1.0,
                equity_risk_pct=1.0,
                ema_length=2,
                consecutive_count=1,
                maker_fee_pct=0.0,
                taker_fee_pct=0.0,
                setup_waiting_replacement_mode="keep_waiting",
            )
        )

        trades, stats = strategy.run(_context(candles))

        self.assertEqual(len(trades), 1)
        self.assertEqual(stats["setups_kept_waiting_short"], 1)
        ignored_event = next(
            event
            for event in stats["decision_log"]
            if event["event"] == "setup_detected_ignored"
        )
        entry_event = next(
            event for event in stats["decision_log"] if event["event"] == "entry_triggered"
        )

        self.assertEqual(ignored_event["waiting_anchor_index"], 1)
        self.assertEqual(ignored_event["ignored_anchor_index"], 3)
        self.assertEqual(entry_event["anchor_index"], 1)

    def test_waiting_setup_can_be_replaced_by_new_same_direction_setup(self) -> None:
        candles = [
            _candle(offset=0, open=15.0, high=15.2, low=13.6, close=14.0),
            _candle(offset=1, open=12.0, high=12.1, low=10.8, close=11.0),
            _candle(offset=2, open=10.9, high=11.0, low=10.7, close=10.95),
            _candle(offset=3, open=10.4, high=10.5, low=9.8, close=10.0),
            _candle(offset=4, open=10.1, high=11.6, low=10.0, close=11.4),
        ]
        strategy = EmaAvwapPullbackStrategy(
            EmaAvwapPullbackStrategyConfig(
                symbol="ETHUSDT",
                timeframe="1h",
                initial_equity=10_000.0,
                leverage=1.0,
                equity_risk_pct=1.0,
                ema_length=2,
                consecutive_count=1,
                maker_fee_pct=0.0,
                taker_fee_pct=0.0,
                setup_waiting_replacement_mode="replace_waiting",
            )
        )

        trades, stats = strategy.run(_context(candles))

        self.assertEqual(len(trades), 1)
        self.assertEqual(stats["waiting_setups_replaced_short"], 1)
        entry_event = next(
            event for event in stats["decision_log"] if event["event"] == "entry_triggered"
        )

        self.assertEqual(entry_event["anchor_index"], 3)

    def test_setup_waits_for_later_pullback_cross_instead_of_invalidating(self) -> None:
        candles = [
            _candle(offset=0, open=14.0, high=14.2, low=12.8, close=13.0),
            _candle(offset=1, open=11.8, high=12.0, low=10.8, close=11.0),
            _candle(offset=2, open=10.7, high=11.2, low=10.6, close=11.1),
            _candle(offset=3, open=11.3, high=12.5, low=11.2, close=12.2),
            _candle(offset=4, open=12.1, high=12.4, low=11.9, close=12.0),
        ]
        strategy = EmaAvwapPullbackStrategy(
            EmaAvwapPullbackStrategyConfig(
                symbol="ETHUSDT",
                timeframe="1h",
                initial_equity=10_000.0,
                leverage=1.0,
                equity_risk_pct=1.0,
                ema_length=2,
                consecutive_count=2,
                maker_fee_pct=0.0,
                taker_fee_pct=0.0,
            )
        )

        trades, stats = strategy.run(_context(candles))

        self.assertEqual(len(trades), 1)
        waiting_event = next(
            event for event in stats["decision_log"] if event["event"] == "setup_waiting"
        )
        entry_event = next(
            event for event in stats["decision_log"] if event["event"] == "entry_triggered"
        )

        self.assertEqual(waiting_event["candle_index"], 2)
        self.assertEqual(entry_event["candle_index"], 3)
        self.assertFalse(
            any(event["event"] == "setup_invalidated" for event in stats["decision_log"])
        )

    def test_restarts_scan_on_same_exit_candle(self) -> None:
        candles = [
            _candle(offset=0, open=9.0, high=10.2, low=8.8, close=10.0),
            _candle(offset=1, open=12.0, high=13.2, low=11.8, close=13.0),
            _candle(offset=2, open=13.0, high=13.1, low=10.8, close=10.9),
            _candle(offset=3, open=13.2, high=14.0, low=13.0, close=13.8),
            _candle(offset=4, open=13.8, high=13.9, low=12.7, close=12.9),
            _candle(offset=5, open=12.9, high=15.0, low=12.8, close=14.8),
        ]
        strategy = EmaAvwapPullbackStrategy(
            EmaAvwapPullbackStrategyConfig(
                symbol="ETHUSDT",
                timeframe="1h",
                initial_equity=10_000.0,
                leverage=1.0,
                equity_risk_pct=1.0,
                ema_length=2,
                consecutive_count=1,
                trailing_gap_pct=1.0,
                maker_fee_pct=0.0,
                taker_fee_pct=0.0,
            )
        )

        trades, stats = strategy.run(_context(candles))

        self.assertEqual(len(trades), 2)
        self.assertEqual(trades[0].notes, "Trailing stop")
        self.assertEqual(trades[1].notes, "Stop loss")

        closed_event = next(
            event
            for event in stats["decision_log"]
            if event["event"] == "position_closed" and event["candle_index"] == 3
        )
        restart_setup = next(
            event
            for event in stats["decision_log"]
            if event["event"] == "setup_detected" and event["candle_index"] == 3
        )

        self.assertEqual(closed_event["exit_reason"], "Trailing stop")
        self.assertEqual(restart_setup["setup_type"], "long")
        self.assertEqual(restart_setup["anchor_index"], 3)

    def test_zero_trailing_gap_pct_is_allowed(self) -> None:
        candles = [
            _candle(offset=0, open=9.0, high=10.2, low=8.8, close=10.0),
            _candle(offset=1, open=12.0, high=13.2, low=11.8, close=13.0),
            _candle(offset=2, open=13.0, high=13.1, low=10.8, close=10.9),
            _candle(offset=3, open=13.2, high=14.0, low=13.0, close=13.8),
            _candle(offset=4, open=13.8, high=13.9, low=12.7, close=12.9),
        ]
        strategy = EmaAvwapPullbackStrategy(
            EmaAvwapPullbackStrategyConfig(
                symbol="ETHUSDT",
                timeframe="1h",
                initial_equity=10_000.0,
                leverage=1.0,
                equity_risk_pct=1.0,
                ema_length=2,
                consecutive_count=1,
                trailing_gap_pct=0.0,
                maker_fee_pct=0.0,
                taker_fee_pct=0.0,
            )
        )

        trades, stats = strategy.run(_context(candles))

        self.assertGreaterEqual(len(trades), 1)
        trailing_event = next(
            event
            for event in stats["decision_log"]
            if event["event"] == "trailing_activated"
        )
        self.assertAlmostEqual(
            trailing_event["trailing_stop"], trailing_event["extreme_price"]
        )

    def test_long_entry_records_exact_vwap_intersection(self) -> None:
        candles = [
            _candle(offset=0, open=9.0, high=10.2, low=8.8, close=10.0),
            _candle(offset=1, open=12.0, high=13.2, low=11.8, close=13.0),
            _candle(offset=2, open=13.0, high=13.1, low=10.8, close=10.9),
            _candle(offset=3, open=13.0, high=13.1, low=10.8, close=10.9),
        ]
        strategy = EmaAvwapPullbackStrategy(
            EmaAvwapPullbackStrategyConfig(
                symbol="ETHUSDT",
                timeframe="1h",
                initial_equity=10_000.0,
                leverage=1.0,
                equity_risk_pct=1.0,
                ema_length=2,
                consecutive_count=2,
                maker_fee_pct=0.0,
                taker_fee_pct=0.0,
            )
        )

        trades, stats = strategy.run(_context(candles))

        self.assertEqual(len(trades), 1)
        entry_event = next(
            event for event in stats["decision_log"] if event["event"] == "entry_triggered"
        )
        trade = trades[0]

        self.assertAlmostEqual(
            entry_event["entry_intersection_price"],
            entry_event["vwap_middle_line"],
        )
        self.assertAlmostEqual(
            float(trade.metadata["entry_raw_price"]),  # type: ignore[index]
            entry_event["entry_intersection_price"],
        )

    def test_price_based_sizing_uses_current_price_without_leverage_multiplier(
        self,
    ) -> None:
        candles = [
            _candle(offset=0, open=9.0, high=10.2, low=8.8, close=10.0),
            _candle(offset=1, open=12.0, high=13.2, low=11.8, close=13.0),
            _candle(offset=2, open=13.0, high=13.1, low=10.8, close=10.9),
            _candle(offset=3, open=13.0, high=13.1, low=10.8, close=10.9),
        ]
        strategy = EmaAvwapPullbackStrategy(
            EmaAvwapPullbackStrategyConfig(
                symbol="ETHUSDT",
                timeframe="1h",
                initial_equity=10_000.0,
                leverage=7.0,
                equity_risk_pct=1.0,
                ema_length=2,
                consecutive_count=2,
                position_sizing_mode="risk_amount_per_price",
                maker_fee_pct=0.001,
                taker_fee_pct=0.002,
                entry_slippage_pct=0.01,
                exit_slippage_pct=0.02,
            )
        )

        trades, stats = strategy.run(_context(candles))

        self.assertEqual(len(trades), 1)
        entry_event = next(
            event for event in stats["decision_log"] if event["event"] == "entry_triggered"
        )

        risk_amount = entry_event["risk_amount"]
        raw_entry = entry_event["entry_intersection_price"]
        entry_price = raw_entry * 1.01
        estimated_exit_price = raw_entry * 0.98
        expected_base_qty = risk_amount / raw_entry
        expected_total_cost_per_unit = (
            (entry_price - raw_entry)
            + (raw_entry - estimated_exit_price)
            + (entry_price * 0.001)
            + (estimated_exit_price * 0.002)
        )
        expected_qty = risk_amount / (raw_entry + expected_total_cost_per_unit)

        self.assertAlmostEqual(entry_event["base_position_qty_before_costs"], expected_base_qty)
        self.assertAlmostEqual(entry_event["estimated_total_cost_per_unit"], expected_total_cost_per_unit)
        self.assertAlmostEqual(entry_event["position_qty"], expected_qty)
        self.assertAlmostEqual(
            entry_event["qty_reduction_from_costs"],
            expected_base_qty - expected_qty,
        )
        self.assertEqual(entry_event["position_sizing_mode"], "risk_amount_per_price")
        self.assertEqual(
            entry_event["risk_amount_interpretation"], "position_notional_budget"
        )

    def test_position_closed_logs_net_pnl_percent(self) -> None:
        candles = [
            _candle(offset=0, open=14.0, high=14.2, low=12.8, close=13.0),
            _candle(offset=1, open=11.8, high=12.0, low=10.8, close=11.0),
            _candle(offset=2, open=10.9, high=12.4, low=10.7, close=12.2),
            _candle(offset=3, open=12.1, high=14.5, low=12.0, close=13.0),
            _candle(offset=4, open=13.0, high=13.2, low=12.7, close=12.9),
        ]
        strategy = EmaAvwapPullbackStrategy(
            EmaAvwapPullbackStrategyConfig(
                symbol="ETHUSDT",
                timeframe="1h",
                initial_equity=10_000.0,
                leverage=1.0,
                equity_risk_pct=1.0,
                ema_length=2,
                consecutive_count=2,
                maker_fee_pct=0.0,
                taker_fee_pct=0.0,
            )
        )

        trades, stats = strategy.run(_context(candles))

        self.assertEqual(len(trades), 1)
        trade = trades[0]
        exit_event = next(
            event for event in stats["decision_log"] if event["event"] == "position_closed"
        )
        entry_notional = float(trade.metadata["entry_price"]) * float(trade.metadata["qty"])  # type: ignore[arg-type,index]
        expected_position_pnl_pct = (trade.pnl / entry_notional) * 100.0

        self.assertAlmostEqual(exit_event["position_pnl_pct"], expected_position_pnl_pct)
        self.assertAlmostEqual(
            float(trade.metadata["position_pnl_pct"]),  # type: ignore[arg-type,index]
            expected_position_pnl_pct,
        )
        self.assertIn("price_return_pct", exit_event)

    def test_rigid_stop_loss_uses_fixed_entry_price_and_exits_immediately(self) -> None:
        candles = [
            _candle(offset=0, open=9.0, high=10.2, low=8.8, close=10.0),
            _candle(offset=1, open=12.0, high=13.2, low=11.8, close=13.0),
            _candle(offset=2, open=13.0, high=13.1, low=10.8, close=10.9),
            _candle(offset=3, open=11.15, high=11.3, low=10.2, close=10.4),
        ]
        strategy = EmaAvwapPullbackStrategy(
            EmaAvwapPullbackStrategyConfig(
                symbol="ETHUSDT",
                timeframe="1h",
                initial_equity=10_000.0,
                leverage=1.0,
                equity_risk_pct=1.0,
                ema_length=2,
                consecutive_count=2,
                rigid_stop_loss_pct=1.0,
                maker_fee_pct=0.0,
                taker_fee_pct=0.0,
            )
        )

        trades, stats = strategy.run(_context(candles))

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].notes, "Rigid stop loss")
        self.assertEqual(stats["rigid_stop_exits"], 1)

        entry_event = next(
            event for event in stats["decision_log"] if event["event"] == "entry_triggered"
        )
        exit_event = next(
            event for event in stats["decision_log"] if event["event"] == "position_closed"
        )
        expected_rigid_level = entry_event["executed_entry_price"] * 0.99

        self.assertAlmostEqual(entry_event["rigid_stop_loss_level"], expected_rigid_level)
        self.assertAlmostEqual(exit_event["rigid_stop_loss_level"], expected_rigid_level)
        self.assertAlmostEqual(exit_event["raw_exit_price"], expected_rigid_level)
        self.assertAlmostEqual(
            float(trades[0].metadata["rigid_stop_level_at_entry"]),  # type: ignore[arg-type,index]
            expected_rigid_level,
        )

    def test_short_open_already_beyond_rigid_stop_exits_before_trailing(self) -> None:
        strategy = EmaAvwapPullbackStrategy(
            EmaAvwapPullbackStrategyConfig(
                symbol="ETHUSDT",
                timeframe="1h",
                initial_equity=10_000.0,
                leverage=1.0,
                equity_risk_pct=1.0,
                ema_length=2,
                consecutive_count=1,
                rigid_stop_loss_pct=2.0,
                maker_fee_pct=0.0,
                taker_fee_pct=0.0,
            )
        )
        entry_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        position = _PositionState(
            direction="short",
            anchor_index=0,
            setup_detected_index=0,
            setup_detected_time=entry_time,
            entry_time=entry_time,
            entry_index=0,
            raw_entry_price=0.5380697249397747,
            entry_price=0.5380697249397747,
            qty=1.0,
            risk_amount=100.0,
            risk_amount_interpretation="position_notional_budget",
            entry_fee=0.0,
            stop_level_at_entry=0.5652344648652496,
            rigid_stop_level_at_entry=0.5488311194385702,
            trailing_activation_level_at_entry=0.529014811631283,
            entry_trigger_mode="intrabar",
            position_sizing_mode="risk_amount_per_price",
            trailing_active=True,
            trailing_stop=0.563,
            extreme_price=0.5574257425742574,
        )
        candle = _candle(
            offset=1,
            open=0.55,
            high=0.564,
            low=0.55,
            close=0.562,
        )
        avwap = _AvwapSnapshot(
            anchor_index=0,
            anchor_time=entry_time,
            candle_index=1,
            vwap=0.59,
            stdev=0.02,
            upper1=0.61,
            lower1=0.57,
            upper2=0.64,
            lower2=0.55,
            upper3=0.65,
            lower3=0.53,
        )
        stats = {"trailing_activations": 0, "trailing_updates": 0}
        decision = strategy._process_position_for_candle(
            position=position,
            candle=candle,
            candle_index=1,
            prev_close=0.55,
            avwap=avwap,
            stats=stats,
            decision_log=[],
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.reason, "Rigid stop loss")  # type: ignore[union-attr]
        self.assertAlmostEqual(
            decision.raw_exit_price,  # type: ignore[union-attr]
            position.rigid_stop_level_at_entry,
        )

    def test_short_stop_loss_uses_current_upper_band_two(self) -> None:
        candles = [
            _candle(offset=0, open=14.0, high=14.2, low=12.8, close=13.0),
            _candle(offset=1, open=11.8, high=12.0, low=10.8, close=11.0),
            _candle(offset=2, open=10.9, high=12.4, low=10.7, close=12.2),
            _candle(offset=3, open=12.1, high=14.5, low=12.0, close=13.0),
            _candle(offset=4, open=13.0, high=13.2, low=12.7, close=12.9),
        ]
        strategy = EmaAvwapPullbackStrategy(
            EmaAvwapPullbackStrategyConfig(
                symbol="ETHUSDT",
                timeframe="1h",
                initial_equity=10_000.0,
                leverage=1.0,
                equity_risk_pct=1.0,
                ema_length=2,
                consecutive_count=2,
                maker_fee_pct=0.0,
                taker_fee_pct=0.0,
            )
        )

        trades, stats = strategy.run(_context(candles))

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].notes, "Stop loss")

        exit_event = next(
            event for event in stats["decision_log"] if event["event"] == "position_closed"
        )
        self.assertEqual(exit_event["setup_type"], "short")
        self.assertAlmostEqual(exit_event["raw_exit_price"], exit_event["stop_loss_level"])
        self.assertAlmostEqual(
            float(trades[0].metadata["exit_raw_price"]),  # type: ignore[index]
            exit_event["raw_exit_price"],
        )


if __name__ == "__main__":
    unittest.main()
