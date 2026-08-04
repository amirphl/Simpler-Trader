from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from backtest.base import BacktestContext, BacktestRunConfig
from backtest.ema_avwap_pullback_strategy import (
    EmaAvwapPullbackStrategy,
    EmaAvwapPullbackStrategyConfig,
    EntryMode,
    ExitBand,
    ExitMode,
    _AvwapSnapshot,
    _PositionState,
    _SetupState,
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
    return Candle(
        symbol="ETHUSDT",
        interval="1h",
        open_time=open_time,
        close_time=open_time + timedelta(hours=1),
        open=open,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _config(**overrides: object) -> EmaAvwapPullbackStrategyConfig:
    values: dict[str, object] = {
        "symbol": "ETHUSDT",
        "timeframe": "1h",
        "initial_equity": 10_000.0,
        "leverage": 10.0,
        "max_entry_notional_usdt": 1_000.0,
        "max_position_size_pct": 10.0,
        "position_notional_pct": 1.0,
        "ema_length": 2,
        "consecutive_count": 1,
        "max_setup_age_bars": 3,
        "max_entry_deviation_pct": 1.0,
        "position_sizing_mode": "risk_amount_per_price",
        # Most unit cases below exercise the explicit live-OHLC approximation.
        # Keep that distinct from the public backtest default, which is
        # closed-candle mode.
        "entry_mode": EntryMode.LIVE,
        "exit_mode": ExitMode.LIVE,
        "rigid_stop_loss_pct": 3.0,
        "maker_fee_pct": 0.0,
        "taker_fee_pct": 0.0,
    }
    values.update(overrides)
    return EmaAvwapPullbackStrategyConfig(**values)  # type: ignore[arg-type]


def _entry_stats() -> dict[str, object]:
    return {
        "entries_long": 0,
        "entries_short": 0,
        "entries_skipped_invalid_risk": 0,
        "entries_skipped_non_positive_equity": 0,
        "entries_skipped_minimum_balance": 0,
        "entries_skipped_zero_qty": 0,
        "entries_capped_by_live_notional_limits": 0,
        "setups_expired": 0,
        "setups_invalidated_by_ema": 0,
        "setups_discarded_unfavorable_first_observation": 0,
        "entries_skipped_excessive_deviation": 0,
        "entries_skipped_unmarketable": 0,
        "entries_skipped_stop_already_breached": 0,
        "trailing_activations": 0,
        "trailing_updates": 0,
        "total_entry_fees": 0.0,
        "max_margin_required": 0.0,
        "decision_log_truncated_count": 0,
    }


def _pending_setup(
    strategy: EmaAvwapPullbackStrategy,
    *,
    setup: _SetupState,
    candles: tuple[Candle, ...],
    ema_value: float = 100.0,
) -> tuple[_SetupState | None, _PositionState | None, dict[str, object], list[dict]]:
    tpv, volume, tpv2 = strategy._build_avwap_prefixes(candles)  # noqa: SLF001
    stats = _entry_stats()
    decision_log: list[dict] = []
    next_setup, position = strategy._process_pending_setup(  # noqa: SLF001
        setup=setup,
        candle=candles[-1],
        candle_index=len(candles) - 1,
        prev_close=candles[-2].close,
        ema_value=ema_value,
        realized_equity=10_000.0,
        candles=candles,
        tpv_prefix=tpv,
        vol_prefix=volume,
        tpv2_prefix=tpv2,
        stats=stats,
        decision_log=decision_log,
    )
    return next_setup, position, stats, decision_log


def _avwap() -> _AvwapSnapshot:
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return _AvwapSnapshot(
        anchor_index=0,
        anchor_time=timestamp,
        candle_index=1,
        vwap=100.0,
        stdev=2.0,
        upper1=102.0,
        lower1=98.0,
        upper2=104.0,
        lower2=96.0,
        upper3=106.0,
        lower3=94.0,
    )


def _position(
    *,
    direction: str = "long",
    entry_mode: EntryMode = EntryMode.LIVE,
    exit_mode: ExitMode = ExitMode.LIVE,
    exit_band: ExitBand = ExitBand.BAND_1,
) -> _PositionState:
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return _PositionState(
        direction=direction,  # type: ignore[arg-type]
        anchor_index=0,
        setup_detected_index=0,
        setup_detected_time=timestamp,
        entry_time=timestamp,
        entry_index=0,
        raw_entry_price=100.0,
        entry_price=100.0,
        qty=1.0,
        position_notional_budget=100.0,
        entry_fee=0.0,
        stop_level_at_entry=98.0 if direction == "long" else 102.0,
        rigid_stop_level_at_entry=97.0 if direction == "long" else 103.0,
        trailing_activation_level_at_entry=102.0 if direction == "long" else 98.0,
        entry_trigger_mode=(
            "candle_close" if entry_mode is EntryMode.CLOSE else "live_tick"
        ),
        position_sizing_mode="risk_amount_per_price",
        entry_mode=entry_mode,
        exit_mode=exit_mode,
        exit_band=exit_band,
    )


class EmaAvwapPullbackStrategyTests(unittest.TestCase):
    def test_backtest_defaults_to_closed_candle_compatibility(self) -> None:
        config = EmaAvwapPullbackStrategyConfig(
            symbol="ETHUSDT",
            timeframe="1h",
        )

        self.assertIs(config.entry_mode, EntryMode.CLOSE)
        self.assertIs(config.exit_mode, ExitMode.CLOSE)

    def test_config_supports_all_independent_live_execution_controls(self) -> None:
        config = _config(
            entry_mode="close",
            exit_mode="close",
            exit_band="band_2",
        )

        self.assertIs(config.entry_mode, EntryMode.CLOSE)
        self.assertIs(config.exit_mode, ExitMode.CLOSE)
        self.assertIs(config.exit_band, ExitBand.BAND_2)

        for field, value in (
            ("entry_mode", "invalid"),
            ("exit_mode", "invalid"),
            ("exit_band", "invalid"),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    _config(**{field: value})

    def test_live_modes_discard_an_unfavorable_first_observation(self) -> None:
        candles = (
            _candle(offset=0, open=100.0, high=102.0, low=99.0, close=101.0),
            # The later low crosses AVWAP, but live has already discarded this
            # setup because its first observed price was above AVWAP.
            _candle(offset=1, open=102.0, high=104.0, low=98.0, close=103.0),
        )
        for exit_band in ExitBand:
            with self.subTest(exit_band=exit_band.value):
                strategy = EmaAvwapPullbackStrategy(_config(exit_band=exit_band))
                setup = _SetupState(
                    direction="long",
                    anchor_index=0,
                    detected_index=0,
                    detected_time=candles[0].close_time,
                    consecutive_count=1,
                )

                next_setup, position, _stats, events = _pending_setup(
                    strategy, setup=setup, candles=candles
                )

                self.assertIsNone(next_setup)
                self.assertIsNone(position)
                self.assertEqual(_stats["setups_discarded_unfavorable_first_observation"], 1)
                self.assertEqual(events[0]["event"], "setup_discarded")

    def test_live_middle_mode_enters_short_on_upward_middle_cross(self) -> None:
        candles = (
            _candle(offset=0, open=102.0, high=103.0, low=100.0, close=101.0),
            _candle(offset=1, open=101.0, high=104.0, low=98.0, close=99.0),
        )
        strategy = EmaAvwapPullbackStrategy(_config(exit_band=ExitBand.BAND_2))
        strategy._build_avwap_snapshot = lambda **_: _avwap()  # type: ignore[method-assign]  # noqa: SLF001
        setup = _SetupState(
            direction="short",
            anchor_index=0,
            detected_index=0,
            detected_time=candles[0].close_time,
            consecutive_count=1,
        )

        next_setup, position, _stats, _events = _pending_setup(
            strategy, setup=setup, candles=candles
        )

        self.assertIsNone(next_setup)
        self.assertIsNotNone(position)
        assert position is not None
        self.assertEqual(position.direction, "short")
        self.assertEqual(position.entry_trigger_mode, "live_tick")

    def test_live_mode_enters_if_first_observation_is_past_middle(self) -> None:
        candles = (
            _candle(offset=0, open=100.0, high=102.0, low=99.0, close=101.0),
            _candle(offset=1, open=99.0, high=101.0, low=97.0, close=101.0),
        )
        strategy = EmaAvwapPullbackStrategy(_config())
        strategy._build_avwap_snapshot = lambda **_: _avwap()  # type: ignore[method-assign]  # noqa: SLF001
        setup = _SetupState(
            direction="long",
            anchor_index=0,
            detected_index=0,
            detected_time=candles[0].close_time,
            consecutive_count=1,
        )

        next_setup, position, stats, events = _pending_setup(
            strategy, setup=setup, candles=candles
        )

        self.assertIsNone(next_setup)
        self.assertIsNotNone(position)
        self.assertEqual(stats["setups_discarded_unfavorable_first_observation"], 0)
        self.assertEqual(events[0]["event"], "entry_triggered")

    def test_live_mode_uses_persisted_observation_pair_for_boundary_cross(self) -> None:
        candles = (
            _candle(offset=0, open=100.0, high=102.0, low=99.0, close=101.0),
            _candle(offset=1, open=101.0, high=101.5, low=99.0, close=100.5),
        )
        strategy = EmaAvwapPullbackStrategy(_config())
        strategy._build_avwap_snapshot = lambda **_: _AvwapSnapshot(  # type: ignore[method-assign]  # noqa: SLF001,E501
            anchor_index=0,
            anchor_time=candles[0].open_time,
            candle_index=1,
            vwap=102.0,
            stdev=2.0,
            upper1=104.0,
            lower1=100.0,
            upper2=106.0,
            lower2=98.0,
            upper3=108.0,
            lower3=96.0,
        )
        setup = _SetupState(
            direction="long",
            anchor_index=0,
            detected_index=0,
            detected_time=candles[0].close_time,
            consecutive_count=1,
            is_waiting_for_cross=True,
            last_observed_price=101.0,
            last_observed_middle=100.0,
        )

        next_setup, position, stats, events = _pending_setup(
            strategy, setup=setup, candles=candles
        )

        self.assertIsNone(next_setup)
        self.assertIsNotNone(position)
        assert position is not None
        self.assertEqual(position.raw_entry_price, 102.0)
        self.assertEqual(stats["setups_discarded_unfavorable_first_observation"], 0)
        self.assertEqual(events[0]["event"], "entry_triggered")

    def test_live_setup_age_and_ema_invalidation_gates_are_applied(self) -> None:
        candles = (
            _candle(offset=0, open=100.0, high=102.0, low=99.0, close=101.0),
            _candle(offset=1, open=99.0, high=101.0, low=98.0, close=99.0),
            _candle(offset=2, open=99.0, high=101.0, low=98.0, close=100.0),
        )
        expired_strategy = EmaAvwapPullbackStrategy(_config(max_setup_age_bars=1))
        expired_setup = _SetupState(
            direction="long",
            anchor_index=0,
            detected_index=0,
            detected_time=candles[0].close_time,
            consecutive_count=1,
        )
        tpv, volume, tpv2 = expired_strategy._build_avwap_prefixes(candles)  # noqa: SLF001
        expired_stats = _entry_stats()
        next_setup, position = expired_strategy._process_pending_setup(  # noqa: SLF001
            setup=expired_setup,
            candle=candles[-1],
            candle_index=2,
            prev_close=candles[-2].close,
            ema_value=90.0,
            realized_equity=10_000.0,
            candles=candles,
            tpv_prefix=tpv,
            vol_prefix=volume,
            tpv2_prefix=tpv2,
            stats=expired_stats,
            decision_log=[],
        )
        self.assertIsNone(next_setup)
        self.assertIsNone(position)
        self.assertEqual(expired_stats["setups_expired"], 1)

        ema_strategy = EmaAvwapPullbackStrategy(_config(entry_mode=EntryMode.CLOSE))
        setup = _SetupState(
            direction="long",
            anchor_index=0,
            detected_index=0,
            detected_time=candles[0].close_time,
            consecutive_count=1,
        )
        next_setup, position, stats, events = _pending_setup(
            ema_strategy,
            setup=setup,
            candles=candles[:2],
            ema_value=100.0,
        )
        self.assertIsNone(next_setup)
        self.assertIsNone(position)
        self.assertEqual(stats["setups_invalidated_by_ema"], 1)
        self.assertEqual(events[0]["reason"], "closed price crossed the EMA")

    def test_live_entry_deviation_gate_uses_the_observed_price_proxy(self) -> None:
        candles = (
            _candle(offset=0, open=100.0, high=102.0, low=99.0, close=101.0),
            _candle(offset=1, open=99.0, high=101.0, low=98.0, close=101.0),
        )
        strategy = EmaAvwapPullbackStrategy(_config(max_entry_deviation_pct=0.5))
        strategy._build_avwap_snapshot = lambda **_: _avwap()  # type: ignore[method-assign]  # noqa: SLF001
        setup = _SetupState(
            direction="long",
            anchor_index=0,
            detected_index=0,
            detected_time=candles[0].close_time,
            consecutive_count=1,
        )

        next_setup, position, stats, events = _pending_setup(
            strategy, setup=setup, candles=candles
        )

        self.assertIsNotNone(next_setup)
        assert next_setup is not None
        self.assertTrue(next_setup.is_waiting_for_cross)
        self.assertIsNone(position)
        self.assertEqual(stats["entries_skipped_excessive_deviation"], 1)
        self.assertEqual(events[0]["event"], "entry_skipped")

    def test_live_entry_does_not_use_the_forming_candle_final_ema_close(self) -> None:
        candles = (
            _candle(offset=0, open=100.0, high=102.0, low=99.0, close=101.0),
            # The open is a favourable first live quote. Its eventual close
            # is below EMA, but that result was unavailable when live trading
            # evaluated the entry.
            _candle(offset=1, open=99.0, high=101.0, low=94.0, close=95.0),
        )
        strategy = EmaAvwapPullbackStrategy(_config())
        strategy._build_avwap_snapshot = lambda **_: _avwap()  # type: ignore[method-assign]  # noqa: SLF001
        setup = _SetupState(
            direction="long",
            anchor_index=0,
            detected_index=0,
            detected_time=candles[0].close_time,
            consecutive_count=1,
        )

        next_setup, position, stats, _events = _pending_setup(
            strategy, setup=setup, candles=candles, ema_value=100.0
        )

        self.assertIsNone(next_setup)
        self.assertIsNotNone(position)
        self.assertEqual(stats["setups_invalidated_by_ema"], 0)

    def test_retried_live_setup_uses_the_next_completed_bar_ema_gate(self) -> None:
        candles = (
            _candle(offset=0, open=100.0, high=102.0, low=99.0, close=101.0),
            # Its first live quote is favourable but too far from the middle,
            # so live keeps the setup available for a later tick.
            _candle(offset=1, open=99.0, high=101.0, low=98.0, close=100.0),
            # On the following closed snapshot, the EMA gate applies before
            # any new forming-candle observation.
            _candle(offset=2, open=99.0, high=100.0, low=96.0, close=97.0),
        )
        strategy = EmaAvwapPullbackStrategy(_config(max_entry_deviation_pct=0.5))
        strategy._build_avwap_snapshot = lambda **_: _avwap()  # type: ignore[method-assign]  # noqa: SLF001
        setup = _SetupState(
            direction="long",
            anchor_index=0,
            detected_index=0,
            detected_time=candles[0].close_time,
            consecutive_count=1,
        )

        retained_setup, position, stats, _events = _pending_setup(
            strategy, setup=setup, candles=candles[:2], ema_value=100.0
        )

        self.assertIsNotNone(retained_setup)
        self.assertIsNone(position)
        self.assertEqual(stats["entries_skipped_excessive_deviation"], 1)

        discarded_setup, position, stats, events = _pending_setup(
            strategy,
            setup=retained_setup,
            candles=candles,
            ema_value=100.0,
        )

        self.assertIsNone(discarded_setup)
        self.assertIsNone(position)
        self.assertEqual(stats["setups_invalidated_by_ema"], 1)
        self.assertEqual(events[0]["reason"], "closed price crossed the EMA")

    def test_minimum_balance_gate_matches_live_entry_safety_floor(self) -> None:
        candles = (
            _candle(offset=0, open=100.0, high=102.0, low=99.0, close=101.0),
            _candle(offset=1, open=99.0, high=101.0, low=97.0, close=100.0),
        )
        strategy = EmaAvwapPullbackStrategy(
            _config(minimum_balance_usdt=10_000.0)
        )
        strategy._build_avwap_snapshot = lambda **_: _avwap()  # type: ignore[method-assign]  # noqa: SLF001
        setup = _SetupState(
            direction="long",
            anchor_index=0,
            detected_index=0,
            detected_time=candles[0].close_time,
            consecutive_count=1,
        )

        next_setup, position, stats, events = _pending_setup(
            strategy, setup=setup, candles=candles
        )

        self.assertIsNotNone(next_setup)
        self.assertIsNone(position)
        self.assertEqual(stats["entries_skipped_minimum_balance"], 1)
        self.assertEqual(events[0]["event"], "entry_skipped")

    def test_candle_close_mode_enters_long_at_close_proxy(self) -> None:
        candles = (
            _candle(offset=0, open=99.0, high=102.0, low=98.0, close=101.0),
            _candle(offset=1, open=102.0, high=103.0, low=95.0, close=98.0),
        )
        strategy = EmaAvwapPullbackStrategy(
            _config(
                entry_mode=EntryMode.CLOSE,
                exit_band=ExitBand.BAND_2,
                max_entry_deviation_pct=10.0,
            )
        )
        setup = _SetupState(
            direction="long",
            anchor_index=0,
            detected_index=0,
            detected_time=candles[0].close_time,
            consecutive_count=1,
        )

        _next_setup, position, _stats, events = _pending_setup(
            strategy, setup=setup, candles=candles, ema_value=90.0
        )

        self.assertIsNotNone(position)
        assert position is not None
        self.assertEqual(position.raw_entry_price, candles[-1].close)
        self.assertEqual(position.decision_price, candles[-1].close)
        self.assertEqual(position.entry_trigger_mode, "candle_close")
        event = next(item for item in events if item["event"] == "entry_triggered")
        self.assertEqual(event["exit_band_number"], 2)

    def test_candle_close_mode_enters_short_on_bullish_close_above_middle(self) -> None:
        candles = (
            _candle(offset=0, open=102.0, high=103.0, low=99.0, close=100.0),
            _candle(offset=1, open=99.0, high=106.0, low=98.0, close=104.0),
        )
        strategy = EmaAvwapPullbackStrategy(
            _config(
                entry_mode=EntryMode.CLOSE,
                exit_band=ExitBand.BAND_2,
                max_entry_deviation_pct=10.0,
            )
        )
        setup = _SetupState(
            direction="short",
            anchor_index=0,
            detected_index=0,
            detected_time=candles[0].close_time,
            consecutive_count=1,
        )

        _next_setup, position, _stats, _events = _pending_setup(
            strategy, setup=setup, candles=candles, ema_value=105.0
        )

        self.assertIsNotNone(position)
        assert position is not None
        self.assertEqual(position.direction, "short")
        self.assertEqual(position.raw_entry_price, candles[-1].close)
        self.assertIs(position.exit_band, ExitBand.BAND_2)

    def test_candle_close_mode_requires_opposite_color_pullback(self) -> None:
        candles = (
            _candle(offset=0, open=99.0, high=102.0, low=98.0, close=101.0),
            # Close is below the middle, but this is bullish and must not enter a long.
            _candle(offset=1, open=95.0, high=104.0, low=94.0, close=98.0),
        )
        strategy = EmaAvwapPullbackStrategy(_config(entry_mode=EntryMode.CLOSE))
        setup = _SetupState(
            direction="long",
            anchor_index=0,
            detected_index=0,
            detected_time=candles[0].close_time,
            consecutive_count=1,
        )

        next_setup, position, _stats, _events = _pending_setup(
            strategy, setup=setup, candles=candles, ema_value=90.0
        )

        self.assertIs(next_setup, setup)
        self.assertIsNone(position)

    def test_exit_bands_match_the_live_target_bands(self) -> None:
        # Close at the favourable extreme so band 2 can activate its trailing
        # stop without subsequently hitting it in this candle.
        first_band_candle = _candle(offset=1, open=100.0, high=102.5, low=99.0, close=102.5)
        second_band_candle = _candle(offset=1, open=100.0, high=104.5, low=99.0, close=103.0)
        stats = _entry_stats()

        mode_1_strategy = EmaAvwapPullbackStrategy(_config(exit_band=ExitBand.BAND_1))
        decision = mode_1_strategy._process_position_for_candle(  # noqa: SLF001
            position=_position(exit_band=ExitBand.BAND_1),
            candle=first_band_candle,
            candle_index=1,
            prev_close=100.0,
            avwap=_avwap(),
            stats=stats,
            decision_log=[],
        )
        self.assertIsNotNone(decision)
        self.assertEqual(decision.reason, "AVWAP band 1 target")  # type: ignore[union-attr]
        self.assertEqual(decision.raw_exit_price, 102.0)  # type: ignore[union-attr]

        for entry_mode in EntryMode:
            with self.subTest(entry_mode=entry_mode.value):
                strategy = EmaAvwapPullbackStrategy(
                    _config(entry_mode=entry_mode, exit_band=ExitBand.BAND_2)
                )
                no_exit = strategy._process_position_for_candle(  # noqa: SLF001
                    position=_position(entry_mode=entry_mode, exit_band=ExitBand.BAND_2),
                    candle=first_band_candle,
                    candle_index=1,
                    prev_close=100.0,
                    avwap=_avwap(),
                    stats=stats,
                    decision_log=[],
                )
                self.assertIsNone(no_exit)
                target_exit = strategy._process_position_for_candle(  # noqa: SLF001
                    position=_position(entry_mode=entry_mode, exit_band=ExitBand.BAND_2),
                    candle=second_band_candle,
                    candle_index=1,
                    prev_close=100.0,
                    avwap=_avwap(),
                    stats=stats,
                    decision_log=[],
                )
                self.assertIsNotNone(target_exit)
                self.assertEqual(
                    target_exit.reason,  # type: ignore[union-attr]
                    "AVWAP band 2 target",
                )
                self.assertEqual(target_exit.raw_exit_price, 104.0)  # type: ignore[union-attr]

    def test_trailing_stop_is_protective_and_avwap_band_is_not_a_stop(self) -> None:
        strategy = EmaAvwapPullbackStrategy(_config())
        position = _position()
        position.trailing_active = True
        position.trailing_stop = 99.0
        candle_above_rigid = _candle(offset=1, open=100.0, high=101.0, low=97.5, close=100.0)

        trailing_exit = strategy._process_position_for_candle(  # noqa: SLF001
            position=position,
            candle=candle_above_rigid,
            candle_index=1,
            prev_close=100.0,
            avwap=_avwap(),
            stats=_entry_stats(),
            decision_log=[],
        )
        self.assertIsNotNone(trailing_exit)
        assert trailing_exit is not None
        self.assertEqual(trailing_exit.reason, "Trailing stop")
        self.assertEqual(trailing_exit.raw_exit_price, 99.0)

        rigid_stop_candle = _candle(offset=2, open=100.0, high=101.0, low=96.5, close=98.0)
        exit_decision = strategy._process_position_for_candle(  # noqa: SLF001
            position=_position(),
            candle=rigid_stop_candle,
            candle_index=2,
            prev_close=100.0,
            avwap=_avwap(),
            stats=_entry_stats(),
            decision_log=[],
        )
        self.assertIsNotNone(exit_decision)
        self.assertEqual(exit_decision.reason, "Rigid stop loss")  # type: ignore[union-attr]
        self.assertEqual(exit_decision.raw_exit_price, 97.0)  # type: ignore[union-attr]

    def test_gap_exit_uses_the_next_observed_open_like_live_ticks(self) -> None:
        strategy = EmaAvwapPullbackStrategy(_config())
        decision = strategy._process_position_for_candle(  # noqa: SLF001
            position=_position(exit_band=ExitBand.BAND_2),
            candle=_candle(offset=1, open=95.0, high=96.0, low=94.0, close=95.0),
            candle_index=1,
            prev_close=100.0,
            avwap=_avwap(),
            stats=_entry_stats(),
            decision_log=[],
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.reason, "Rigid stop loss")
        self.assertEqual(decision.raw_exit_price, 95.0)

    def test_ohlc_proxy_activates_and_triggers_long_trailing_stop(self) -> None:
        strategy = EmaAvwapPullbackStrategy(
            _config(
                exit_band=ExitBand.BAND_2,
                trailing_activation_threshold_pct=0.0,
                trailing_gap_pct=1.0,
            )
        )
        position = _position(exit_band=ExitBand.BAND_2)
        stats = _entry_stats()
        decision = strategy._process_position_for_candle(  # noqa: SLF001
            position=position,
            candle=_candle(offset=1, open=100.0, high=103.0, low=101.0, close=101.0),
            candle_index=1,
            prev_close=100.0,
            avwap=_avwap(),
            stats=stats,
            decision_log=[],
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.reason, "Trailing stop")
        self.assertAlmostEqual(decision.raw_exit_price, 101.97)
        self.assertTrue(position.trailing_active)
        self.assertAlmostEqual(position.extreme_price or 0.0, 103.0)
        self.assertAlmostEqual(position.trailing_stop or 0.0, 101.97)
        self.assertEqual(stats["trailing_activations"], 1)

    def test_ohlc_proxy_activates_and_triggers_short_trailing_stop(self) -> None:
        strategy = EmaAvwapPullbackStrategy(
            _config(
                exit_band=ExitBand.BAND_2,
                trailing_activation_threshold_pct=0.0,
                trailing_gap_pct=1.0,
            )
        )
        position = _position(direction="short", exit_band=ExitBand.BAND_2)
        stats = _entry_stats()
        decision = strategy._process_position_for_candle(  # noqa: SLF001
            position=position,
            candle=_candle(offset=1, open=100.0, high=101.0, low=97.0, close=99.0),
            candle_index=1,
            prev_close=100.0,
            avwap=_avwap(),
            stats=stats,
            decision_log=[],
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.reason, "Trailing stop")
        self.assertAlmostEqual(decision.raw_exit_price, 97.97)
        self.assertTrue(position.trailing_active)
        self.assertAlmostEqual(position.extreme_price or 0.0, 97.0)
        self.assertAlmostEqual(position.trailing_stop or 0.0, 97.97)
        self.assertEqual(stats["trailing_activations"], 1)

    def test_trailing_uses_the_last_closed_avwap_not_forming_target_avwap(self) -> None:
        strategy = EmaAvwapPullbackStrategy(_config(exit_band=ExitBand.BAND_2))
        position = _position(exit_band=ExitBand.BAND_2)
        position.trailing_avwap = replace(_avwap(), upper1=110.0)
        stats = _entry_stats()

        decision = strategy._process_position_for_candle(  # noqa: SLF001
            position=position,
            candle=_candle(offset=1, open=100.0, high=103.0, low=99.0, close=103.0),
            candle_index=1,
            prev_close=100.0,
            avwap=_avwap(),
            trailing_avwap=position.trailing_avwap,
            stats=stats,
            decision_log=[],
        )

        self.assertIsNone(decision)
        self.assertFalse(position.trailing_active)
        self.assertEqual(stats["trailing_activations"], 0)

    def test_target_has_precedence_over_initial_trailing_activation(self) -> None:
        strategy = EmaAvwapPullbackStrategy(
            _config(
                trailing_activation_threshold_pct=0.0,
            )
        )
        stats = _entry_stats()
        decision = strategy._process_position_for_candle(  # noqa: SLF001
            position=_position(),
            candle=_candle(offset=1, open=100.0, high=103.0, low=101.0, close=101.0),
            candle_index=1,
            prev_close=100.0,
            avwap=_avwap(),
            stats=stats,
            decision_log=[],
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.reason, "AVWAP band 1 target")
        self.assertEqual(stats["trailing_activations"], 0)

    def test_close_exit_mode_ignores_intrabar_target_and_exits_on_close_only(self) -> None:
        strategy = EmaAvwapPullbackStrategy(
            _config(
                entry_mode=EntryMode.LIVE,
                exit_mode=ExitMode.CLOSE,
                exit_band=ExitBand.BAND_1,
                trailing_activation_threshold_pct=10.0,
            )
        )
        position = _position(exit_mode=ExitMode.CLOSE)
        intrabar_only = _candle(
            offset=1, open=100.0, high=105.0, low=99.0, close=101.0
        )
        stats = _entry_stats()

        decision = strategy._process_position_for_candle(  # noqa: SLF001
            position=position,
            candle=intrabar_only,
            candle_index=1,
            prev_close=100.0,
            avwap=_avwap(),
            stats=stats,
            decision_log=[],
        )

        self.assertIsNone(decision)

        closes_at_target = _candle(
            offset=2, open=100.0, high=105.0, low=99.0, close=103.0
        )
        decision = strategy._process_position_for_candle(  # noqa: SLF001
            position=position,
            candle=closes_at_target,
            candle_index=2,
            prev_close=101.0,
            avwap=_avwap(),
            stats=stats,
            decision_log=[],
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.reason, "AVWAP band 1 target on candle close")
        self.assertEqual(decision.raw_exit_price, 103.0)

    def test_zero_gap_trailing_closes_on_its_activation_price(self) -> None:
        strategy = EmaAvwapPullbackStrategy(
            _config(
                exit_band=ExitBand.BAND_2,
                trailing_gap_pct=0.0,
            )
        )
        stats = _entry_stats()
        decision = strategy._process_position_for_candle(  # noqa: SLF001
            position=_position(exit_band=ExitBand.BAND_2),
            candle=_candle(offset=1, open=100.0, high=103.0, low=101.0, close=103.0),
            candle_index=1,
            prev_close=100.0,
            avwap=_avwap(),
            stats=stats,
            decision_log=[],
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.reason, "Trailing stop")
        self.assertEqual(decision.raw_exit_price, 103.0)
        self.assertEqual(stats["trailing_activations"], 1)

    def test_live_incompatible_sizing_and_disabled_rigid_stop_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "position-notional budget"):
            _config(position_sizing_mode="risk_distance")
        with self.assertRaisesRegex(ValueError, "protective stop"):
            _config(rigid_stop_loss_pct=0.0)

    def test_max_entry_notional_clamps_quantity(self) -> None:
        candles = (
            _candle(offset=0, open=100.0, high=102.0, low=99.0, close=101.0),
            _candle(offset=1, open=102.0, high=104.0, low=97.0, close=98.0),
        )
        strategy = EmaAvwapPullbackStrategy(
            _config(
                entry_mode=EntryMode.CLOSE,
                max_entry_notional_usdt=25.0,
                max_entry_deviation_pct=10.0,
            )
        )
        setup = _SetupState(
            direction="long",
            anchor_index=0,
            detected_index=0,
            detected_time=candles[0].close_time,
            consecutive_count=1,
        )

        _next_setup, position, stats, _events = _pending_setup(
            strategy, setup=setup, candles=candles, ema_value=90.0
        )

        self.assertIsNotNone(position)
        assert position is not None
        self.assertAlmostEqual(position.qty * position.raw_entry_price, 25.0)
        self.assertEqual(stats["entries_capped_by_live_notional_limits"], 1)

    def test_price_based_sizing_matches_live_cost_buffer_math(self) -> None:
        strategy = EmaAvwapPullbackStrategy(
            _config(
                maker_fee_pct=0.001,
                taker_fee_pct=0.002,
                entry_slippage_pct=0.01,
                exit_slippage_pct=0.02,
            )
        )

        sizing = strategy._build_sizing_decision(  # noqa: SLF001
            direction="long",
            raw_entry_price=100.0,
            position_notional_budget=100.0,
        )

        self.assertIsNotNone(sizing)
        assert sizing is not None
        expected_cost_per_unit = 1.0 + 2.0 + 0.202 + 0.196
        self.assertAlmostEqual(sizing.total_cost_per_unit, expected_cost_per_unit)
        self.assertAlmostEqual(sizing.qty, 100.0 / (100.0 + expected_cost_per_unit))
        self.assertEqual(sizing.position_notional_budget, 100.0)

    def test_config_snapshot_reports_mode_and_notional_cap(self) -> None:
        strategy = EmaAvwapPullbackStrategy(
            _config(
                exit_band=ExitBand.BAND_2,
                max_entry_notional_usdt=50.0,
            )
        )

        config = strategy._config_as_dict()  # noqa: SLF001

        self.assertEqual(config["entry_mode"], "live")
        self.assertEqual(config["exit_mode"], "live")
        self.assertEqual(config["exit_band"], "band_2")
        self.assertEqual(config["max_entry_notional_usdt"], 50.0)

    def test_execution_metadata_distinguishes_closed_and_live_ohlc_modes(self) -> None:
        candles = [
            _candle(offset=0, open=99.0, high=101.0, low=98.0, close=100.0),
            _candle(offset=1, open=100.0, high=101.0, low=99.0, close=100.0),
        ]
        context = BacktestContext(
            config=BacktestRunConfig(
                start=candles[0].open_time,
                end=candles[-1].close_time,
            ),
            data={"ETHUSDT": {"1h": candles}},
            ignore_candles={"ETHUSDT": {"1h": 0}},
        )

        _trades, closed_stats = EmaAvwapPullbackStrategy(
            _config(entry_mode=EntryMode.CLOSE, exit_mode=ExitMode.CLOSE)
        ).run(context)
        _trades, live_stats = EmaAvwapPullbackStrategy(_config()).run(context)

        assert closed_stats is not None
        assert live_stats is not None
        self.assertEqual(
            closed_stats["execution_assumptions"]["live_strategy_compatibility"],
            "closed_candle_equivalent",
        )
        self.assertIsNone(
            closed_stats["execution_assumptions"]["live_tick_approximation_warning"]
        )
        self.assertEqual(
            live_stats["execution_assumptions"]["live_strategy_compatibility"],
            "live_tick_ohlc_approximation",
        )
        self.assertIsNotNone(
            live_stats["execution_assumptions"]["live_tick_approximation_warning"]
        )

    def test_cli_and_web_builders_forward_mode_and_notional_cap(self) -> None:
        from cmd.backtest.main import (
            build_ema_avwap_pullback_parser,
            build_ema_avwap_pullback_strategy,
            load_ema_avwap_pullback_env_config,
            resolve_ema_avwap_pullback_config,
        )
        from webserver.models import EmaAvwapPullbackParams
        from webserver.runner import _build_ema_avwap_pullback_strategy

        parser = build_ema_avwap_pullback_parser()
        args = parser.parse_args(
            [
                "--symbol",
                "ETHUSDT",
                "--timeframe",
                "1h",
                "--leverage",
                "2",
                "--initial-capital",
                "1000",
                "--entry-mode",
                "close",
                "--exit-mode",
                "close",
                "--exit-band",
                "band_2",
                "--max-entry-notional-usdt",
                "25",
                "--position-notional-pct",
                "2",
                "--max-position-size-pct",
                "5",
            ]
        )
        resolved = resolve_ema_avwap_pullback_config(args, load_ema_avwap_pullback_env_config())
        cli_strategy = build_ema_avwap_pullback_strategy(resolved)
        self.assertIs(cli_strategy._config.entry_mode, EntryMode.CLOSE)  # noqa: SLF001
        self.assertIs(cli_strategy._config.exit_mode, ExitMode.CLOSE)  # noqa: SLF001
        self.assertIs(cli_strategy._config.exit_band, ExitBand.BAND_2)  # noqa: SLF001
        self.assertEqual(cli_strategy._config.max_entry_notional_usdt, 25.0)  # noqa: SLF001
        self.assertEqual(cli_strategy._config.position_notional_pct, 2.0)  # noqa: SLF001

        params = EmaAvwapPullbackParams(
            entry_mode="close",
            exit_mode="live",
            exit_band="band_2",
            max_entry_notional_usdt=30.0,
            max_setup_age_bars=4,
        )
        web_strategy = _build_ema_avwap_pullback_strategy(params, initial_equity=1000.0)
        self.assertIs(web_strategy._config.entry_mode, EntryMode.CLOSE)  # noqa: SLF001
        self.assertIs(web_strategy._config.exit_mode, ExitMode.LIVE)  # noqa: SLF001
        self.assertIs(web_strategy._config.exit_band, ExitBand.BAND_2)  # noqa: SLF001
        self.assertEqual(web_strategy._config.max_entry_notional_usdt, 30.0)  # noqa: SLF001
        self.assertEqual(web_strategy._config.max_setup_age_bars, 4)  # noqa: SLF001

        direct_args = parser.parse_args(
            [
                "--symbol",
                "ETHUSDT",
                "--timeframe",
                "1h",
                "--leverage",
                "2",
                "--initial-capital",
                "1000",
                "--entry-mode",
                "close",
                "--exit-mode",
                "close",
                "--exit-band",
                "band_2",
                "--minimum-balance-usdt",
                "25",
            ]
        )
        direct_resolved = resolve_ema_avwap_pullback_config(
            direct_args, load_ema_avwap_pullback_env_config()
        )
        direct_strategy = build_ema_avwap_pullback_strategy(direct_resolved)
        self.assertIs(direct_strategy._config.entry_mode, EntryMode.CLOSE)  # noqa: SLF001
        self.assertIs(direct_strategy._config.exit_mode, ExitMode.CLOSE)  # noqa: SLF001
        self.assertIs(direct_strategy._config.exit_band, ExitBand.BAND_2)  # noqa: SLF001
        self.assertEqual(direct_strategy._config.minimum_balance_usdt, 25.0)  # noqa: SLF001

        direct_params = EmaAvwapPullbackParams(
            entry_mode="close",
            exit_mode="close",
            exit_band="band_2",
            minimum_balance_usdt=50.0,
        )
        direct_web_strategy = _build_ema_avwap_pullback_strategy(
            direct_params, initial_equity=1000.0
        )
        self.assertIs(direct_web_strategy._config.entry_mode, EntryMode.CLOSE)  # noqa: SLF001
        self.assertIs(direct_web_strategy._config.exit_mode, ExitMode.CLOSE)  # noqa: SLF001
        self.assertIs(direct_web_strategy._config.exit_band, ExitBand.BAND_2)  # noqa: SLF001
        self.assertEqual(direct_web_strategy._config.minimum_balance_usdt, 50.0)  # noqa: SLF001

    def test_full_backtest_reports_mode_and_live_avwap_target_exit(self) -> None:
        candles = [
            _candle(offset=0, open=99.0, high=101.0, low=98.0, close=100.0),
            _candle(offset=1, open=104.0, high=106.0, low=103.0, close=105.0),
            _candle(offset=2, open=104.0, high=110.0, low=103.0, close=105.0),
            _candle(offset=3, open=105.0, high=106.0, low=104.0, close=105.0),
        ]
        context = BacktestContext(
            config=BacktestRunConfig(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 3, tzinfo=timezone.utc),
            ),
            data={"ETHUSDT": {"1h": candles}},
            ignore_candles={"ETHUSDT": {"1h": 0}},
        )
        strategy = EmaAvwapPullbackStrategy(
            _config(
                max_entry_deviation_pct=2.0,
            )
        )

        trades, stats = strategy.run(context)

        self.assertEqual(len(trades), 1)
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(trades[0].notes, "AVWAP band 1 target")
        self.assertIsNotNone(trades[0].metadata)
        assert trades[0].metadata is not None
        self.assertEqual(trades[0].metadata["entry_mode"], "live")
        self.assertEqual(trades[0].metadata["exit_mode"], "live")
        self.assertEqual(trades[0].metadata["exit_band"], "band_1")
        self.assertEqual(stats["target_exits_band_1"], 1)
        self.assertEqual(stats["target_exits_band_2"], 0)
        self.assertEqual(
            stats["execution_assumptions"]["entry_avwap_value_source"],
            "completed_bar_proxy_for_forming_candle",
        )
        self.assertEqual(
            stats["execution_assumptions"]["target_avwap_value_source"],
            "completed_bar_proxy_for_forming_candle",
        )


if __name__ == "__main__":
    unittest.main()
