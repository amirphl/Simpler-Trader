from __future__ import annotations

import unittest
import time
import threading
import json
import logging
from argparse import Namespace
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from cmd.live_trading import _shared as live_trading_shared
from candle_downloader.models import Candle, to_milliseconds
from live_trading.ema_avwap_pullback_strategy import (
    EmaAvwapPullbackLiveConfig,
    EmaAvwapPullbackLiveCoordinator,
    EntryMode,
    ExitBand,
    ExitMode,
    _AvwapSnapshot,
    _CrossDecision,
    _EntryCandidate,
    _PendingEntryMeta,
    _PositionRuntime,
    _SetupState,
    _SizingDecision,
    _SymbolSnapshot,
)
from live_trading.exchange import (
    ExchangeConfig,
    KlineUpdate,
    MarginMode,
    OrderResult,
    OrderType,
    Position,
    PositionSide,
)
from live_trading.exchanges.bitunix.kline_stream import (
    BitunixKlineStream,
    kline_channel_for_interval,
)
from live_trading.models import LiveTradingConfig, PendingEntryRecord, PositionRecord


class _FakeExchange:
    def __init__(self, proxies: dict[str, str] | None = None) -> None:
        self._config = ExchangeConfig(
            api_key="",
            api_secret="",
            proxies=proxies,
        )
        self.balance = 10_000.0
        self.price = 100.0
        self.positions: list[Position] = []
        self.positions_exc: Exception | None = None
        self.stop_update_ok = True
        self.stop_updates: list[float] = []
        self.close_calls: list[tuple[str, PositionSide | None]] = []
        self.close_error: Exception | None = None
        self.close_error_removes_position = False
        self.open_orders: list[tuple[str, PositionSide, float, float, float | None]] = []
        self.open_order_client_ids: list[str | None] = []
        self.validated_quantities: list[tuple[str, float]] = []
        self.quantity_validation_error: Exception | None = None
        self.margin_mode_updates: list[tuple[str, MarginMode]] = []
        self.leverage_updates: list[tuple[str, int]] = []
        self.open_limit_attempts = 0
        self.order_statuses: dict[str, dict[str, str]] = {}
        self.klines: list[list] = []
        self.kline_requests: list[tuple[str, str, int]] = []

    def get_account_balance(self) -> float:
        return self.balance

    def fetch_price(self, symbol: str) -> float:
        return self.price

    def get_current_positions(self) -> list[Position]:
        if self.positions_exc is not None:
            raise self.positions_exc
        return list(self.positions)

    def get_position(self, symbol: str) -> Position | None:
        for position in self.get_current_positions():
            if position.symbol == symbol:
                return position
        return None

    def set_margin_mode(self, symbol: str, margin_mode: MarginMode) -> None:
        self.margin_mode_updates.append((symbol, margin_mode))
        return None

    def set_leverage(self, symbol: str, leverage: int) -> None:
        self.leverage_updates.append((symbol, leverage))
        return None

    def validate_order_quantity(self, symbol: str, quantity: float) -> float:
        self.validated_quantities.append((symbol, quantity))
        if self.quantity_validation_error is not None:
            raise self.quantity_validation_error
        return quantity

    def open_limit_position(
        self,
        symbol: str,
        side: PositionSide,
        quantity: float,
        price: float,
        leverage: int,
        margin_mode: MarginMode,
        take_profit=None,
        stop_loss=None,
        client_id: str | None = None,
    ) -> OrderResult:
        self.open_limit_attempts += 1
        self.open_orders.append((symbol, side, quantity, price, stop_loss))
        self.open_order_client_ids.append(client_id)
        return OrderResult(
            order_id="order-1",
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            price=price,
            quantity=quantity,
            status="NEW",
            timestamp=datetime.now(timezone.utc),
        )

    def update_position_stop_loss(self, position: Position, stop_price: float) -> bool:
        self.stop_updates.append(stop_price)
        return self.stop_update_ok

    def ensure_position_stop_loss(
        self, position: Position, stop_price: float
    ) -> float | None:
        self.stop_updates.append(stop_price)
        return stop_price if self.stop_update_ok else None

    def get_order_status(
        self,
        *,
        symbol: str,
        order_id: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, str] | None:
        del symbol
        if order_id and order_id in self.order_statuses:
            return self.order_statuses[order_id]
        if client_id and client_id in self.order_statuses:
            return self.order_statuses[client_id]
        if order_id:
            return {"orderId": order_id, "status": "FILLED"}
        return None

    def close_position(
        self, symbol: str, side: PositionSide | None = None
    ) -> OrderResult:
        self.close_calls.append((symbol, side))
        if self.close_error is not None:
            if self.close_error_removes_position:
                self.positions = [
                    position
                    for position in self.positions
                    if position.symbol != symbol
                ]
            raise self.close_error
        self.positions = [
            position for position in self.positions if position.symbol != symbol
        ]
        return OrderResult(
            order_id="close-1",
            symbol=symbol,
            side=side or PositionSide.SHORT,
            order_type=OrderType.MARKET,
            price=self.price,
            quantity=1.0,
            status="FILLED",
            timestamp=datetime.now(timezone.utc),
        )

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        return True

    def get_klines(self, symbol: str, interval: str, limit: int = 500, **kwargs):
        self.kline_requests.append((symbol, interval, limit))
        return self.klines[-limit:]

    def close(self) -> None:
        return None


class _FakeTelegram:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def send_message(self, text: str) -> None:
        self.messages.append(text)


def _candle(
    *,
    offset: int,
    open: float,
    high: float,
    low: float,
    close: float,
    volume: float = 100.0,
) -> Candle:
    open_time = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=offset)
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


def _binance_row(candle: Candle) -> list:
    return [
        candle.open_time_ms,
        f"{candle.open:.10f}",
        f"{candle.high:.10f}",
        f"{candle.low:.10f}",
        f"{candle.close:.10f}",
        f"{candle.volume:.10f}",
        to_milliseconds(candle.close_time),
    ]


def _kline_update(candle: Candle) -> KlineUpdate:
    return KlineUpdate(
        symbol=candle.symbol,
        interval=candle.interval,
        event_time_ms=to_milliseconds(candle.open_time + timedelta(minutes=30)),
        open=candle.open,
        high=candle.high,
        low=candle.low,
        close=candle.close,
        base_volume=candle.volume,
        quote_volume=candle.volume * candle.close,
    )


def _candidate() -> _EntryCandidate:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    avwap = _AvwapSnapshot(
        anchor_index=0,
        anchor_time=now,
        candle_index=1,
        vwap=100.0,
        stdev=2.5,
        upper1=102.5,
        lower1=97.5,
        upper2=105.0,
        lower2=95.0,
        upper3=107.5,
        lower3=92.5,
    )
    sizing = _SizingDecision(
        qty=1.0,
        distance=2.5,
        entry_price=100.0,
        estimated_exit_price=100.0,
        position_notional_budget=5.0,
        base_qty_before_costs=1.0,
        qty_reduction_from_costs=0.0,
        sizing_reference_price=100.0,
        effective_price_for_sizing=2.5,
        entry_slippage_per_unit=0.0,
        exit_slippage_per_unit=0.0,
        entry_fee_per_unit=0.0,
        exit_fee_per_unit=0.0,
        total_cost_per_unit=0.0,
    )
    return _EntryCandidate(
        symbol="ETHUSDT",
        side=PositionSide.LONG,
        direction="long",
        signal_time=now,
        anchor_time=now,
        setup_detected_time=now,
        candle_index=1,
        raw_entry_price=100.0,
        order_price=100.0,
        stop_for_risk=97.5,
        dynamic_stop_at_entry=97.5,
        rigid_stop_at_entry=None,
        trailing_activation_at_entry=102.5,
        quantity=1.0,
        position_notional_budget=5.0,
        entry_trigger_mode="intrabar",
        sizing=sizing,
        avwap=avwap,
    )


def _snapshot() -> _SymbolSnapshot:
    first = _candle(offset=0, open=99.0, high=101.0, low=98.0, close=100.0)
    latest = _candle(offset=1, open=99.0, high=101.0, low=98.0, close=100.0)
    return _SymbolSnapshot(
        symbol="ETHUSDT",
        timeframe="1h",
        timeframe_minutes=60,
        candles=(first, latest),
        candle_index=1,
        candle=latest,
        previous_candle=first,
        ema_value=90.0,
        tpv_prefix=(0.0,),
        vol_prefix=(0.0,),
        tpv2_prefix=(0.0,),
    )


def _short_avwap() -> _AvwapSnapshot:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return _AvwapSnapshot(
        anchor_index=0,
        anchor_time=now,
        candle_index=1,
        vwap=100.0,
        stdev=2.5,
        upper1=102.5,
        lower1=97.5,
        upper2=105.0,
        lower2=95.0,
        upper3=107.5,
        lower3=92.5,
    )


class EmaAvwapPullbackLiveCoordinatorTests(unittest.TestCase):
    def _persistent_config(
        self, tmpdir: str, **overrides
    ) -> EmaAvwapPullbackLiveConfig:
        base = Path(tmpdir)
        values = {
            "symbols": ("ETHUSDT",),
            "max_entry_notional_usdt": 1_000.0,
            "position_sizing_mode": "risk_amount_per_price",
            "rigid_stop_loss_pct": 3.0,
            "state_file": base / "state.json",
            "positions_db": base / "positions.db",
        }
        values.update(overrides)
        return EmaAvwapPullbackLiveConfig(
            **values,
        )

    def _drain_market_data_until(
        self,
        coordinator: EmaAvwapPullbackLiveCoordinator,
        now: datetime,
        predicate,
    ) -> None:
        for _ in range(100):
            coordinator._maybe_process_new_candles(now)  # noqa: SLF001
            if predicate():
                return
            time.sleep(0.01)
        self.fail("market-data worker did not complete in time")

    def test_signal_candles_use_binance_not_execution_exchange(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            execution_calls = 0
            binance_calls = 0

            def fail_get_klines(*args, **kwargs):
                nonlocal execution_calls
                execution_calls += 1
                raise AssertionError("Bitunix kline data must not feed EMA/AVWAP")

            exchange.get_klines = fail_get_klines  # type: ignore[method-assign]
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(tmpdir),
            )
            candles = [
                replace(
                    _candle(
                        offset=0, open=98.0, high=101.0, low=97.0, close=100.0
                    ),
                    close_time=datetime(2026, 1, 1, 1, tzinfo=timezone.utc)
                    - timedelta(milliseconds=1),
                ),
                replace(
                    _candle(
                        offset=1, open=100.0, high=102.0, low=99.0, close=101.0
                    ),
                    close_time=datetime(2026, 1, 1, 2, tzinfo=timezone.utc)
                    - timedelta(milliseconds=1),
                ),
            ]

            def fetch_recent_klines(**kwargs):
                nonlocal binance_calls
                binance_calls += 1
                self.assertEqual(kwargs, {
                    "symbol": "ETHUSDT", "interval": "1h", "limit": 3
                })
                return candles

            coordinator._binance_signal_client.fetch_recent_klines = fetch_recent_klines  # type: ignore[method-assign]  # noqa: SLF001,E501
            result = coordinator._fetch_binance_signal_candles(  # noqa: SLF001
                "ETHUSDT", "1h", 3
            )

            self.assertEqual(result, candles)
            self.assertEqual(binance_calls, 1)
            self.assertEqual(execution_calls, 0)
            coordinator.stop()

    def test_ready_closed_candles_use_timestamp_and_configured_delay(self) -> None:
        with TemporaryDirectory() as tmpdir:
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=_FakeExchange(),
                config=self._persistent_config(tmpdir, candle_ready_delay_seconds=30),
            )
            first = _candle(offset=0, open=99.0, high=101.0, low=98.0, close=100.0)
            latest = _candle(
                offset=1, open=100.0, high=102.0, low=99.0, close=101.0
            )
            rows = [_binance_row(first), _binance_row(latest)]

            before_delay = coordinator._ready_closed_candles(  # noqa: SLF001
                "ETHUSDT",
                "1h",
                rows,
                now=datetime(2026, 1, 1, 2, 0, 29, tzinfo=timezone.utc),
            )
            after_delay = coordinator._ready_closed_candles(  # noqa: SLF001
                "ETHUSDT",
                "1h",
                rows,
                now=datetime(2026, 1, 1, 2, 0, 30, tzinfo=timezone.utc),
            )

            self.assertEqual(before_delay, [first])
            self.assertEqual(after_delay, [first, latest])

    def test_market_data_poll_never_skips_a_strategy_candle(self) -> None:
        with TemporaryDirectory() as tmpdir:
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=_FakeExchange(),
                config=self._persistent_config(
                    tmpdir,
                    timeframe="1m",
                    execution_interval_minutes=5,
                    candle_ready_delay_seconds=0,
                ),
            )
            first_boundary = datetime(2026, 1, 1, 2, tzinfo=timezone.utc)

            self.assertTrue(coordinator._market_data_poll_is_due(first_boundary))  # noqa: SLF001
            coordinator._last_market_data_poll_slot_by_symbol["ETHUSDT"] = (  # noqa: SLF001
                coordinator._market_data_poll_slot(first_boundary)  # noqa: SLF001
            )
            self.assertFalse(
                coordinator._market_data_poll_is_due(  # noqa: SLF001
                    first_boundary + timedelta(seconds=30)
                )
            )
            self.assertTrue(
                coordinator._market_data_poll_is_due(  # noqa: SLF001
                    first_boundary + timedelta(minutes=1)
                )
            )

    def test_market_data_retries_same_close_until_new_candle_is_available(self) -> None:
        with TemporaryDirectory() as tmpdir:
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=_FakeExchange(),
                config=self._persistent_config(
                    tmpdir,
                    timeframe="1h",
                    candle_ready_delay_seconds=0,
                ),
            )
            snapshot = _snapshot()
            previous = snapshot.previous_candle
            latest = snapshot.candle
            calls = 0

            def fetch_latest(symbol: str) -> Candle:
                nonlocal calls
                del symbol
                calls += 1
                return previous if calls == 1 else latest

            coordinator._last_closed_candle_time_by_symbol["ETHUSDT"] = (  # noqa: SLF001
                previous.close_time
            )
            coordinator._fetch_latest_closed_candle = fetch_latest  # type: ignore[method-assign]  # noqa: SLF001,E501
            coordinator._build_snapshot = lambda symbol: snapshot  # type: ignore[method-assign]  # noqa: SLF001,E501

            self._drain_market_data_until(
                coordinator,
                latest.close_time,
                lambda: coordinator._last_closed_candle_time_by_symbol.get(  # noqa: SLF001
                    "ETHUSDT"
                )
                == latest.close_time,
            )

            self.assertGreaterEqual(calls, 2)

    def test_bitunix_inclusive_close_time_covers_later_poll_slot(self) -> None:
        """A Bitunix bar ends at boundary - 1 ms, not at the boundary itself."""
        with TemporaryDirectory() as tmpdir:
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=_FakeExchange(),
                config=self._persistent_config(
                    tmpdir,
                    timeframe="1h",
                    execution_interval_minutes=5,
                    candle_ready_delay_seconds=0,
                ),
            )
            boundary = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
            bitunix_closed = Candle(
                symbol="ETHUSDT",
                interval="1h",
                open_time=boundary - timedelta(hours=1),
                close_time=boundary - timedelta(milliseconds=1),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=100.0,
            )
            later_slot = coordinator._market_data_poll_slot(  # noqa: SLF001
                boundary + timedelta(minutes=5)
            )

            self.assertTrue(
                coordinator._latest_closed_covers_poll_slot(  # noqa: SLF001
                    bitunix_closed, later_slot
                )
            )
            coordinator.stop()

    def test_slow_market_data_does_not_block_tick_position_management(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            request_started = threading.Event()
            allow_request_to_finish = threading.Event()

            def slow_fetch_recent_klines(*args, **kwargs):
                request_started.set()
                allow_request_to_finish.wait(timeout=2)
                return []

            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(
                    tmpdir,
                    candle_ready_delay_seconds=0,
                    execution_interval_minutes=1,
                ),
            )
            coordinator._binance_signal_client.fetch_recent_klines = slow_fetch_recent_klines  # type: ignore[method-assign]  # noqa: SLF001,E501
            now = datetime(2026, 1, 1, 2, tzinfo=timezone.utc)

            started_at = time.monotonic()
            coordinator._maybe_process_new_candles(now)  # noqa: SLF001
            self.assertLess(time.monotonic() - started_at, 0.1)
            self.assertTrue(request_started.wait(timeout=0.1))

            started_at = time.monotonic()
            coordinator._on_tick(now)  # noqa: SLF001
            self.assertLess(time.monotonic() - started_at, 0.1)

            allow_request_to_finish.set()
            coordinator.stop()

    def test_unfavorable_first_live_observation_discards_setup(self) -> None:
        cases = (("long", 101.0), ("short", 99.0))
        for direction, price in cases:
            with self.subTest(direction=direction), TemporaryDirectory() as tmpdir:
                exchange = _FakeExchange()
                exchange.price = price
                coordinator = EmaAvwapPullbackLiveCoordinator(
                    exchange=exchange,
                    config=self._persistent_config(tmpdir),
                )
                setup = _SetupState(
                    symbol="ETHUSDT",
                    direction=direction,  # type: ignore[arg-type]
                    anchor_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    detected_time=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
                    consecutive_count=1,
                    is_waiting_for_cross=True,
                )
                key = coordinator._setup_key(  # noqa: SLF001
                    "ETHUSDT", direction  # type: ignore[arg-type]
                )
                coordinator._active_setups[key] = setup  # noqa: SLF001
                coordinator._last_snapshot_by_symbol["ETHUSDT"] = _snapshot()  # noqa: SLF001
                coordinator._build_live_avwap_snapshot = lambda *args: (  # type: ignore[method-assign]  # noqa: SLF001,E501
                    _snapshot(),
                    _short_avwap(),
                )

                coordinator._process_live_setup_crosses(  # noqa: SLF001
                    datetime(2026, 1, 1, 2, tzinfo=timezone.utc)
                )

                self.assertNotIn(key, coordinator._active_setups)  # noqa: SLF001
                self.assertEqual(exchange.open_orders, [])

    def test_setup_is_discarded_after_maximum_age_or_ema_invalidation(self) -> None:
        with TemporaryDirectory() as tmpdir:
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=_FakeExchange(),
                config=self._persistent_config(tmpdir, max_setup_age_bars=3),
            )
            first = _candle(offset=0, open=99.0, high=101.0, low=98.0, close=100.0)
            latest = _candle(offset=5, open=100.0, high=101.0, low=88.0, close=89.0)
            snapshot = _SymbolSnapshot(
                symbol="ETHUSDT",
                timeframe="1h",
                timeframe_minutes=60,
                candles=(first, latest),
                candle_index=1,
                candle=latest,
                previous_candle=first,
                ema_value=90.0,
                tpv_prefix=(0.0,),
                vol_prefix=(0.0,),
                tpv2_prefix=(0.0,),
            )
            setup = _SetupState(
                symbol="ETHUSDT",
                direction="long",
                anchor_time=first.open_time,
                detected_time=first.close_time,
                consecutive_count=1,
            )
            key = coordinator._setup_key("ETHUSDT", "long")  # noqa: SLF001
            coordinator._active_setups[key] = setup  # noqa: SLF001

            self.assertFalse(
                coordinator._process_pending_setup(  # noqa: SLF001
                    "long", snapshot, datetime(2026, 1, 1, 6, tzinfo=timezone.utc)
                )
            )
            self.assertNotIn(key, coordinator._active_setups)  # noqa: SLF001

            recent_snapshot = replace(
                snapshot,
                candle=replace(latest, close_time=first.close_time + timedelta(hours=1)),
                candles=(first, replace(latest, close_time=first.close_time + timedelta(hours=1))),
            )
            recent_setup = replace(setup, detected_time=first.close_time)
            coordinator._active_setups[key] = recent_setup  # noqa: SLF001

            self.assertFalse(
                coordinator._process_pending_setup(  # noqa: SLF001
                    "long",
                    recent_snapshot,
                    datetime(2026, 1, 1, 2, tzinfo=timezone.utc),
                )
            )
            self.assertNotIn(key, coordinator._active_setups)  # noqa: SLF001

    def test_entry_deviation_is_bounded_and_slippage_is_submitted_as_taker_limit(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            exchange.price = 98.5
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(tmpdir, max_entry_deviation_pct=1.0),
            )
            setup = _SetupState(
                symbol="ETHUSDT",
                direction="long",
                anchor_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                detected_time=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
                consecutive_count=1,
            )
            self.assertIsNone(
                coordinator._build_entry_candidate(  # noqa: SLF001
                    setup=setup,
                    snapshot=_snapshot(),
                    avwap=_short_avwap(),
                    cross=_CrossDecision(True, "live_tick"),
                    current_price=exchange.price,
                )
            )

            exchange.price = 99.5
            coordinator.stop()
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(
                    tmpdir,
                    entry_slippage_pct=0.01,
                    max_entry_deviation_pct=1.0,
                    taker_fee_pct=0.0006,
                ),
            )
            candidate = coordinator._build_entry_candidate(  # noqa: SLF001
                setup=setup,
                snapshot=_snapshot(),
                avwap=_short_avwap(),
                cross=_CrossDecision(True, "live_tick"),
                current_price=exchange.price,
            )

            self.assertIsNotNone(candidate)
            assert candidate is not None
            self.assertAlmostEqual(candidate.order_price, 101.0)
            self.assertAlmostEqual(candidate.sizing.entry_fee_per_unit, 0.0606)
            self.assertTrue(
                coordinator._queue_entry_candidate(  # noqa: SLF001
                    candidate, datetime(2026, 1, 1, 2, tzinfo=timezone.utc)
                )
            )
            self.assertEqual(exchange.open_orders[0][3], 101.0)

    def test_latest_closed_candle_uses_timestamp_not_rest_row_position(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(tmpdir),
            )
            candles = [
                _candle(offset=0, open=98.0, high=101.0, low=97.0, close=100.0),
                _candle(offset=1, open=100.0, high=102.0, low=99.0, close=101.0),
                _candle(offset=2, open=101.0, high=103.0, low=100.0, close=102.0),
            ]
            coordinator._fetch_binance_signal_candles = lambda *args: candles  # type: ignore[method-assign]  # noqa: SLF001,E501

            latest = coordinator._fetch_latest_closed_candle("ETHUSDT")  # noqa: SLF001

            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.open_time, candles[-1].open_time)
            self.assertEqual(exchange.kline_requests, [])
            coordinator.stop()

    def test_live_avwap_uses_fresh_binance_rest_forming_candle(self) -> None:
        with TemporaryDirectory() as tmpdir:
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=_FakeExchange(),
                config=self._persistent_config(tmpdir),
            )
            snapshot = _snapshot()
            forming = _candle(
                offset=2,
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
            )
            coordinator._cache_forming_binance_candle("ETHUSDT", forming)  # noqa: SLF001

            candles = coordinator._live_avwap_candles(snapshot)  # noqa: SLF001

            self.assertEqual(candles[-1].open_time, forming.open_time)
            self.assertEqual(candles[-1].close, forming.close)
            with coordinator._forming_binance_candle_lock:  # noqa: SLF001
                cached, _ = coordinator._forming_binance_candle_by_symbol["ETHUSDT"]  # noqa: SLF001
                coordinator._forming_binance_candle_by_symbol["ETHUSDT"] = (  # noqa: SLF001
                    cached,
                    time.monotonic()
                    - coordinator._cfg.live_kline_stale_seconds  # noqa: SLF001
                    - 0.1,
                )

            self.assertEqual(
                coordinator._live_avwap_candles(snapshot),  # noqa: SLF001
                tuple(snapshot.candles),
            )
            coordinator.stop()

    def test_trailing_tick_emulation_uses_latest_ready_rest_candle(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(
                    tmpdir,
                    use_trailing_tick_emulation=True,
                    trailing_tick_timeframe="1h",
                ),
            )
            exchange.klines = [
                _binance_row(
                    _candle(
                        offset=0, open=98.0, high=101.0, low=97.0, close=100.0
                    )
                ),
                _binance_row(
                    _candle(
                        offset=1, open=100.0, high=103.0, low=99.0, close=102.0
                    )
                ),
            ]

            price = coordinator._latest_price_for_trailing("ETHUSDT")  # noqa: SLF001

            self.assertEqual(price, 102.0)
            self.assertEqual(exchange.kline_requests, [("ETHUSDT", "1h", 2)])

    def test_bitunix_websocket_kline_message_maps_to_current_update(self) -> None:
        received: list[KlineUpdate] = []
        stream = BitunixKlineStream(
            symbols=("FETUSDT",),
            interval="1h",
            on_kline=received.append,
            logger=logging.getLogger("test.bitunix.kline"),
        )

        stream._on_message(  # noqa: SLF001
            None,
            json.dumps(
                {
                    "ch": "market_kline_60min",
                    "symbol": "FETUSDT",
                    "ts": 1787825000409,
                    "data": {
                        "o": "0.1671",
                        "c": "0.1670",
                        "h": "0.1673",
                        "l": "0.1668",
                        "b": "9670",
                        "q": "1615.0708",
                    },
                }
            ),
        )

        self.assertEqual(kline_channel_for_interval("1h"), "market_kline_60min")
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].symbol, "FETUSDT")
        self.assertEqual(received[0].interval, "1h")
        self.assertAlmostEqual(received[0].base_volume, 9670.0)

    def test_bitunix_websocket_sends_application_heartbeat_after_subscribing(self) -> None:
        class _Socket:
            def __init__(self) -> None:
                self.sent: list[str] = []

            def send(self, payload: str) -> None:
                self.sent.append(payload)

            def close(self) -> None:
                return None

        stream = BitunixKlineStream(
            symbols=("FETUSDT",),
            interval="1h",
            on_kline=lambda _update: None,
            logger=logging.getLogger("test.bitunix.kline"),
            heartbeat_interval_seconds=0.01,
        )
        socket = _Socket()
        try:
            stream._on_open(socket)  # noqa: SLF001
            deadline = time.monotonic() + 0.5
            while len(socket.sent) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
        finally:
            stream._stop_application_heartbeat(socket)  # noqa: SLF001

        subscription, heartbeat = [json.loads(message) for message in socket.sent[:2]]
        self.assertEqual(subscription["op"], "subscribe")
        self.assertEqual(heartbeat["op"], "ping")
        self.assertIsInstance(heartbeat["ping"], int)

    def test_bitunix_reconnect_delay_resets_after_stable_subscription(self) -> None:
        stream = BitunixKlineStream(
            symbols=("FETUSDT",),
            interval="1h",
            on_kline=lambda _update: None,
            logger=logging.getLogger("test.bitunix.kline"),
            reconnect_backoff_reset_seconds=15.0,
        )

        self.assertEqual(
            stream._reconnect_delay_after_disconnect(  # noqa: SLF001
                16.0, subscription_age_seconds=60.0
            ),
            1.0,
        )
        self.assertEqual(
            stream._reconnect_delay_after_disconnect(  # noqa: SLF001
                16.0, subscription_age_seconds=1.0
            ),
            30.0,
        )

    def test_telegram_proxy_falls_back_to_generic_https_proxy(self) -> None:
        args = Namespace(
            telegram_proxy=None,
            proxy=None,
            https_proxy="http://127.0.0.1:12334",
            http_proxy="http://127.0.0.1:8080",
        )

        self.assertEqual(
            live_trading_shared._resolve_telegram_proxy(args),  # noqa: SLF001
            "http://127.0.0.1:12334",
        )

    def test_telegram_proxy_keeps_explicit_telegram_proxy_first(self) -> None:
        args = Namespace(
            telegram_proxy="http://127.0.0.1:7897",
            proxy="http://127.0.0.1:12334",
            https_proxy="http://127.0.0.1:12335",
            http_proxy="http://127.0.0.1:12336",
        )

        self.assertEqual(
            live_trading_shared._resolve_telegram_proxy(args),  # noqa: SLF001
            "http://127.0.0.1:7897",
        )

    def test_shared_builder_forwards_max_position_size_percentage(self) -> None:
        shared_config = LiveTradingConfig(
            exchange_name="bitunix",
            api_key="key",
            api_secret="secret",
            testnet=False,
            strategy_name="ema_avwap_pullback",
            timeframe="1h",
            max_entry_notional_usdt=500.0,
            max_position_size_pct=7.5,
        )
        account_lock_file = Path("/tmp/ema-avwap-builder-test.lock")
        args = Namespace(
            symbols="ETHUSDT",
            position_notional_pct=3.0,
            rigid_stop_loss_pct=3.0,
            ema_avwap_account_lock_file=account_lock_file,
            max_entry_reprice_pct=0.25,
            entry_mode="close",
            exit_mode="live",
            exit_band="band_2",
        )

        config = live_trading_shared.build_ema_avwap_pullback_config(
            args, shared_config
        )

        self.assertEqual(config.max_position_size_pct, 7.5)
        self.assertEqual(config.position_notional_pct, 3.0)
        self.assertEqual(config.max_entry_reprice_pct, 0.25)
        self.assertEqual(config.account_lock_file, account_lock_file)
        self.assertIs(config.entry_mode, EntryMode.CLOSE)
        self.assertIs(config.exit_mode, ExitMode.LIVE)
        self.assertIs(config.exit_band, ExitBand.BAND_2)
        self.assertEqual(config.candle_ready_delay_seconds, 0.0)

    def test_entry_exit_settings_are_independently_validated(self) -> None:
        for entry_mode in EntryMode:
            for exit_mode in ExitMode:
                for exit_band in ExitBand:
                    with self.subTest(
                        entry_mode=entry_mode.value,
                        exit_mode=exit_mode.value,
                        exit_band=exit_band.value,
                    ):
                        config = EmaAvwapPullbackLiveConfig(
                            rigid_stop_loss_pct=3.0,
                            entry_mode=entry_mode,
                            exit_mode=exit_mode,
                            exit_band=exit_band,
                        )
                        self.assertIs(config.entry_mode, entry_mode)
                        self.assertIs(config.exit_mode, exit_mode)
                        self.assertIs(config.exit_band, exit_band)

        for field, value in (
            ("entry_mode", "invalid"),
            ("exit_mode", "invalid"),
            ("exit_band", "invalid"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                EmaAvwapPullbackLiveConfig(
                    rigid_stop_loss_pct=3.0,
                    **{field: value},
                )

    def test_removed_combined_mode_is_rejected_in_live_env_config(self) -> None:
        config = live_trading_shared._load_ema_avwap_pullback_env_config(  # noqa: SLF001
            lambda *keys: "legacy" if "ENTRY_EXIT_MODE" in keys else ""
        )

        with self.assertRaisesRegex(ValueError, "ENTRY_EXIT_MODE has been removed"):
            live_trading_shared._apply_ema_avwap_pullback_env_defaults(  # noqa: SLF001
                Namespace(),
                config,
            )

    def test_stale_bulk_snapshot_does_not_mark_newer_candle_processed(self) -> None:
        with TemporaryDirectory() as tmpdir:
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=_FakeExchange(),
                config=self._persistent_config(tmpdir),
            )
            latest_closed = _candle(
                offset=2,
                open=100.0,
                high=102.0,
                low=99.0,
                close=101.0,
            )
            stale_snapshot = _snapshot()
            coordinator._fetch_latest_closed_candle = lambda symbol: latest_closed  # type: ignore[method-assign]  # noqa: SLF001,E501
            now = datetime(2026, 1, 1, 3, tzinfo=timezone.utc)

            candles = (*stale_snapshot.candles, latest_closed)
            tpv_prefix, vol_prefix, tpv2_prefix = (
                coordinator._build_avwap_prefixes(candles)  # noqa: SLF001
            )
            fresh_snapshot = replace(
                stale_snapshot,
                candles=candles,
                candle_index=2,
                candle=latest_closed,
                previous_candle=stale_snapshot.candle,
                tpv_prefix=tpv_prefix,
                vol_prefix=vol_prefix,
                tpv2_prefix=tpv2_prefix,
            )
            build_calls = 0

            def build_snapshot(symbol: str) -> _SymbolSnapshot:
                nonlocal build_calls
                del symbol
                build_calls += 1
                return stale_snapshot if build_calls == 1 else fresh_snapshot

            coordinator._build_snapshot = build_snapshot  # type: ignore[method-assign]  # noqa: SLF001,E501

            self._drain_market_data_until(
                coordinator,
                now,
                lambda: "ETHUSDT" in coordinator._last_closed_candle_time_by_symbol,  # noqa: SLF001,E501
            )

            self.assertGreaterEqual(build_calls, 2)
            self.assertEqual(
                coordinator._last_closed_candle_time_by_symbol["ETHUSDT"],  # noqa: SLF001
                latest_closed.close_time,
            )
            coordinator.stop()

    def test_non_marketable_closed_bar_cross_is_not_left_as_limit_order(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            exchange.balance = 1_000.0
            exchange.price = 101.0
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(tmpdir),
            )
            first = _candle(offset=0, open=99.0, high=101.0, low=98.0, close=100.5)
            pullback = _candle(
                offset=1, open=101.0, high=102.0, low=99.0, close=100.8
            )
            setup = _SetupState(
                symbol="ETHUSDT",
                direction="long",
                anchor_time=first.open_time,
                detected_time=first.close_time,
                consecutive_count=1,
            )
            avwap = _AvwapSnapshot(
                anchor_index=0,
                anchor_time=first.open_time,
                candle_index=1,
                vwap=100.0,
                stdev=2.5,
                upper1=102.5,
                lower1=97.5,
                upper2=105.0,
                lower2=95.0,
                upper3=107.5,
                lower3=92.5,
            )
            snapshot = _SymbolSnapshot(
                symbol="ETHUSDT",
                timeframe="1h",
                timeframe_minutes=60,
                candles=(first, pullback),
                candle_index=1,
                candle=pullback,
                previous_candle=first,
                ema_value=110.0,
                tpv_prefix=(0.0,),
                vol_prefix=(0.0,),
                tpv2_prefix=(0.0,),
            )

            candidate = coordinator._build_entry_candidate(  # noqa: SLF001
                setup=setup,
                snapshot=snapshot,
                avwap=avwap,
                cross=_CrossDecision(True, "intrabar"),
            )

            self.assertIsNone(candidate)

    def test_small_non_marketable_move_is_repriced_and_submitted(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(tmpdir, max_entry_reprice_pct=0.5),
            )
            candidate = _candidate()
            now = datetime(2026, 1, 1, tzinfo=timezone.utc)

            activate_due_entries = coordinator._activate_due_entries  # noqa: SLF001
            coordinator._activate_due_entries = lambda _now: None  # type: ignore[method-assign]  # noqa: SLF001,E501
            self.assertTrue(
                coordinator._queue_entry_candidate(candidate, now)  # noqa: SLF001
            )
            coordinator._activate_due_entries = activate_due_entries  # type: ignore[method-assign]  # noqa: SLF001,E501

            exchange.price = 100.25
            activate_due_entries(now)

            pending = coordinator._state.pending_entries["ETHUSDT:LONG"]  # noqa: SLF001
            self.assertEqual(len(exchange.open_orders), 1)
            self.assertEqual(exchange.open_orders[0][3], 100.25)
            self.assertEqual(pending.entry_price, 100.25)
            self.assertEqual(pending.status, "PLACED")
            self.assertIn("repriced marketable limit", pending.notes)
            coordinator.stop()

    def test_large_non_marketable_move_is_not_repriced(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            exchange.price = 100.51
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(tmpdir, max_entry_reprice_pct=0.5),
            )
            now = datetime(2026, 1, 1, tzinfo=timezone.utc)
            pending = PendingEntryRecord(
                order_key="ETHUSDT:LONG",
                symbol="ETHUSDT",
                side=PositionSide.LONG,
                entry_price=100.25,
                quantity=1.0,
                leverage=2,
                margin_mode=MarginMode.ISOLATED,
                risk_amount=5.0,
                stop_for_risk=97.5,
                created_time=now,
                signal_time=now,
                activate_time=now,
                client_id="emaavwap-test-id",
            )
            coordinator._pending_meta_by_key[pending.order_key] = _PendingEntryMeta(  # noqa: SLF001
                candidate=_candidate()
            )

            with self.assertRaisesRegex(
                RuntimeError, "max_reprice=0.5000%"
            ):
                coordinator._ensure_pending_limit_is_marketable(pending)  # noqa: SLF001

            self.assertEqual(pending.entry_price, 100.25)
            coordinator.stop()

    def test_filled_position_confirms_initial_rigid_stop(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            exchange.positions = [
                Position(
                    symbol="ETHUSDT",
                    side=PositionSide.LONG,
                    size=1.0,
                    entry_price=100.0,
                    leverage=2,
                    margin_mode=MarginMode.ISOLATED,
                    unrealized_pnl=0.0,
                    position_id="pos-1",
                )
            ]
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(
                    tmpdir,
                    emergency_close_on_stop_failure=True,
                ),
            )
            now = datetime(2026, 1, 1, tzinfo=timezone.utc)
            pending = PendingEntryRecord(
                order_key="ETHUSDT:LONG",
                symbol="ETHUSDT",
                side=PositionSide.LONG,
                entry_price=100.0,
                quantity=1.0,
                leverage=2,
                margin_mode=MarginMode.ISOLATED,
                risk_amount=5.0,
                stop_for_risk=97.5,
                created_time=now,
                signal_time=now,
                activate_time=now,
                order_id="order-1",
                status="PLACED",
            )
            coordinator._state.pending_entries[pending.order_key] = pending  # noqa: SLF001
            coordinator._pending_meta_by_key[pending.order_key] = _PendingEntryMeta(  # noqa: SLF001
                candidate=_candidate()
            )

            coordinator._sync_positions(now)  # noqa: SLF001

            self.assertEqual(exchange.stop_updates, [97.5])
            self.assertEqual(exchange.close_calls, [])
            self.assertEqual(
                coordinator._state.active_positions["ETHUSDT"].stop_loss,  # noqa: SLF001
                97.5,
            )

    def test_close_error_marks_position_closed_when_exchange_confirms_it_is_flat(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            exchange.price = 97.0
            exchange.positions = [
                Position(
                    symbol="ETHUSDT",
                    side=PositionSide.LONG,
                    size=1.0,
                    entry_price=100.0,
                    leverage=2,
                    margin_mode=MarginMode.ISOLATED,
                    unrealized_pnl=0.0,
                    position_id="pos-1",
                )
            ]
            exchange.close_error = RuntimeError(
                "Bitunix error code 20008: Insufficient amount"
            )
            exchange.close_error_removes_position = True
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(tmpdir),
            )
            now = datetime(2026, 1, 1, tzinfo=timezone.utc)
            position = PositionRecord(
                position_id="pos-1",
                symbol="ETHUSDT",
                side=PositionSide.LONG,
                entry_time=now,
                entry_price=100.0,
                quantity=1.0,
                leverage=2,
                margin_mode=MarginMode.ISOLATED,
                take_profit=None,
                stop_loss=97.5,
                strategy="ema_avwap_pullback",
            )
            coordinator._state.active_positions[position.symbol] = position  # noqa: SLF001

            coordinator._close_position(position, now, "Rigid stop loss")  # noqa: SLF001

            self.assertEqual(exchange.close_calls, [("ETHUSDT", PositionSide.LONG)])
            self.assertEqual(position.status, "CLOSED")
            self.assertEqual(position.exit_price, 97.0)
            self.assertNotIn("ETHUSDT", coordinator._state.active_positions)  # noqa: SLF001
            self.assertIn("position confirmed absent", position.notes)

    def test_filled_position_without_confirmed_stop_is_emergency_closed(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            exchange.stop_update_ok = False
            exchange.positions = [
                Position(
                    symbol="ETHUSDT",
                    side=PositionSide.LONG,
                    size=1.0,
                    entry_price=100.0,
                    leverage=2,
                    margin_mode=MarginMode.ISOLATED,
                    unrealized_pnl=0.0,
                    position_id="pos-1",
                )
            ]
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(tmpdir),
            )
            now = datetime(2026, 1, 1, tzinfo=timezone.utc)
            pending = PendingEntryRecord(
                order_key="ETHUSDT:LONG",
                symbol="ETHUSDT",
                side=PositionSide.LONG,
                entry_price=100.0,
                quantity=1.0,
                leverage=2,
                margin_mode=MarginMode.ISOLATED,
                risk_amount=5.0,
                stop_for_risk=97.5,
                created_time=now,
                signal_time=now,
                activate_time=now,
                order_id="order-1",
                status="PLACED",
            )
            coordinator._state.pending_entries[pending.order_key] = pending  # noqa: SLF001
            coordinator._pending_meta_by_key[pending.order_key] = _PendingEntryMeta(  # noqa: SLF001
                candidate=_candidate()
            )

            coordinator._sync_positions(now)  # noqa: SLF001

            self.assertEqual(exchange.stop_updates, [97.5])
            self.assertEqual(exchange.close_calls, [("ETHUSDT", PositionSide.LONG)])
            self.assertNotIn("ETHUSDT", coordinator._state.active_positions)  # noqa: SLF001

    def test_partial_fill_keeps_entry_order_until_terminal_reconciliation(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            exchange.positions = [
                Position(
                    symbol="ETHUSDT",
                    side=PositionSide.LONG,
                    size=0.4,
                    entry_price=100.0,
                    leverage=2,
                    margin_mode=MarginMode.ISOLATED,
                    unrealized_pnl=0.0,
                    position_id="pos-1",
                )
            ]
            exchange.order_statuses["order-1"] = {
                "orderId": "order-1",
                "status": "PART_FILLED",
            }
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(tmpdir),
            )
            now = datetime(2026, 1, 1, tzinfo=timezone.utc)
            pending = PendingEntryRecord(
                order_key="ETHUSDT:LONG",
                symbol="ETHUSDT",
                side=PositionSide.LONG,
                entry_price=100.0,
                quantity=1.0,
                leverage=2,
                margin_mode=MarginMode.ISOLATED,
                risk_amount=5.0,
                stop_for_risk=97.5,
                created_time=now,
                signal_time=now,
                activate_time=now,
                order_id="order-1",
                status="PLACED",
            )
            coordinator._state.pending_entries[pending.order_key] = pending  # noqa: SLF001
            coordinator._pending_meta_by_key[pending.order_key] = _PendingEntryMeta(  # noqa: SLF001
                candidate=_candidate()
            )

            with self.assertLogs(level="WARNING") as logs:
                coordinator._sync_positions(now)  # noqa: SLF001

            self.assertEqual(
                coordinator._state.active_positions["ETHUSDT"].quantity,  # noqa: SLF001
                0.4,
            )
            self.assertIn("ETHUSDT:LONG", coordinator._state.pending_entries)  # noqa: SLF001
            self.assertIn("unconfirmed/partial fill", "\n".join(logs.output))

            exchange.positions[0] = replace(exchange.positions[0], size=1.0)
            exchange.order_statuses["order-1"]["status"] = "FILLED"
            coordinator._sync_positions(now + timedelta(seconds=5))  # noqa: SLF001

            self.assertEqual(
                coordinator._state.active_positions["ETHUSDT"].quantity,  # noqa: SLF001
                1.0,
            )
            self.assertNotIn("ETHUSDT:LONG", coordinator._state.pending_entries)  # noqa: SLF001

    def test_pending_entry_recovers_ambiguous_submission_by_client_id(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(tmpdir),
            )
            now = datetime(2026, 1, 1, tzinfo=timezone.utc)
            pending = PendingEntryRecord(
                order_key="ETHUSDT:LONG",
                symbol="ETHUSDT",
                side=PositionSide.LONG,
                entry_price=100.0,
                quantity=1.0,
                leverage=2,
                margin_mode=MarginMode.ISOLATED,
                risk_amount=5.0,
                stop_for_risk=97.5,
                created_time=now,
                signal_time=now,
                activate_time=now,
                status="PENDING",
                client_id="emaavwap-recovery-id",
            )
            exchange.order_statuses[pending.client_id] = {
                "orderId": "order-recovered",
                "status": "NEW",
            }
            coordinator._state.pending_entries[pending.order_key] = pending  # noqa: SLF001

            coordinator._activate_due_entries(now)  # noqa: SLF001

            self.assertEqual(exchange.open_orders, [])
            self.assertEqual(pending.order_id, "order-recovered")
            self.assertEqual(pending.status, "PLACED")

    def test_terminal_unfilled_entry_removal_is_persisted(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = self._persistent_config(tmpdir)
            exchange = _FakeExchange()
            exchange.order_statuses["order-1"] = {
                "orderId": "order-1",
                "status": "CANCELLED",
            }
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=config,
            )
            now = datetime(2026, 1, 1, tzinfo=timezone.utc)
            pending = PendingEntryRecord(
                order_key="ETHUSDT:LONG",
                symbol="ETHUSDT",
                side=PositionSide.LONG,
                entry_price=100.0,
                quantity=1.0,
                leverage=2,
                margin_mode=MarginMode.ISOLATED,
                risk_amount=5.0,
                stop_for_risk=97.5,
                created_time=now,
                signal_time=now,
                activate_time=now,
                order_id="order-1",
                status="PLACED",
                client_id="emaavwap-terminal-id",
            )
            coordinator._state.pending_entries[pending.order_key] = pending  # noqa: SLF001

            coordinator._sync_positions(now)  # noqa: SLF001

            self.assertNotIn("ETHUSDT:LONG", coordinator._state.pending_entries)  # noqa: SLF001
            coordinator.stop()
            restarted = EmaAvwapPullbackLiveCoordinator(
                exchange=_FakeExchange(),
                config=config,
            )
            self.assertNotIn("ETHUSDT:LONG", restarted._state.pending_entries)  # noqa: SLF001

    def test_legacy_pending_entry_without_identifiers_is_never_retried(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(tmpdir),
            )
            now = datetime(2026, 1, 1, tzinfo=timezone.utc)
            pending = PendingEntryRecord(
                order_key="ETHUSDT:LONG",
                symbol="ETHUSDT",
                side=PositionSide.LONG,
                entry_price=100.0,
                quantity=1.0,
                leverage=2,
                margin_mode=MarginMode.ISOLATED,
                risk_amount=5.0,
                stop_for_risk=97.5,
                created_time=now,
                signal_time=now,
                activate_time=now,
                status="PENDING",
            )
            coordinator._state.pending_entries[pending.order_key] = pending  # noqa: SLF001

            coordinator._activate_due_entries(now)  # noqa: SLF001

            self.assertEqual(exchange.open_orders, [])
            self.assertEqual(pending.status, "ERROR")
            coordinator._cancel_stale_entries(  # noqa: SLF001
                _snapshot(), now + timedelta(hours=2)
            )
            self.assertIn("ETHUSDT:LONG", coordinator._state.pending_entries)  # noqa: SLF001

    def test_position_fetch_error_does_not_mark_active_position_closed(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            exchange.positions_exc = RuntimeError("temporary Bitunix outage")
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(tmpdir),
            )
            position = PositionRecord(
                position_id="pos-1",
                symbol="ETHUSDT",
                side=PositionSide.LONG,
                entry_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                entry_price=100.0,
                quantity=1.0,
                leverage=2,
                margin_mode=MarginMode.ISOLATED,
                take_profit=None,
                stop_loss=95.0,
                strategy="ema_avwap_pullback",
            )
            coordinator._state.active_positions[position.symbol] = position  # noqa: SLF001

            coordinator._sync_positions(datetime(2026, 1, 1, tzinfo=timezone.utc))  # noqa: SLF001

            self.assertIs(
                coordinator._state.active_positions["ETHUSDT"],  # noqa: SLF001
                position,
            )
            self.assertEqual(position.status, "OPEN")

    def test_waiting_setup_keeps_existing_setup_by_default(self) -> None:
        with TemporaryDirectory() as tmpdir:
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=_FakeExchange(),
                config=self._persistent_config(tmpdir),
            )
            old_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
            new_time = datetime(2026, 1, 1, 1, tzinfo=timezone.utc)
            old_setup = _SetupState(
                symbol="ETHUSDT",
                direction="long",
                anchor_time=old_time,
                detected_time=old_time,
                consecutive_count=4,
                is_waiting_for_cross=True,
            )
            new_setup = _SetupState(
                symbol="ETHUSDT",
                direction="long",
                anchor_time=new_time,
                detected_time=new_time,
                consecutive_count=4,
            )
            key = coordinator._setup_key("ETHUSDT", "long")  # noqa: SLF001
            coordinator._active_setups[key] = old_setup  # noqa: SLF001

            result = coordinator._replace_or_store_setup(  # noqa: SLF001
                new_setup, None  # type: ignore[arg-type]
            )

            self.assertIs(result, old_setup)
            self.assertIs(coordinator._active_setups[key], old_setup)  # noqa: SLF001

    def test_live_entry_works_with_either_exit_band_on_first_tick(self) -> None:
        for exit_band in ExitBand:
            for direction, first_price, expected_side in (
                ("long", 100.0, PositionSide.LONG),
                ("long", 99.0, PositionSide.LONG),
                ("short", 100.0, PositionSide.SHORT),
                ("short", 101.0, PositionSide.SHORT),
            ):
                with self.subTest(
                    exit_band=exit_band.value,
                    direction=direction,
                    first_price=first_price,
                ), TemporaryDirectory() as tmpdir:
                    exchange = _FakeExchange()
                    exchange.price = first_price
                    coordinator = EmaAvwapPullbackLiveCoordinator(
                        exchange=exchange,
                        config=self._persistent_config(
                            tmpdir,
                            entry_mode=EntryMode.LIVE,
                            exit_band=exit_band,
                        ),
                    )
                    setup = _SetupState(
                        symbol="ETHUSDT",
                        direction=direction,  # type: ignore[arg-type]
                        anchor_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                        detected_time=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
                        consecutive_count=1,
                        detected_avwap=_short_avwap(),
                        is_waiting_for_cross=True,
                    )
                    key = coordinator._setup_key(  # noqa: SLF001
                        "ETHUSDT", direction  # type: ignore[arg-type]
                    )
                    coordinator._active_setups[key] = setup  # noqa: SLF001
                    coordinator._last_snapshot_by_symbol["ETHUSDT"] = _snapshot()  # noqa: SLF001
                    coordinator._build_live_avwap_snapshot = lambda *args: (  # type: ignore[method-assign]  # noqa: SLF001,E501
                        _snapshot(),
                        _short_avwap(),
                    )

                    coordinator._process_live_setup_crosses(  # noqa: SLF001
                        datetime(2026, 1, 1, 2, tzinfo=timezone.utc)
                    )

                    self.assertNotIn(key, coordinator._active_setups)  # noqa: SLF001
                    self.assertEqual(len(exchange.open_orders), 1)
                    self.assertEqual(exchange.open_orders[0][1], expected_side)
                    self.assertEqual(exchange.open_orders[0][3], 100.0)
                    pending_meta = next(  # noqa: SLF001
                        iter(coordinator._pending_meta_by_key.values())
                    )
                    self.assertIs(pending_meta.candidate.entry_mode, EntryMode.LIVE)
                    self.assertIs(pending_meta.candidate.exit_band, exit_band)

    def test_closed_bar_does_not_execute_live_touch_mode_entry(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            exchange.price = 100.5
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(tmpdir),
            )
            first = _candle(offset=0, open=99.0, high=99.5, low=98.0, close=99.0)
            pullback = _candle(
                offset=1, open=99.0, high=101.0, low=98.5, close=100.5
            )
            setup = _SetupState(
                symbol="ETHUSDT",
                direction="short",
                anchor_time=first.open_time,
                detected_time=first.close_time,
                consecutive_count=1,
                detected_avwap=_short_avwap(),
                is_waiting_for_cross=True,
            )
            key = coordinator._setup_key("ETHUSDT", "short")  # noqa: SLF001
            coordinator._active_setups[key] = setup  # noqa: SLF001
            snapshot = _SymbolSnapshot(
                symbol="ETHUSDT",
                timeframe="1h",
                timeframe_minutes=60,
                candles=(first, pullback),
                candle_index=1,
                candle=pullback,
                previous_candle=first,
                ema_value=110.0,
                tpv_prefix=(0.0,),
                vol_prefix=(0.0,),
                tpv2_prefix=(0.0,),
            )
            coordinator._build_avwap_snapshot = lambda **_: _short_avwap()  # type: ignore[method-assign]  # noqa: SLF001,E501
            coordinator._build_live_avwap_snapshot = lambda *args: (  # type: ignore[method-assign]  # noqa: SLF001,E501
                snapshot,
                _short_avwap(),
            )

            queued = coordinator._process_pending_setup(  # noqa: SLF001
                "short",
                snapshot,
                datetime(2026, 1, 1, 2, tzinfo=timezone.utc),
            )

            self.assertFalse(queued)
            self.assertIn(key, coordinator._active_setups)  # noqa: SLF001
            self.assertTrue(coordinator._active_setups[key].is_waiting_for_cross)  # noqa: SLF001,E501
            self.assertEqual(exchange.open_orders, [])

    def test_live_entry_works_with_either_exit_band_on_crossing_tick(self) -> None:
        for exit_band in ExitBand:
            with self.subTest(exit_band=exit_band.value), TemporaryDirectory() as tmpdir:
                exchange = _FakeExchange()
                exchange.price = 99.0
                coordinator = EmaAvwapPullbackLiveCoordinator(
                    exchange=exchange,
                    config=self._persistent_config(
                        tmpdir,
                        entry_mode=EntryMode.LIVE,
                        exit_band=exit_band,
                    ),
                )
                setup = _SetupState(
                    symbol="ETHUSDT",
                    direction="short",
                    anchor_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    detected_time=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
                    consecutive_count=1,
                    detected_avwap=_short_avwap(),
                    is_waiting_for_cross=True,
                )
                key = coordinator._setup_key("ETHUSDT", "short")  # noqa: SLF001
                coordinator._active_setups[key] = setup  # noqa: SLF001
                coordinator._last_snapshot_by_symbol["ETHUSDT"] = _snapshot()  # noqa: SLF001
                coordinator._build_live_avwap_snapshot = lambda *args: (  # type: ignore[method-assign]  # noqa: SLF001,E501
                    _snapshot(),
                    _short_avwap(),
                )
                coordinator._last_price_by_setup_key[key] = 99.0  # noqa: SLF001
                coordinator._last_middle_by_setup_key[key] = 100.0  # noqa: SLF001

                now = datetime(2026, 1, 1, 2, tzinfo=timezone.utc)
                exchange.price = 100.1
                crossing_tick = now + timedelta(seconds=5)
                coordinator._process_live_setup_crosses(crossing_tick)  # noqa: SLF001

                self.assertEqual(len(exchange.open_orders), 1)
                symbol, side, _quantity, price, stop_loss = exchange.open_orders[0]
                self.assertEqual(symbol, "ETHUSDT")
                self.assertEqual(side, PositionSide.SHORT)
                self.assertEqual(price, 100.0)
                self.assertEqual(stop_loss, 103.0)
                self.assertNotIn(key, coordinator._active_setups)  # noqa: SLF001
                pending_meta = next(iter(coordinator._pending_meta_by_key.values()))  # noqa: SLF001
                self.assertEqual(pending_meta.candidate.signal_time, crossing_tick)
                self.assertIs(pending_meta.candidate.entry_mode, EntryMode.LIVE)
                self.assertIs(pending_meta.candidate.exit_band, exit_band)
                self.assertEqual(pending_meta.candidate.entry_trigger_mode, "live_tick")
                self.assertEqual(
                    pending_meta.candidate.exit_band.number,
                    exit_band.number,
                )

    def test_live_cross_compares_price_to_each_ticks_moving_middle(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            exchange.price = 101.0
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(tmpdir),
            )
            setup = _SetupState(
                symbol="ETHUSDT",
                direction="long",
                anchor_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                detected_time=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
                consecutive_count=1,
                detected_avwap=_short_avwap(),
                is_waiting_for_cross=True,
            )
            key = coordinator._setup_key("ETHUSDT", "long")  # noqa: SLF001
            coordinator._active_setups[key] = setup  # noqa: SLF001
            coordinator._last_snapshot_by_symbol["ETHUSDT"] = _snapshot()  # noqa: SLF001
            avwaps = iter(
                (
                    _short_avwap(),
                    replace(_short_avwap(), vwap=102.0),
                )
            )
            coordinator._build_live_avwap_snapshot = lambda *args: (  # type: ignore[method-assign]  # noqa: SLF001,E501
                _snapshot(),
                next(avwaps),
            )
            coordinator._last_price_by_setup_key[key] = 101.0  # noqa: SLF001
            coordinator._last_middle_by_setup_key[key] = 100.0  # noqa: SLF001

            now = datetime(2026, 1, 1, 2, tzinfo=timezone.utc)
            coordinator._process_live_setup_crosses(now)  # noqa: SLF001
            self.assertEqual(exchange.open_orders, [])

            coordinator._process_live_setup_crosses(  # noqa: SLF001
                now + timedelta(seconds=5)
            )

            self.assertEqual(len(exchange.open_orders), 1)
            self.assertEqual(exchange.open_orders[0][1], PositionSide.LONG)
            self.assertEqual(exchange.open_orders[0][3], 102.0)

    def test_live_tick_cross_uses_forming_candle_avwap(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            exchange.price = 99.0
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(tmpdir),
            )
            previous = _candle(offset=0, open=95.0, high=96.0, low=94.0, close=95.0)
            setup_candle = _candle(
                offset=1, open=101.0, high=101.0, low=99.0, close=100.0
            )
            forming = _candle(
                offset=2, open=100.0, high=111.0, low=109.0, close=110.0
            )
            tpv_prefix, vol_prefix, tpv2_prefix = coordinator._build_avwap_prefixes(  # noqa: SLF001
                (previous, setup_candle)
            )
            snapshot = _SymbolSnapshot(
                symbol="ETHUSDT",
                timeframe="1h",
                timeframe_minutes=60,
                candles=(previous, setup_candle),
                candle_index=1,
                candle=setup_candle,
                previous_candle=previous,
                ema_value=120.0,
                tpv_prefix=tpv_prefix,
                vol_prefix=vol_prefix,
                tpv2_prefix=tpv2_prefix,
            )
            setup = _SetupState(
                symbol="ETHUSDT",
                direction="short",
                anchor_time=setup_candle.open_time,
                detected_time=setup_candle.close_time,
                consecutive_count=1,
                detected_avwap=_short_avwap(),
                is_waiting_for_cross=True,
            )
            key = coordinator._setup_key("ETHUSDT", "short")  # noqa: SLF001
            coordinator._active_setups[key] = setup  # noqa: SLF001
            coordinator._last_snapshot_by_symbol["ETHUSDT"] = snapshot  # noqa: SLF001
            coordinator._cache_forming_binance_candle("ETHUSDT", forming)  # noqa: SLF001
            coordinator._last_price_by_setup_key[key] = 99.0  # noqa: SLF001
            coordinator._last_middle_by_setup_key[key] = 100.0  # noqa: SLF001
            now = datetime(2026, 1, 1, 3, tzinfo=timezone.utc)
            coordinator._process_live_setup_crosses(now)  # noqa: SLF001

            self.assertEqual(exchange.open_orders, [])
            self.assertEqual(coordinator._last_price_by_setup_key[key], 99.0)  # noqa: SLF001

            exchange.price = 106.0
            coordinator._process_live_setup_crosses(now + timedelta(seconds=5))  # noqa: SLF001

            self.assertEqual(len(exchange.open_orders), 1)
            self.assertEqual(exchange.open_orders[0][3], 105.0)

    def test_legacy_waiting_setup_uses_live_avwap_without_frozen_value(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            exchange.price = 99.0
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(tmpdir),
            )
            previous = _candle(offset=0, open=95.0, high=96.0, low=94.0, close=95.0)
            setup_candle = _candle(
                offset=1, open=101.0, high=101.0, low=99.0, close=100.0
            )
            tpv_prefix, vol_prefix, tpv2_prefix = coordinator._build_avwap_prefixes(  # noqa: SLF001
                (previous, setup_candle)
            )
            snapshot = _SymbolSnapshot(
                symbol="ETHUSDT",
                timeframe="1h",
                timeframe_minutes=60,
                candles=(previous, setup_candle),
                candle_index=1,
                candle=setup_candle,
                previous_candle=previous,
                ema_value=120.0,
                tpv_prefix=tpv_prefix,
                vol_prefix=vol_prefix,
                tpv2_prefix=tpv2_prefix,
            )
            setup = _SetupState(
                symbol="ETHUSDT",
                direction="short",
                anchor_time=previous.open_time,
                detected_time=setup_candle.close_time,
                consecutive_count=1,
                is_waiting_for_cross=True,
            )
            key = coordinator._setup_key("ETHUSDT", "short")  # noqa: SLF001
            coordinator._active_setups[key] = setup  # noqa: SLF001
            coordinator._last_snapshot_by_symbol["ETHUSDT"] = snapshot  # noqa: SLF001
            coordinator._last_price_by_setup_key[key] = 99.0  # noqa: SLF001
            coordinator._last_middle_by_setup_key[key] = 100.0  # noqa: SLF001

            coordinator._build_live_avwap_snapshot = lambda *args: (  # type: ignore[method-assign]  # noqa: SLF001,E501
                snapshot,
                _short_avwap(),
            )

            now = datetime(2026, 1, 1, 3, tzinfo=timezone.utc)
            coordinator._process_live_setup_crosses(now)  # noqa: SLF001

            recovered = coordinator._active_setups[key]  # noqa: SLF001
            self.assertIsNone(recovered.detected_avwap)
            self.assertEqual(exchange.open_orders, [])

            exchange.price = 100.5
            coordinator._process_live_setup_crosses(now + timedelta(seconds=5))  # noqa: SLF001

            self.assertEqual(len(exchange.open_orders), 1)
            self.assertEqual(exchange.open_orders[0][3], 100.0)

    def test_restart_recovers_pending_metadata_and_claims_filled_position(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = self._persistent_config(
                tmpdir,
                entry_mode=EntryMode.CLOSE,
                exit_mode=ExitMode.CLOSE,
                exit_band=ExitBand.BAND_2,
            )
            first_exchange = _FakeExchange()
            first_exchange.price = 100.0
            first = EmaAvwapPullbackLiveCoordinator(
                exchange=first_exchange,
                config=config,
            )
            now = datetime(2026, 1, 1, tzinfo=timezone.utc)

            candidate = replace(
                _candidate(),
                entry_mode=EntryMode.CLOSE,
                exit_mode=ExitMode.CLOSE,
                exit_band=ExitBand.BAND_2,
            )
            self.assertTrue(first._queue_entry_candidate(candidate, now))  # noqa: SLF001
            self.assertEqual(len(first_exchange.open_orders), 1)
            client_id = first._state.pending_entries["ETHUSDT:LONG"].client_id  # noqa: SLF001
            self.assertIsNotNone(client_id)
            first.stop()

            second_exchange = _FakeExchange()
            second_exchange.price = 100.0
            second_exchange.positions = [
                Position(
                    symbol="ETHUSDT",
                    side=PositionSide.LONG,
                    size=1.0,
                    entry_price=99.0,
                    leverage=2,
                    margin_mode=MarginMode.ISOLATED,
                    unrealized_pnl=0.0,
                    position_id="pos-1",
                )
            ]
            restarted = EmaAvwapPullbackLiveCoordinator(
                exchange=second_exchange,
                config=config,
            )

            self.assertIn("ETHUSDT:LONG", restarted._state.pending_entries)  # noqa: SLF001
            self.assertIn("ETHUSDT:LONG", restarted._pending_meta_by_key)  # noqa: SLF001
            self.assertEqual(
                restarted._state.pending_entries["ETHUSDT:LONG"].client_id,  # noqa: SLF001
                client_id,
            )
            recovered_candidate = restarted._pending_meta_by_key[  # noqa: SLF001
                "ETHUSDT:LONG"
            ].candidate
            self.assertIs(recovered_candidate.entry_mode, EntryMode.CLOSE)
            self.assertIs(recovered_candidate.exit_mode, ExitMode.CLOSE)
            self.assertIs(recovered_candidate.exit_band, ExitBand.BAND_2)

            restarted._sync_positions(now + timedelta(minutes=1))  # noqa: SLF001

            self.assertIn("ETHUSDT", restarted._state.active_positions)  # noqa: SLF001
            self.assertNotIn("ETHUSDT:LONG", restarted._state.pending_entries)  # noqa: SLF001
            self.assertIn("ETHUSDT", restarted._position_runtime_by_symbol)  # noqa: SLF001
            self.assertEqual(
                restarted._state.active_positions["ETHUSDT"].entry_price, 99.0  # noqa: SLF001
            )
            self.assertEqual(restarted._state.active_positions["ETHUSDT"].stop_loss, 97.5)  # noqa: SLF001,E501
            self.assertEqual(
                restarted._position_runtime_by_symbol["ETHUSDT"].rigid_stop_level,  # noqa: SLF001
                97.5,
            )
            runtime = restarted._position_runtime_by_symbol["ETHUSDT"]  # noqa: SLF001
            self.assertIs(runtime.entry_mode, EntryMode.CLOSE)
            self.assertIs(runtime.exit_mode, ExitMode.CLOSE)
            self.assertIs(runtime.exit_band, ExitBand.BAND_2)
            self.assertEqual(second_exchange.stop_updates, [97.5])

    def test_restart_preserves_live_price_and_middle_observation_pair(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = self._persistent_config(tmpdir)
            first = EmaAvwapPullbackLiveCoordinator(
                exchange=_FakeExchange(),
                config=config,
            )
            setup = _SetupState(
                symbol="ETHUSDT",
                direction="long",
                anchor_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                detected_time=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
                consecutive_count=1,
                detected_avwap=_short_avwap(),
                is_waiting_for_cross=True,
            )
            key = first._setup_key("ETHUSDT", "long")  # noqa: SLF001
            first._active_setups[key] = setup  # noqa: SLF001
            first._last_price_by_setup_key[key] = 101.0  # noqa: SLF001
            first._last_middle_by_setup_key[key] = 100.0  # noqa: SLF001
            first._save_state()  # noqa: SLF001
            first.stop()

            restarted = EmaAvwapPullbackLiveCoordinator(
                exchange=_FakeExchange(),
                config=config,
            )

            self.assertEqual(restarted._last_price_by_setup_key[key], 101.0)  # noqa: SLF001
            self.assertEqual(restarted._last_middle_by_setup_key[key], 100.0)  # noqa: SLF001

    def test_on_tick_schedules_dynamic_trailing_stop(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(tmpdir),
            )

            trailing_checks: list[datetime] = []

            coordinator._manage_tick_trailing = trailing_checks.append  # type: ignore[method-assign]  # noqa: SLF001,E501

            coordinator._on_tick(  # noqa: SLF001
                datetime(2026, 1, 1, tzinfo=timezone.utc)
            )

            self.assertEqual(
                trailing_checks, [datetime(2026, 1, 1, tzinfo=timezone.utc)]
            )

    def test_sync_claims_untracked_exchange_position_for_configured_symbol(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            exchange.price = 100.0
            exchange.positions = [
                Position(
                    symbol="ETHUSDT",
                    side=PositionSide.SHORT,
                    size=2.0,
                    entry_price=101.0,
                    leverage=3,
                    margin_mode=MarginMode.ISOLATED,
                    unrealized_pnl=0.0,
                    position_id="orphan-1",
                )
            ]
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(tmpdir),
            )

            coordinator._sync_positions(datetime(2026, 1, 1, tzinfo=timezone.utc))  # noqa: SLF001,E501

            record = coordinator._state.active_positions["ETHUSDT"]  # noqa: SLF001
            self.assertEqual(record.position_id, "orphan-1")
            self.assertEqual(record.side, PositionSide.SHORT)
            self.assertNotIn("ETHUSDT", coordinator._position_runtime_by_symbol)  # noqa: SLF001

    def test_risk_distance_sizing_is_rejected(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "notional budget"):
                self._persistent_config(tmpdir, position_sizing_mode="risk_distance")

    def test_notional_budget_is_capped_by_percentage_and_absolute_limits(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(
                    tmpdir,
                    position_notional_pct=25.0,
                    max_position_size_pct=10.0,
                    max_entry_notional_usdt=5_000.0,
                ),
            )

            self.assertEqual(
                coordinator._compute_position_notional_budget("ETHUSDT"),  # noqa: SLF001
                1_000.0,
            )
            coordinator.stop()

            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(
                    tmpdir,
                    position_notional_pct=25.0,
                    max_position_size_pct=10.0,
                    max_entry_notional_usdt=500.0,
                ),
            )
            self.assertEqual(
                coordinator._compute_position_notional_budget("ETHUSDT"),  # noqa: SLF001
                500.0,
            )

    def test_invalid_minimum_quantity_is_dropped_before_account_changes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            exchange.quantity_validation_error = RuntimeError(
                "Quantity 1.0 below minTradeVolume 15.0 for ADAUSDT"
            )
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(tmpdir),
            )
            now = datetime(2026, 1, 1, tzinfo=timezone.utc)

            self.assertTrue(coordinator._queue_entry_candidate(_candidate(), now))  # noqa: SLF001

            self.assertEqual(exchange.validated_quantities, [("ETHUSDT", 1.0)])
            self.assertEqual(exchange.margin_mode_updates, [])
            self.assertEqual(exchange.leverage_updates, [])
            self.assertEqual(exchange.open_limit_attempts, 0)
            self.assertNotIn("ETHUSDT:LONG", coordinator._state.pending_entries)  # noqa: SLF001

    def test_pending_entries_are_independent_per_symbol(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(
                    tmpdir,
                    symbols=("ETHUSDT", "BTCUSDT"),
                    max_concurrent_positions=1,
                ),
            )
            now = datetime(2026, 1, 1, tzinfo=timezone.utc)

            self.assertTrue(coordinator._queue_entry_candidate(_candidate(), now))  # noqa: SLF001
            self.assertTrue(
                coordinator._queue_entry_candidate(  # noqa: SLF001
                    replace(_candidate(), symbol="BTCUSDT"), now
                )
            )
            self.assertEqual(len(exchange.open_orders), 2)

    def test_unconfigured_exchange_position_does_not_block_another_symbol(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            exchange.positions = [
                Position(
                    symbol="BTCUSDT",
                    side=PositionSide.LONG,
                    size=1.0,
                    entry_price=100.0,
                    leverage=2.0,
                    margin_mode=MarginMode.ISOLATED,
                    unrealized_pnl=0.0,
                    position_id="manual-btc-position",
                )
            ]
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(tmpdir, max_concurrent_positions=1),
            )

            coordinator._sync_positions(datetime(2026, 1, 1, tzinfo=timezone.utc))  # noqa: SLF001,E501

            self.assertTrue(
                coordinator._queue_entry_candidate(  # noqa: SLF001
                    _candidate(), datetime(2026, 1, 1, tzinfo=timezone.utc)
                )
            )
            self.assertIn("BTCUSDT", coordinator._state.active_positions)  # noqa: SLF001
            self.assertIn("ETHUSDT:LONG", coordinator._state.pending_entries)  # noqa: SLF001

    def test_pending_entry_is_cancelled_on_exact_timeout_bar(self) -> None:
        with TemporaryDirectory() as tmpdir:
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=_FakeExchange(),
                config=self._persistent_config(tmpdir, entry_cancel_bars=1),
            )
            signal_time = datetime(2026, 1, 1, 1, tzinfo=timezone.utc)
            pending = PendingEntryRecord(
                order_key="ETHUSDT:LONG",
                symbol="ETHUSDT",
                side=PositionSide.LONG,
                entry_price=100.0,
                quantity=1.0,
                leverage=2,
                margin_mode=MarginMode.ISOLATED,
                risk_amount=5.0,
                stop_for_risk=97.5,
                created_time=signal_time,
                signal_time=signal_time,
                activate_time=signal_time,
                client_id="emaavwap-timeout-id",
            )
            coordinator._state.pending_entries[pending.order_key] = pending  # noqa: SLF001

            coordinator._cancel_stale_entries(_snapshot(), signal_time)  # noqa: SLF001

            self.assertNotIn("ETHUSDT:LONG", coordinator._state.pending_entries)  # noqa: SLF001

    def test_unhealthy_position_sync_blocks_new_entries(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            exchange.positions_exc = RuntimeError("malformed position response")
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(tmpdir),
            )

            coordinator._sync_positions(datetime(2026, 1, 1, tzinfo=timezone.utc))  # noqa: SLF001,E501

            self.assertFalse(
                coordinator._queue_entry_candidate(  # noqa: SLF001
                    _candidate(), datetime(2026, 1, 1, tzinfo=timezone.utc)
                )
            )
            self.assertEqual(exchange.open_orders, [])

    def test_state_is_mainnet_tagged_and_exclusively_locked(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config = self._persistent_config(
                tmpdir, account_lock_file=base / "account.lock"
            )
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=_FakeExchange(), config=config
            )
            coordinator._save_state()  # noqa: SLF001
            data = json.loads(config.state_file.read_text(encoding="utf-8"))
            self.assertEqual(data["environment"], "mainnet")
            self.assertEqual(data["schema_version"], 3)

            with self.assertRaisesRegex(RuntimeError, "already held"):
                EmaAvwapPullbackLiveCoordinator(
                    exchange=_FakeExchange(),
                    config=replace(
                        config,
                        state_file=base / "other-state.json",
                        positions_db=base / "other-positions.db",
                    ),
                )
            coordinator.stop()

    def test_legacy_unscoped_state_is_refused(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = self._persistent_config(tmpdir)
            config.state_file.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "state load failed"):
                EmaAvwapPullbackLiveCoordinator(exchange=_FakeExchange(), config=config)

    def test_less_protective_stop_update_can_be_suppressed(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(
                    tmpdir,
                    allow_dynamic_stop_widening=False,
                ),
            )
            record = PositionRecord(
                position_id="pos-1",
                symbol="ETHUSDT",
                side=PositionSide.LONG,
                entry_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                entry_price=100.0,
                quantity=1.0,
                leverage=2,
                margin_mode=MarginMode.ISOLATED,
                take_profit=None,
                stop_loss=95.0,
                strategy="ema_avwap_pullback",
            )
            exchange_position = Position(
                symbol="ETHUSDT",
                side=PositionSide.LONG,
                size=1.0,
                entry_price=100.0,
                leverage=2,
                margin_mode=MarginMode.ISOLATED,
                unrealized_pnl=0.0,
                position_id="pos-1",
            )

            ok = coordinator._update_protective_stop(  # noqa: SLF001
                record,
                exchange_position,
                94.0,
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                reason="test",
                allow_widen=False,
            )

            self.assertTrue(ok)
            self.assertEqual(exchange.stop_updates, [])
            self.assertEqual(record.stop_loss, 95.0)

    def test_close_entry_enters_short_when_pullback_closes_above_middle(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            exchange.price = 101.25
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(
                    tmpdir,
                    entry_mode=EntryMode.CLOSE,
                    max_entry_deviation_pct=2.0,
                ),
            )
            first = _candle(offset=0, open=99.0, high=101.0, low=98.0, close=100.0)
            closed = _candle(offset=1, open=100.0, high=102.0, low=99.0, close=101.0)
            tpv_prefix, vol_prefix, tpv2_prefix = (
                coordinator._build_avwap_prefixes((first, closed))  # noqa: SLF001
            )
            snapshot = _SymbolSnapshot(
                symbol="ETHUSDT",
                timeframe="1h",
                timeframe_minutes=60,
                candles=(first, closed),
                candle_index=1,
                candle=closed,
                previous_candle=first,
                ema_value=102.0,
                tpv_prefix=tpv_prefix,
                vol_prefix=vol_prefix,
                tpv2_prefix=tpv2_prefix,
            )
            setup = _SetupState(
                symbol="ETHUSDT",
                direction="short",
                anchor_time=first.open_time,
                detected_time=first.close_time,
                consecutive_count=1,
                is_waiting_for_cross=True,
            )
            coordinator._active_setups[("ETHUSDT", "short")] = setup  # noqa: SLF001

            def fail_live_rebuild(*args):
                raise AssertionError("closed-candle mode must not fetch a forming bar")

            coordinator._build_live_avwap_snapshot = fail_live_rebuild  # type: ignore[method-assign]  # noqa: SLF001,E501

            now = datetime(2026, 1, 1, 2, tzinfo=timezone.utc)
            coordinator._process_live_setup_crosses(now)  # noqa: SLF001
            self.assertEqual(exchange.open_orders, [])

            queued = coordinator._process_pending_setup(  # noqa: SLF001
                "short", snapshot, now
            )

            self.assertTrue(queued)
            self.assertEqual(len(exchange.open_orders), 1)
            symbol, side, _qty, price, stop_loss = exchange.open_orders[0]
            self.assertEqual((symbol, side), ("ETHUSDT", PositionSide.SHORT))
            self.assertEqual(price, 101.25)
            self.assertAlmostEqual(stop_loss or 0.0, 104.2875)

    def test_live_exit_uses_selected_target_band_and_keeps_rigid_stop(self) -> None:
        for exit_band in ExitBand:
            with self.subTest(exit_band=exit_band.value), TemporaryDirectory() as tmpdir:
                exchange = _FakeExchange()
                coordinator = EmaAvwapPullbackLiveCoordinator(
                    exchange=exchange,
                    config=self._persistent_config(
                        tmpdir,
                        exit_mode=ExitMode.LIVE,
                        exit_band=exit_band,
                    ),
                )
                record = PositionRecord(
                    position_id="pos-1",
                    symbol="ETHUSDT",
                    side=PositionSide.LONG,
                    entry_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    entry_price=100.0,
                    quantity=1.0,
                    leverage=2,
                    margin_mode=MarginMode.ISOLATED,
                    take_profit=None,
                    stop_loss=97.0,
                    strategy="ema_avwap_pullback",
                )
                runtime = _PositionRuntime(
                    direction="long",
                    anchor_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    setup_detected_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    entry_signal_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    raw_entry_price=100.0,
                    dynamic_stop_at_entry=97.5,
                    rigid_stop_level=97.0,
                    trailing_activation_at_entry=102.5,
                    entry_trigger_mode="live_tick",
                    exit_mode=ExitMode.LIVE,
                    exit_band=exit_band,
                )
                coordinator._state.active_positions["ETHUSDT"] = record  # noqa: SLF001
                coordinator._position_runtime_by_symbol["ETHUSDT"] = runtime  # noqa: SLF001
                coordinator._last_snapshot_by_symbol["ETHUSDT"] = _snapshot()  # noqa: SLF001
                coordinator._build_live_avwap_snapshot = lambda *args: (  # type: ignore[method-assign]  # noqa: SLF001,E501
                    _snapshot(),
                    _short_avwap(),
                )
                exchange.price = 102.5

                coordinator._manage_live_position_exits(  # noqa: SLF001
                    datetime(2026, 1, 1, 2, tzinfo=timezone.utc)
                )

                if exit_band is ExitBand.BAND_1:
                    self.assertEqual(
                        exchange.close_calls, [("ETHUSDT", PositionSide.LONG)]
                    )
                    self.assertEqual(record.stop_loss, 97.0)
                    self.assertEqual(exchange.stop_updates, [])
                    continue

                self.assertEqual(exchange.close_calls, [])
                self.assertEqual(record.stop_loss, 97.0)
                self.assertEqual(exchange.stop_updates, [])
                exchange.price = 105.0
                coordinator._manage_live_position_exits(  # noqa: SLF001
                    datetime(2026, 1, 1, 2, 0, 5, tzinfo=timezone.utc)
                )
                self.assertEqual(
                    exchange.close_calls, [("ETHUSDT", PositionSide.LONG)]
                )

    def test_close_exit_uses_newly_closed_price_not_wick_or_live_price(self) -> None:
        for exit_band, target in (
            (ExitBand.BAND_1, 102.5),
            (ExitBand.BAND_2, 105.0),
        ):
            with self.subTest(exit_band=exit_band.value), TemporaryDirectory() as tmpdir:
                exchange = _FakeExchange()
                exchange.price = target + 10.0
                coordinator = EmaAvwapPullbackLiveCoordinator(
                    exchange=exchange,
                    config=self._persistent_config(
                        tmpdir,
                        exit_mode=ExitMode.CLOSE,
                        exit_band=exit_band,
                    ),
                )
                record = PositionRecord(
                    position_id="pos-1",
                    symbol="ETHUSDT",
                    side=PositionSide.LONG,
                    entry_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    entry_price=100.0,
                    quantity=1.0,
                    leverage=2,
                    margin_mode=MarginMode.ISOLATED,
                    take_profit=None,
                    stop_loss=97.0,
                    strategy="ema_avwap_pullback",
                )
                runtime = _PositionRuntime(
                    direction="long",
                    anchor_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    setup_detected_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    entry_signal_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    raw_entry_price=100.0,
                    dynamic_stop_at_entry=97.5,
                    rigid_stop_level=97.0,
                    trailing_activation_at_entry=102.5,
                    entry_trigger_mode="live_tick",
                    exit_mode=ExitMode.CLOSE,
                    exit_band=exit_band,
                )
                coordinator._state.active_positions["ETHUSDT"] = record  # noqa: SLF001
                coordinator._position_runtime_by_symbol["ETHUSDT"] = runtime  # noqa: SLF001
                coordinator._build_avwap_snapshot = lambda **_: _short_avwap()  # type: ignore[method-assign]  # noqa: SLF001,E501

                snapshot = _snapshot()
                wick_only = replace(
                    snapshot.candle,
                    high=target + 1.0,
                    close=target - 0.1,
                )
                snapshot = replace(
                    snapshot,
                    candles=(snapshot.candles[0], wick_only),
                    candle=wick_only,
                )
                coordinator._manage_position_on_bar(  # noqa: SLF001
                    snapshot,
                    wick_only.close_time,
                )
                self.assertEqual(exchange.close_calls, [])

                def fail_live_rebuild(*args):
                    raise AssertionError("close exit mode must not use forming AVWAP")

                coordinator._build_live_avwap_snapshot = fail_live_rebuild  # type: ignore[method-assign]  # noqa: SLF001,E501
                coordinator._manage_live_position_exits(  # noqa: SLF001
                    wick_only.close_time + timedelta(seconds=1)
                )
                self.assertEqual(exchange.close_calls, [])

                closed_at_target = replace(wick_only, close=target)
                snapshot = replace(
                    snapshot,
                    candles=(snapshot.candles[0], closed_at_target),
                    candle=closed_at_target,
                )
                coordinator._manage_position_on_bar(  # noqa: SLF001
                    snapshot,
                    closed_at_target.close_time,
                )
                self.assertEqual(
                    exchange.close_calls,
                    [("ETHUSDT", PositionSide.LONG)],
                )

    def test_close_exit_short_uses_lower_selected_band(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(
                    tmpdir,
                    exit_mode=ExitMode.CLOSE,
                    exit_band=ExitBand.BAND_2,
                ),
            )
            record = PositionRecord(
                position_id="pos-1",
                symbol="ETHUSDT",
                side=PositionSide.SHORT,
                entry_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                entry_price=100.0,
                quantity=1.0,
                leverage=2,
                margin_mode=MarginMode.ISOLATED,
                take_profit=None,
                stop_loss=103.0,
                strategy="ema_avwap_pullback",
            )
            runtime = _PositionRuntime(
                direction="short",
                anchor_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                setup_detected_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                entry_signal_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                raw_entry_price=100.0,
                dynamic_stop_at_entry=102.5,
                rigid_stop_level=103.0,
                trailing_activation_at_entry=97.5,
                entry_trigger_mode="live_tick",
                exit_mode=ExitMode.CLOSE,
                exit_band=ExitBand.BAND_2,
            )
            coordinator._state.active_positions["ETHUSDT"] = record  # noqa: SLF001
            coordinator._position_runtime_by_symbol["ETHUSDT"] = runtime  # noqa: SLF001
            coordinator._build_avwap_snapshot = lambda **_: _short_avwap()  # type: ignore[method-assign]  # noqa: SLF001,E501
            snapshot = _snapshot()
            closed_at_lower_band = replace(snapshot.candle, low=94.0, close=95.0)
            snapshot = replace(
                snapshot,
                candles=(snapshot.candles[0], closed_at_lower_band),
                candle=closed_at_lower_band,
            )

            coordinator._manage_position_on_bar(  # noqa: SLF001
                snapshot,
                closed_at_lower_band.close_time,
            )

            self.assertEqual(
                exchange.close_calls,
                [("ETHUSDT", PositionSide.SHORT)],
            )

    def test_close_entry_enters_long_when_pullback_closes_below_middle(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            exchange.price = 99.0
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(
                    tmpdir,
                    entry_mode=EntryMode.CLOSE,
                    exit_band=ExitBand.BAND_2,
                ),
            )
            snapshot = _snapshot()
            below_middle = replace(snapshot.candle, open=100.5, close=99.0)
            candles = (snapshot.candles[0], below_middle)
            tpv_prefix, vol_prefix, tpv2_prefix = (
                coordinator._build_avwap_prefixes(candles)  # noqa: SLF001
            )
            snapshot = replace(
                snapshot,
                candles=candles,
                candle=below_middle,
                tpv_prefix=tpv_prefix,
                vol_prefix=vol_prefix,
                tpv2_prefix=tpv2_prefix,
            )
            setup = _SetupState(
                symbol="ETHUSDT",
                direction="long",
                anchor_time=snapshot.candles[0].open_time,
                detected_time=snapshot.candles[0].close_time,
                consecutive_count=1,
            )
            coordinator._active_setups[("ETHUSDT", "long")] = setup  # noqa: SLF001

            def fail_live_rebuild(*args):
                raise AssertionError("closed-candle mode must not fetch a forming bar")

            coordinator._build_live_avwap_snapshot = fail_live_rebuild  # type: ignore[method-assign]  # noqa: SLF001,E501

            queued = coordinator._process_pending_setup(  # noqa: SLF001
                "long",
                snapshot,
                datetime(2026, 1, 1, 2, tzinfo=timezone.utc),
            )

            self.assertTrue(queued)
            self.assertEqual(exchange.open_orders[0][1], PositionSide.LONG)
            self.assertEqual(exchange.open_orders[0][4], 96.03)

    def test_close_entry_uses_newly_closed_sol_candle_without_bar_delay(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            exchange.price = 76.79
            telegram = _FakeTelegram()
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(
                    tmpdir,
                    entry_mode=EntryMode.CLOSE,
                    exit_band=ExitBand.BAND_2,
                ),
                telegram_client=telegram,  # type: ignore[arg-type]
            )
            candles = (
                _candle(
                    offset=0,
                    open=77.16,
                    high=77.24,
                    low=77.06,
                    close=77.19,
                    volume=41_275.13,
                ),
                _candle(
                    offset=1,
                    open=77.19,
                    high=77.35,
                    low=77.12,
                    close=77.21,
                    volume=46_946.98,
                ),
                _candle(
                    offset=2,
                    open=77.21,
                    high=77.28,
                    low=77.04,
                    close=77.24,
                    volume=33_358.07,
                ),
                _candle(
                    offset=3,
                    open=77.24,
                    high=77.49,
                    low=77.13,
                    close=77.33,
                    volume=59_013.58,
                ),
                _candle(
                    offset=4,
                    open=77.33,
                    high=77.82,
                    low=76.64,
                    close=76.80,
                    volume=196_825.94,
                ),
            )
            tpv_prefix, vol_prefix, tpv2_prefix = (
                coordinator._build_avwap_prefixes(candles)  # noqa: SLF001
            )
            snapshot = _SymbolSnapshot(
                symbol="ETHUSDT",
                timeframe="1h",
                timeframe_minutes=60,
                candles=candles,
                candle_index=4,
                candle=candles[4],
                previous_candle=candles[3],
                ema_value=75.8,
                tpv_prefix=tpv_prefix,
                vol_prefix=vol_prefix,
                tpv2_prefix=tpv2_prefix,
            )
            setup = _SetupState(
                symbol="ETHUSDT",
                direction="long",
                anchor_time=candles[0].open_time,
                detected_time=candles[3].close_time,
                consecutive_count=4,
            )
            coordinator._active_setups[("ETHUSDT", "long")] = setup  # noqa: SLF001

            def fail_live_rebuild(*args):
                raise AssertionError("closed-candle mode must not fetch a forming bar")

            coordinator._build_live_avwap_snapshot = fail_live_rebuild  # type: ignore[method-assign]  # noqa: SLF001,E501

            coordinator._fetch_latest_closed_candle = lambda symbol: candles[4]  # type: ignore[method-assign]  # noqa: SLF001,E501
            coordinator._build_snapshot = lambda symbol: snapshot  # type: ignore[method-assign]  # noqa: SLF001,E501

            now = datetime(2026, 1, 1, 5, tzinfo=timezone.utc)
            self._drain_market_data_until(
                coordinator,
                now,
                lambda: len(exchange.open_orders) == 1,
            )

            self.assertTrue(candles[4].is_bearish())
            self.assertEqual(len(exchange.open_orders), 1)
            self.assertEqual(exchange.open_orders[0][1], PositionSide.LONG)
            self.assertEqual(exchange.open_orders[0][3], 76.79)
            pending_meta = next(iter(coordinator._pending_meta_by_key.values()))  # noqa: SLF001
            self.assertEqual(pending_meta.candidate.signal_time, candles[4].close_time)
            self.assertAlmostEqual(pending_meta.candidate.decision_price or 0.0, 76.80)
            self.assertAlmostEqual(pending_meta.candidate.avwap.vwap, 77.15726694)
            self.assertIn(
                "Reason: bearish candle closed below AVWAP middle",
                telegram.messages[0],
            )
            self.assertIn("Exit Band: 2", telegram.messages[0])
            coordinator.stop()

    def test_close_entry_rejects_bullish_candle_below_middle(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            exchange.price = 99.0
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(
                    tmpdir,
                    entry_mode=EntryMode.CLOSE,
                ),
            )
            snapshot = _snapshot()
            bullish = replace(snapshot.candle, open=98.0, close=99.0)
            snapshot = replace(snapshot, candle=bullish)
            setup = _SetupState(
                symbol="ETHUSDT",
                direction="long",
                anchor_time=snapshot.candles[0].open_time,
                detected_time=snapshot.candles[0].close_time,
                consecutive_count=1,
            )
            coordinator._active_setups[("ETHUSDT", "long")] = setup  # noqa: SLF001
            coordinator._build_avwap_snapshot = lambda **kwargs: _short_avwap()  # type: ignore[method-assign]  # noqa: SLF001,E501

            queued = coordinator._process_pending_setup(  # noqa: SLF001
                "long",
                snapshot,
                datetime(2026, 1, 1, 2, tzinfo=timezone.utc),
            )

            self.assertFalse(queued)
            self.assertEqual(exchange.open_orders, [])

    def test_tick_trailing_is_independent_of_exit_mode_and_band(self) -> None:
        for exit_mode in ExitMode:
            with self.subTest(exit_mode=exit_mode.value), TemporaryDirectory() as tmpdir:
                exchange = _FakeExchange()
                exchange_position = Position(
                    symbol="ETHUSDT",
                    side=PositionSide.LONG,
                    size=1.0,
                    entry_price=100.0,
                    leverage=2,
                    margin_mode=MarginMode.ISOLATED,
                    unrealized_pnl=0.0,
                    position_id="pos-1",
                )
                exchange.positions = [exchange_position]
                coordinator = EmaAvwapPullbackLiveCoordinator(
                    exchange=exchange,
                    config=self._persistent_config(tmpdir, exit_mode=exit_mode),
                )
                record = PositionRecord(
                    position_id="pos-1",
                    symbol="ETHUSDT",
                    side=PositionSide.LONG,
                    entry_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    entry_price=100.0,
                    quantity=1.0,
                    leverage=2,
                    margin_mode=MarginMode.ISOLATED,
                    take_profit=None,
                    stop_loss=97.0,
                    strategy="ema_avwap_pullback",
                )
                runtime = _PositionRuntime(
                    direction="long",
                    anchor_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    setup_detected_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    entry_signal_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    raw_entry_price=100.0,
                    dynamic_stop_at_entry=97.5,
                    rigid_stop_level=97.0,
                    trailing_activation_at_entry=102.5,
                    entry_trigger_mode="live_tick",
                    last_avwap=_short_avwap(),
                    exit_mode=exit_mode,
                )
                coordinator._state.active_positions["ETHUSDT"] = record  # noqa: SLF001
                coordinator._position_runtime_by_symbol["ETHUSDT"] = runtime  # noqa: SLF001

                exchange.price = 103.0
                coordinator._manage_tick_trailing(  # noqa: SLF001
                    datetime(2026, 1, 1, 2, tzinfo=timezone.utc)
                )
                self.assertTrue(runtime.trailing_active)
                self.assertAlmostEqual(runtime.trailing_stop or 0.0, 101.97)
                self.assertEqual(exchange.stop_updates, [runtime.trailing_stop])
                self.assertEqual(exchange.close_calls, [])

                exchange.price = 104.0
                coordinator._manage_tick_trailing(  # noqa: SLF001
                    datetime(2026, 1, 1, 2, 0, 5, tzinfo=timezone.utc)
                )
                self.assertAlmostEqual(runtime.trailing_stop or 0.0, 102.96)
                self.assertEqual(len(exchange.stop_updates), 2)

                exchange.price = 102.0
                coordinator._manage_tick_trailing(  # noqa: SLF001
                    datetime(2026, 1, 1, 2, 0, 10, tzinfo=timezone.utc)
                )
                self.assertEqual(
                    exchange.close_calls, [("ETHUSDT", PositionSide.LONG)]
                )

    def test_zero_gap_trailing_closes_on_its_activation_tick(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            exchange_position = Position(
                symbol="ETHUSDT",
                side=PositionSide.LONG,
                size=1.0,
                entry_price=100.0,
                leverage=2,
                margin_mode=MarginMode.ISOLATED,
                unrealized_pnl=0.0,
                position_id="pos-1",
            )
            exchange.positions = [exchange_position]
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(tmpdir, trailing_gap_pct=0.0),
            )
            record = PositionRecord(
                position_id="pos-1",
                symbol="ETHUSDT",
                side=PositionSide.LONG,
                entry_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                entry_price=100.0,
                quantity=1.0,
                leverage=2,
                margin_mode=MarginMode.ISOLATED,
                take_profit=None,
                stop_loss=97.0,
                strategy="ema_avwap_pullback",
            )
            runtime = _PositionRuntime(
                direction="long",
                anchor_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                setup_detected_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                entry_signal_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                raw_entry_price=100.0,
                dynamic_stop_at_entry=97.5,
                rigid_stop_level=97.0,
                trailing_activation_at_entry=102.5,
                entry_trigger_mode="live_tick",
                last_avwap=_short_avwap(),
            )
            coordinator._state.active_positions["ETHUSDT"] = record  # noqa: SLF001
            coordinator._position_runtime_by_symbol["ETHUSDT"] = runtime  # noqa: SLF001

            exchange.price = 103.0
            now = datetime(2026, 1, 1, 2, tzinfo=timezone.utc)
            coordinator._manage_tick_trailing(now)  # noqa: SLF001

            self.assertTrue(runtime.trailing_active)
            self.assertEqual(runtime.trailing_stop, 103.0)
            self.assertEqual(exchange.stop_updates, [])
            self.assertEqual(
                exchange.close_calls, [("ETHUSDT", PositionSide.LONG)]
            )

    def test_trade_notifications_include_mode_trigger_and_indicators(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            telegram = _FakeTelegram()
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(
                    tmpdir,
                    entry_mode=EntryMode.LIVE,
                    exit_mode=ExitMode.CLOSE,
                    exit_band=ExitBand.BAND_2,
                ),
                telegram_client=telegram,  # type: ignore[arg-type]
            )
            candidate = replace(
                _candidate(),
                entry_mode=EntryMode.LIVE,
                exit_mode=ExitMode.CLOSE,
                exit_band=ExitBand.BAND_2,
                ema_value=99.0,
                stop_for_risk=97.0,
                rigid_stop_at_entry=97.0,
            )

            coordinator._queue_entry_candidate(  # noqa: SLF001
                candidate, datetime(2026, 1, 1, tzinfo=timezone.utc)
            )

            message = telegram.messages[0]
            self.assertIn("[EMA AVWAP ENTRY SIGNAL]", message)
            self.assertIn("Entry Mode: live", message)
            self.assertIn("Exit Mode: close", message)
            self.assertIn("Trigger: intrabar", message)
            self.assertIn("EMA: 99", message)
            self.assertIn("AVWAP Middle: 100", message)
            self.assertIn("Exit Band: 2 @ 105", message)


if __name__ == "__main__":
    unittest.main()
