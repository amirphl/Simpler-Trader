from __future__ import annotations

import unittest
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from cmd.live_trading import _shared as live_trading_shared
from candle_downloader.models import Candle, to_milliseconds
from live_trading.ema_avwap_pullback_strategy import (
    EmaAvwapPullbackLiveConfig,
    EmaAvwapPullbackLiveCoordinator,
    _AvwapSnapshot,
    _CrossDecision,
    _EntryCandidate,
    _PendingEntryMeta,
    _SetupState,
    _SizingDecision,
    _SymbolSnapshot,
)
from live_trading.exchange import (
    ExchangeConfig,
    MarginMode,
    OrderResult,
    OrderType,
    Position,
    PositionSide,
)
from live_trading.models import PendingEntryRecord, PositionRecord


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
        self.open_orders: list[tuple[str, PositionSide, float, float, float | None]] = []
        self.klines: list[list] = []

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
        return None

    def set_leverage(self, symbol: str, leverage: int) -> None:
        return None

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
    ) -> OrderResult:
        self.open_orders.append((symbol, side, quantity, price, stop_loss))
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

    def close_position(
        self, symbol: str, side: PositionSide | None = None
    ) -> OrderResult:
        self.close_calls.append((symbol, side))
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
        return self.klines[-limit:]

    def close(self) -> None:
        return None


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
        distance=5.0,
        entry_price=100.0,
        estimated_exit_price=100.0,
        risk_amount_interpretation="stop_loss_risk",
        base_qty_before_costs=1.0,
        qty_reduction_from_costs=0.0,
        sizing_reference_price=100.0,
        effective_price_for_sizing=5.0,
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
        stop_for_risk=95.0,
        dynamic_stop_at_entry=95.0,
        rigid_stop_at_entry=None,
        trailing_activation_at_entry=102.5,
        quantity=1.0,
        risk_amount=5.0,
        risk_amount_interpretation="stop_loss_risk",
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
        return EmaAvwapPullbackLiveConfig(
            symbols=("ETHUSDT",),
            max_entry_notional_usdt=1_000.0,
            state_file=base / "state.json",
            positions_db=base / "positions.db",
            **overrides,
        )

    def test_http_proxy_config_is_used_for_https_binance_fallback(self) -> None:
        with TemporaryDirectory() as tmpdir:
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=_FakeExchange(proxies={"http": "http://127.0.0.1:8080"}),
                config=self._persistent_config(tmpdir),
            )

            self.assertEqual(
                coordinator._binance_proxies,  # noqa: SLF001
                {
                    "http": "http://127.0.0.1:8080",
                    "https": "http://127.0.0.1:8080",
                },
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
                ema_value=90.0,
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

    def test_initial_stop_failure_force_closes_filled_position(self) -> None:
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
                stop_for_risk=95.0,
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

            self.assertEqual(exchange.close_calls, [("ETHUSDT", PositionSide.LONG)])
            self.assertNotIn("ETHUSDT", coordinator._state.active_positions)  # noqa: SLF001

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

            result = coordinator._replace_or_store_setup(new_setup, None)  # noqa: SLF001[arg-type]

            self.assertIs(result, old_setup)
            self.assertIs(coordinator._active_setups[key], old_setup)  # noqa: SLF001

    def test_live_cross_skips_setup_already_past_entry_on_first_tick(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            exchange.price = 101.0
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(tmpdir),
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

            coordinator._process_live_setup_crosses(  # noqa: SLF001
                datetime(2026, 1, 1, 2, tzinfo=timezone.utc)
            )

            self.assertNotIn(key, coordinator._active_setups)  # noqa: SLF001
            self.assertEqual(exchange.open_orders, [])

    def test_closed_bar_cross_discards_stale_setup_without_entry(self) -> None:
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
                ema_value=90.0,
                tpv_prefix=(0.0,),
                vol_prefix=(0.0,),
                tpv2_prefix=(0.0,),
            )
            coordinator._build_avwap_snapshot = lambda **_: _short_avwap()  # type: ignore[method-assign]  # noqa: SLF001,E501

            queued = coordinator._process_pending_setup(  # noqa: SLF001
                "short",
                snapshot,
                datetime(2026, 1, 1, 2, tzinfo=timezone.utc),
            )

            self.assertFalse(queued)
            self.assertNotIn(key, coordinator._active_setups)  # noqa: SLF001
            self.assertEqual(exchange.open_orders, [])

    def test_live_tick_cross_queues_entry_after_setup_baseline(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            exchange.price = 99.0
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=exchange,
                config=self._persistent_config(tmpdir),
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

            now = datetime(2026, 1, 1, 2, tzinfo=timezone.utc)
            coordinator._process_live_setup_crosses(now)  # noqa: SLF001

            self.assertEqual(exchange.open_orders, [])
            self.assertEqual(coordinator._last_price_by_setup_key[key], 99.0)  # noqa: SLF001

            exchange.price = 100.1
            coordinator._process_live_setup_crosses(now + timedelta(seconds=5))  # noqa: SLF001

            self.assertEqual(len(exchange.open_orders), 1)
            symbol, side, _quantity, price, stop_loss = exchange.open_orders[0]
            self.assertEqual(symbol, "ETHUSDT")
            self.assertEqual(side, PositionSide.SHORT)
            self.assertEqual(price, 100.0)
            self.assertEqual(stop_loss, 105.0)
            self.assertNotIn(key, coordinator._active_setups)  # noqa: SLF001

    def test_live_tick_cross_uses_frozen_setup_avwap(self) -> None:
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
            exchange.klines = [_binance_row(forming)]
            live_rebuild_calls = 0

            def fail_live_rebuild(*args):
                nonlocal live_rebuild_calls
                live_rebuild_calls += 1
                raise AssertionError("frozen setup AVWAP should be used")

            coordinator._build_live_avwap_snapshot = fail_live_rebuild  # type: ignore[method-assign]  # noqa: SLF001,E501

            now = datetime(2026, 1, 1, 3, tzinfo=timezone.utc)
            coordinator._process_live_setup_crosses(now)  # noqa: SLF001

            self.assertEqual(exchange.open_orders, [])
            self.assertEqual(coordinator._last_price_by_setup_key[key], 99.0)  # noqa: SLF001

            exchange.price = 100.5
            coordinator._process_live_setup_crosses(now + timedelta(seconds=5))  # noqa: SLF001

            self.assertEqual(len(exchange.open_orders), 1)
            self.assertEqual(exchange.open_orders[0][3], 100.0)
            self.assertEqual(live_rebuild_calls, 0)

    def test_legacy_waiting_setup_recovers_detected_avwap_for_live_cross(self) -> None:
        with TemporaryDirectory() as tmpdir:
            exchange = _FakeExchange()
            exchange.price = 97.0
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

            def fail_live_rebuild(*args):
                raise AssertionError("legacy setup should recover detected AVWAP")

            coordinator._build_live_avwap_snapshot = fail_live_rebuild  # type: ignore[method-assign]  # noqa: SLF001,E501

            now = datetime(2026, 1, 1, 3, tzinfo=timezone.utc)
            coordinator._process_live_setup_crosses(now)  # noqa: SLF001

            recovered = coordinator._active_setups[key]  # noqa: SLF001
            self.assertIsNotNone(recovered.detected_avwap)
            self.assertEqual(recovered.detected_avwap.vwap, 97.5)
            self.assertEqual(exchange.open_orders, [])

            exchange.price = 98.0
            coordinator._process_live_setup_crosses(now + timedelta(seconds=5))  # noqa: SLF001

            self.assertEqual(len(exchange.open_orders), 1)
            self.assertEqual(exchange.open_orders[0][3], 97.5)

    def test_restart_recovers_pending_metadata_and_claims_filled_position(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = self._persistent_config(tmpdir)
            first_exchange = _FakeExchange()
            first_exchange.price = 100.0
            first = EmaAvwapPullbackLiveCoordinator(
                exchange=first_exchange,
                config=config,
            )
            now = datetime(2026, 1, 1, tzinfo=timezone.utc)

            self.assertTrue(first._queue_entry_candidate(_candidate(), now))  # noqa: SLF001
            self.assertEqual(len(first_exchange.open_orders), 1)

            second_exchange = _FakeExchange()
            second_exchange.price = 100.0
            second_exchange.positions = [
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
            restarted = EmaAvwapPullbackLiveCoordinator(
                exchange=second_exchange,
                config=config,
            )

            self.assertIn("ETHUSDT:LONG", restarted._state.pending_entries)  # noqa: SLF001
            self.assertIn("ETHUSDT:LONG", restarted._pending_meta_by_key)  # noqa: SLF001

            restarted._sync_positions(now + timedelta(minutes=1))  # noqa: SLF001

            self.assertIn("ETHUSDT", restarted._state.active_positions)  # noqa: SLF001
            self.assertNotIn("ETHUSDT:LONG", restarted._state.pending_entries)  # noqa: SLF001
            self.assertIn("ETHUSDT", restarted._position_runtime_by_symbol)  # noqa: SLF001
            self.assertEqual(restarted._state.active_positions["ETHUSDT"].stop_loss, 95.0)  # noqa: SLF001,E501

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

    def test_invalid_stop_geometry_returns_no_sizing(self) -> None:
        with TemporaryDirectory() as tmpdir:
            coordinator = EmaAvwapPullbackLiveCoordinator(
                exchange=_FakeExchange(),
                config=self._persistent_config(tmpdir),
            )

            self.assertIsNone(
                coordinator._build_sizing_decision(  # noqa: SLF001
                    direction="long",
                    raw_entry_price=100.0,
                    stop_level=101.0,
                    risk_amount=10.0,
                )
            )
            self.assertIsNone(
                coordinator._build_sizing_decision(  # noqa: SLF001
                    direction="short",
                    raw_entry_price=100.0,
                    stop_level=99.0,
                    risk_amount=10.0,
                )
            )

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


if __name__ == "__main__":
    unittest.main()
