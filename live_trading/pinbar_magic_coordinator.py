"""Simple PinBar Magic v2 live coordinator (single-file).

This module intentionally avoids scanner/dispatcher plumbing and focuses on:
- default ETHUSDT config
- candle-close signal generation
- risk-based pending stop-entry management
- periodic tick-based trailing updates
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from candle_downloader.models import Candle
from signal_notifier import TelegramClient

from .exchange import Exchange, MarginMode, OrderResult, Position, PositionSide
from .models import (
    LiveTradingConfig,
    PendingEntryRecord,
    PinBarMagicSnapshot,
    PositionRecord,
    SymbolInfo,
    TradingSignal,
    TradingState,
)
from .pinbar_magic_strategy import PinBarMagicLiveStrategy


@dataclass(frozen=True)
class PinBarMagicCoordinatorV2Config:
    symbols: tuple[str, ...] = ("ETHUSDT",)
    timeframe: str = "1h"
    poll_interval_seconds: float = 5.0
    trailing_check_interval_seconds: float = 5.0
    leverage: int = 10
    margin_mode: MarginMode = MarginMode.ISOLATED
    max_concurrent_positions: int = 1
    equity_risk_pct: float = 3.0
    atr_multiple: float = 0.5
    trail_points: float = 1.0
    trail_offset: float = 1.0
    slow_sma_period: int = 50
    medium_ema_period: int = 18
    fast_ema_period: int = 6
    atr_period: int = 14
    entry_cancel_bars: int = 3
    entry_activation_mode: str = "next_bar"
    enable_friday_close: bool = True
    friday_close_hour_utc: int = 16
    enable_ema_cross_close: bool = True
    risk_equity_include_unrealized: bool = True
    risk_equity_mark_source: str = "close"
    use_stop_fill_open_gap: bool = True
    disable_symbol_hours: float = 0.0


class PinBarMagicCoordinatorV2:
    """Live coordinator for PinBar Magic v2."""

    def __init__(
        self,
        exchange: Exchange,
        config: PinBarMagicCoordinatorV2Config | None = None,
        telegram_client: Optional[TelegramClient] = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._exchange = exchange
        self._cfg = config or PinBarMagicCoordinatorV2Config()
        self._log = logger or logging.getLogger(self.__class__.__name__)
        self._telegram = telegram_client
        self._state = TradingState()
        self._running = False
        self._last_closed_candle_time_by_symbol: Dict[str, datetime] = {}
        self._last_tick_trailing_check_ts: float = 0.0

        self._strategy = PinBarMagicLiveStrategy(
            config=self._build_strategy_config(self._cfg),
            exchange=self._exchange,
            state=self._state,
            logger=self._log,
        )

    def _build_strategy_config(
        self, cfg: PinBarMagicCoordinatorV2Config
    ) -> LiveTradingConfig:
        return LiveTradingConfig(
            exchange_name="runtime",
            api_key="",
            api_secret="",
            strategy_name="pinbar_magic_v2",
            timeframe=cfg.timeframe,
            pinbar_symbols=cfg.symbols,
            leverage=cfg.leverage,
            equity_risk_pct=cfg.equity_risk_pct,
            atr_multiple=cfg.atr_multiple,
            trail_points=cfg.trail_points,
            trail_offset=cfg.trail_offset,
            slow_sma_period=cfg.slow_sma_period,
            medium_ema_period=cfg.medium_ema_period,
            fast_ema_period=cfg.fast_ema_period,
            atr_period=cfg.atr_period,
            entry_cancel_bars=cfg.entry_cancel_bars,
            entry_activation_mode=cfg.entry_activation_mode,  # type: ignore[arg-type]
            enable_friday_close=cfg.enable_friday_close,
            friday_close_hour_utc=cfg.friday_close_hour_utc,
            enable_ema_cross_close=cfg.enable_ema_cross_close,
            risk_equity_include_unrealized=cfg.risk_equity_include_unrealized,
            risk_equity_mark_source=cfg.risk_equity_mark_source,  # type: ignore[arg-type]
            use_stop_fill_open_gap=cfg.use_stop_fill_open_gap,
            max_concurrent_positions=cfg.max_concurrent_positions,
            disable_symbol_hours=cfg.disable_symbol_hours,
            margin_mode=cfg.margin_mode,
        )

    def run_forever(self) -> None:
        self._running = True
        self._log.info(
            "PinBarMagicCoordinatorV2 started (symbols=%s timeframe=%s)",
            ",".join(self._cfg.symbols),
            self._cfg.timeframe,
        )
        while self._running:
            now = datetime.now(timezone.utc)
            try:
                self._maybe_process_new_candle(now)
                self._on_tick(now)
            except Exception as exc:
                self._log.error(
                    "PinBarMagicCoordinatorV2 loop error: %s", exc, exc_info=True
                )
            time.sleep(max(self._cfg.poll_interval_seconds, 0.2))

    def stop(self) -> None:
        self._running = False
        self._strategy.close()

    def _maybe_process_new_candle(self, now: datetime) -> None:
        symbol_infos = [self._build_symbol_info(symbol) for symbol in self._cfg.symbols]
        if not symbol_infos:
            return

        latest_closed_by_symbol: Dict[str, Candle | None] = {}
        max_workers = min(8, max(1, len(symbol_infos)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._fetch_latest_closed_candle, symbol_info.symbol): symbol_info.symbol
                for symbol_info in symbol_infos
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    latest_closed_by_symbol[symbol] = future.result()
                except Exception as exc:
                    self._log.warning(
                        "PinBarMagicCoordinatorV2: latest candle fetch failed for %s: %s",
                        symbol,
                        exc,
                    )
                    latest_closed_by_symbol[symbol] = None

        any_new_candle = False
        for symbol_info in symbol_infos:
            latest_closed = latest_closed_by_symbol.get(symbol_info.symbol)
            if latest_closed is None:
                continue
            last_seen = self._last_closed_candle_time_by_symbol.get(symbol_info.symbol)
            if last_seen is None or latest_closed.close_time > last_seen:
                self._last_closed_candle_time_by_symbol[symbol_info.symbol] = (
                    latest_closed.close_time
                )
                any_new_candle = True

        if not any_new_candle:
            return

        signals, snapshots = self._strategy.generate_signals_for_symbols(
            symbol_infos, now
        )
        if not snapshots:
            return

        self._log.info("New candle processed for %s symbol(s)", len(snapshots))
        self._sync_positions(now)
        # 1) Bar-based trailing pass (matches backtest bar-order semantics).
        for snapshot in snapshots.values():
            self._apply_bar_trailing(snapshot, now)

        # 2) Fill previously queued stop entries.
        self._activate_due_entries(now)
        self._sync_positions(now)

        # 3) Bar-close exit rules.
        for snapshot in snapshots.values():
            self._apply_bar_close_rules(snapshot, now)

        # 4) Cancel stale pending entries.
        self._cancel_stale_entries(snapshots, now)

        # 5) Queue new signals from this candle.
        snapshot_by_symbol = {
            snapshot.symbol: snapshot for snapshot in snapshots.values()
        }
        for signal in signals:
            self._queue_signal(signal, snapshot_by_symbol.get(signal.symbol))
        if self._cfg.entry_activation_mode == "same_bar":
            self._activate_due_entries(now)
            self._sync_positions(now)

    def _on_tick(self, now: datetime) -> None:
        self._sync_positions(now)
        if (
            time.time() - self._last_tick_trailing_check_ts
        ) < self._cfg.trailing_check_interval_seconds:
            return
        self._last_tick_trailing_check_ts = time.time()
        self._manage_tick_trailing(now)

    def _build_symbol_info(self, symbol: str) -> SymbolInfo:
        price = 0.0
        try:
            fetch = getattr(self._exchange, "fetch_price", None)
            if callable(fetch):
                value = fetch(symbol)
                if value is not None:
                    price = float(value)
        except Exception:
            pass
        return SymbolInfo(
            symbol=symbol.upper(),
            current_price=price,
            price_change_pct=0.0,
            volume=0.0,
            quote_volume=0.0,
        )

    def _fetch_latest_closed_candle(self, symbol: str) -> Candle | None:
        rows: list = []
        try:
            rows = self._exchange.get_klines(
                symbol=symbol,
                interval=self._cfg.timeframe,
                limit=3,
            )
        except Exception as exc:
            self._log.warning(
                "PinBarMagicCoordinatorV2: exchange.get_klines failed for %s (%s): %s",
                symbol,
                self._cfg.timeframe,
                exc,
            )

        if not rows:
            # Compatibility fallback while some exchanges are still on legacy data path.
            rows = self._strategy._fetch_binance_klines(  # noqa: SLF001
                symbol=symbol,
                interval=self._cfg.timeframe,
                limit=3,
            )
        if len(rows) < 2:
            return None
        return Candle.from_binance(symbol, self._cfg.timeframe, rows[-2])

    def _queue_signal(
        self, signal: TradingSignal, snapshot: Optional[PinBarMagicSnapshot] = None
    ) -> bool:
        symbol = signal.symbol
        side = signal.side
        stop_for_risk = signal.stop_loss
        if stop_for_risk is None:
            return False

        current = self._state.active_positions.get(symbol)
        if current is not None and current.side == side:
            return False
        if (
            current is None
            and len(self._state.active_positions) >= self._cfg.max_concurrent_positions
        ):
            return False

        entry_price = float(signal.entry_price)
        stop_price = float(stop_for_risk)
        if side == PositionSide.LONG and stop_price >= entry_price:
            return False
        if side == PositionSide.SHORT and stop_price <= entry_price:
            return False

        risk_amount = self._compute_risk_amount(snapshot)
        qty = self._compute_quantity(
            entry_price, stop_price, signal.leverage, risk_amount
        )
        if qty <= 0:
            return False

        signal_time = self._ensure_aware(signal.timestamp)
        key = self._pending_key(symbol, side)
        existing = self._state.pending_entries.get(key)
        if existing is not None:
            self._cancel_pending_entry(existing, "Replaced by newer signal")

        self._state.pending_entries[key] = PendingEntryRecord(
            order_key=key,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=qty,
            leverage=signal.leverage,
            margin_mode=signal.margin_mode,
            risk_amount=risk_amount,
            stop_for_risk=stop_price,
            created_time=self._ensure_aware(datetime.now(timezone.utc)),
            signal_time=signal_time,
            activate_time=self._activate_time(signal_time),
            status="PENDING",
            notes=signal.reason,
        )
        self._state.last_pinbar_signal_times[key] = signal_time
        self._log.info(
            "Queued stop-entry %s %s @ %.6f qty=%.6f",
            side.value,
            symbol,
            entry_price,
            qty,
        )
        return True

    def _sync_positions(self, now: datetime) -> None:
        exchange_positions = self._exchange.get_current_positions()
        by_symbol = {p.symbol: p for p in exchange_positions}

        for symbol, ex_pos in by_symbol.items():
            if symbol in self._state.active_positions:
                continue
            pending = self._find_matching_pending(symbol, ex_pos.side)
            if pending is None:
                continue
            self._state.active_positions[symbol] = PositionRecord(
                position_id=ex_pos.position_id or pending.order_id or pending.order_key,
                symbol=symbol,
                side=ex_pos.side,
                entry_time=now,
                entry_price=ex_pos.entry_price or pending.entry_price,
                quantity=ex_pos.size if ex_pos.size > 0 else pending.quantity,
                leverage=pending.leverage,
                margin_mode=pending.margin_mode,
                take_profit=None,
                stop_loss=pending.stop_for_risk,
                risk_amount=pending.risk_amount,
                strategy="pinbar_magic_v2",
                status="OPEN",
                notes=f"Filled from pending {pending.order_key}",
            )
            self._notify_trade_opened(self._state.active_positions[symbol], pending)
            self._state.pending_entries.pop(pending.order_key, None)

        for symbol, pos in list(self._state.active_positions.items()):
            if symbol not in by_symbol:
                pos.status = "CLOSED"
                pos.exit_time = now
                self._state.active_positions.pop(symbol, None)
                self._notify_trade_closed(
                    pos,
                    reason="Position no longer present on exchange",
                    exit_price=None,
                )

    def _activate_due_entries(self, now: datetime) -> None:
        for pending in list(self._state.pending_entries.values()):
            if pending.status != "PENDING":
                continue
            if now < pending.activate_time:
                continue
            order = self._place_stop_entry(pending)
            if order is None:
                pending.status = "ERROR"
                continue
            pending.order_id = order.order_id
            pending.status = "PLACED"

    def _apply_bar_close_rules(
        self, snapshot: PinBarMagicSnapshot, now: datetime
    ) -> None:
        position = self._state.active_positions.get(snapshot.symbol)
        if position is None or position.strategy != "pinbar_magic_v2":
            return
        close_reason: str | None = None
        if snapshot.friday_close:
            close_reason = "Market close"
        elif self._cfg.enable_ema_cross_close and (
            snapshot.crossunder or snapshot.crossover
        ):
            close_reason = "EMA cross close"
        if close_reason is not None:
            self._close_position(position, now, close_reason)

    def _apply_bar_trailing(self, snapshot: PinBarMagicSnapshot, now: datetime) -> None:
        position = self._state.active_positions.get(snapshot.symbol)
        if position is None or position.strategy != "pinbar_magic_v2":
            return

        exchange_position = self._exchange.get_position(snapshot.symbol)
        if exchange_position is None:
            return
        entry_price = exchange_position.entry_price or position.entry_price
        if entry_price <= 0:
            return

        if abs(snapshot.bar.open - snapshot.bar.high) < abs(
            snapshot.bar.open - snapshot.bar.low
        ):
            nodes = ("high", "low", "close")
        else:
            nodes = ("low", "high", "close")

        start_price = snapshot.bar.open
        stop_updated = False
        triggered = False

        for node in nodes:
            end_price = getattr(snapshot.bar, node)
            seg_high = max(start_price, end_price)
            seg_low = min(start_price, end_price)

            if position.side == PositionSide.LONG:
                activation = entry_price + self._cfg.trail_points
                if not position.trailing_active and seg_high >= activation:
                    position.trailing_active = True
                    position.extreme_since_activation = seg_high
                    position.trailing_stop = seg_high - self._cfg.trail_offset
                if position.trailing_active:
                    if position.extreme_since_activation is None:
                        position.extreme_since_activation = seg_high
                    elif seg_high > position.extreme_since_activation:
                        position.extreme_since_activation = seg_high
                    desired = position.extreme_since_activation - self._cfg.trail_offset
                    if (
                        position.trailing_stop is None
                        or desired > position.trailing_stop
                    ):
                        position.trailing_stop = desired
                        stop_updated = True
                    if (
                        position.trailing_stop is not None
                        and seg_low <= position.trailing_stop
                    ):
                        triggered = True
                        break
            else:
                activation = entry_price - self._cfg.trail_points
                if not position.trailing_active and seg_low <= activation:
                    position.trailing_active = True
                    position.extreme_since_activation = seg_low
                    position.trailing_stop = seg_low + self._cfg.trail_offset
                if position.trailing_active:
                    if position.extreme_since_activation is None:
                        position.extreme_since_activation = seg_low
                    elif seg_low < position.extreme_since_activation:
                        position.extreme_since_activation = seg_low
                    desired = position.extreme_since_activation + self._cfg.trail_offset
                    if (
                        position.trailing_stop is None
                        or desired < position.trailing_stop
                    ):
                        position.trailing_stop = desired
                        stop_updated = True
                    if (
                        position.trailing_stop is not None
                        and seg_high >= position.trailing_stop
                    ):
                        triggered = True
                        break

            start_price = end_price

        if stop_updated and position.trailing_stop is not None:
            self._update_stop_loss_on_exchange(
                exchange_position, position.trailing_stop
            )
        if triggered:
            self._close_position(position, now, "Trailing stop")

    def _manage_tick_trailing(self, now: datetime) -> None:
        for symbol, position in list(self._state.active_positions.items()):
            if position.strategy != "pinbar_magic_v2":
                continue
            price = self._get_last_price(symbol)
            if price is None or price <= 0:
                continue
            entry = position.entry_price
            if entry <= 0:
                continue

            triggered = False
            if position.side == PositionSide.LONG:
                activation = entry + self._cfg.trail_points
                if not position.trailing_active and price >= activation:
                    position.trailing_active = True
                    position.extreme_since_activation = price
                    position.trailing_stop = price - self._cfg.trail_offset
                if position.trailing_active:
                    if (
                        position.extreme_since_activation is None
                        or price > position.extreme_since_activation
                    ):
                        position.extreme_since_activation = price
                    desired = position.extreme_since_activation - self._cfg.trail_offset
                    if (
                        position.trailing_stop is None
                        or desired > position.trailing_stop
                    ):
                        position.trailing_stop = desired
                    triggered = (
                        position.trailing_stop is not None
                        and price <= position.trailing_stop
                    )
            else:
                activation = entry - self._cfg.trail_points
                if not position.trailing_active and price <= activation:
                    position.trailing_active = True
                    position.extreme_since_activation = price
                    position.trailing_stop = price + self._cfg.trail_offset
                if position.trailing_active:
                    if (
                        position.extreme_since_activation is None
                        or price < position.extreme_since_activation
                    ):
                        position.extreme_since_activation = price
                    desired = position.extreme_since_activation + self._cfg.trail_offset
                    if (
                        position.trailing_stop is None
                        or desired < position.trailing_stop
                    ):
                        position.trailing_stop = desired
                    triggered = (
                        position.trailing_stop is not None
                        and price >= position.trailing_stop
                    )

            if position.trailing_stop is not None:
                ex_pos = self._exchange.get_position(symbol)
                if ex_pos is not None:
                    self._update_stop_loss_on_exchange(ex_pos, position.trailing_stop)
            if triggered:
                self._close_position(position, now, "Trailing stop")

    def _cancel_stale_entries(
        self, snapshots: Dict[str, PinBarMagicSnapshot], now: datetime
    ) -> None:
        tf_seconds = max(self._timeframe_seconds(), 1)
        for key, pending in list(self._state.pending_entries.items()):
            signal_time = self._state.last_pinbar_signal_times.get(
                key, pending.signal_time
            )
            symbol_snapshot = snapshots.get(pending.symbol)
            reference = (
                symbol_snapshot.bar.close_time if symbol_snapshot is not None else now
            )
            bars_since = int(
                max(0.0, (reference - signal_time).total_seconds()) // tf_seconds
            )
            if bars_since > self._cfg.entry_cancel_bars:
                self._cancel_pending_entry(
                    pending, f"entry timeout after {bars_since} bars"
                )

    def _compute_risk_amount(self, snapshot: Optional[PinBarMagicSnapshot]) -> float:
        realized_equity = max(float(self._exchange.get_account_balance()), 0.0)
        if not self._cfg.risk_equity_include_unrealized or snapshot is None:
            return (self._cfg.equity_risk_pct / 100.0) * realized_equity

        position = self._state.active_positions.get(snapshot.symbol)
        if position is None or position.strategy != "pinbar_magic_v2":
            return (self._cfg.equity_risk_pct / 100.0) * realized_equity

        mark_price = self._mark_price_for_equity(snapshot)
        if position.side == PositionSide.LONG:
            unrealized = (mark_price - position.entry_price) * position.quantity
        else:
            unrealized = (position.entry_price - mark_price) * position.quantity
        return (self._cfg.equity_risk_pct / 100.0) * max(
            realized_equity + unrealized, 0.0
        )

    def _mark_price_for_equity(self, snapshot: PinBarMagicSnapshot) -> float:
        source = self._cfg.risk_equity_mark_source
        if source == "open":
            return snapshot.bar.open
        if source == "hl2":
            return (snapshot.bar.high + snapshot.bar.low) / 2.0
        if source == "ohlc4":
            return (
                snapshot.bar.open
                + snapshot.bar.high
                + snapshot.bar.low
                + snapshot.bar.close
            ) / 4.0
        return snapshot.bar.close

    def _compute_quantity(
        self, entry_price: float, stop_price: float, leverage: int, risk_amount: float
    ) -> float:
        distance = abs(entry_price - stop_price)
        if distance <= 0 or risk_amount <= 0:
            return 0.0
        return (risk_amount * float(leverage)) / distance

    def _place_stop_entry(self, pending: PendingEntryRecord) -> Optional[OrderResult]:
        placer = getattr(self._exchange, "place_stop_entry_order", None)
        if callable(placer):
            return placer(
                symbol=pending.symbol,
                side=pending.side,
                quantity=pending.quantity,
                stop_price=pending.entry_price,
                leverage=pending.leverage,
                margin_mode=pending.margin_mode,
                stop_loss=pending.stop_for_risk,
            )
        fallback_price = pending.entry_price
        if self._cfg.use_stop_fill_open_gap:
            last_price = self._get_last_price(pending.symbol)
            if last_price is not None:
                if (
                    pending.side == PositionSide.LONG
                    and last_price >= pending.entry_price
                ):
                    fallback_price = last_price
                elif (
                    pending.side == PositionSide.SHORT
                    and last_price <= pending.entry_price
                ):
                    fallback_price = last_price
        return self._exchange.open_limit_position(
            symbol=pending.symbol,
            side=pending.side,
            quantity=pending.quantity,
            price=fallback_price,
            leverage=pending.leverage,
            margin_mode=pending.margin_mode,
            take_profit=None,
            stop_loss=pending.stop_for_risk,
        )

    def _close_position(
        self, position: PositionRecord, now: datetime, reason: str
    ) -> None:
        result = self._exchange.close_position(position.symbol, side=position.side)
        exit_price = (
            result.price if result.price > 0 else self._get_last_price(position.symbol)
        )
        position.status = "CLOSED"
        position.exit_time = now
        position.exit_price = exit_price
        if exit_price is not None:
            if position.side == PositionSide.LONG:
                position.pnl = (exit_price - position.entry_price) * position.quantity
            else:
                position.pnl = (position.entry_price - exit_price) * position.quantity
        position.notes = f"{position.notes}; {reason}".strip("; ")
        self._state.active_positions.pop(position.symbol, None)
        self._log.info("Closed %s due to %s", position.symbol, reason)
        self._notify_trade_closed(position, reason=reason, exit_price=exit_price)

    def _update_stop_loss_on_exchange(
        self, exchange_position: Position, stop_price: float
    ) -> bool:
        updater = getattr(self._exchange, "update_stop_loss", None)
        if callable(updater):
            try:
                return bool(updater(exchange_position, stop_price))
            except Exception:
                return False
        return True

    def _cancel_pending_entry(self, pending: PendingEntryRecord, reason: str) -> None:
        if pending.order_id:
            cancel = getattr(self._exchange, "cancel_order", None)
            cancelled = False
            if callable(cancel):
                try:
                    cancelled = bool(
                        cancel(symbol=pending.symbol, order_id=pending.order_id)
                    )
                except Exception:
                    cancelled = False
            if not cancelled:
                try:
                    self._exchange.cancel_all_orders(pending.symbol)
                except Exception:
                    pass
        pending.status = "CANCELLED"
        pending.notes = f"{pending.notes}; {reason}".strip("; ")
        self._state.pending_entries.pop(pending.order_key, None)

    def _find_matching_pending(
        self, symbol: str, side: PositionSide
    ) -> Optional[PendingEntryRecord]:
        key = self._pending_key(symbol, side)
        pending = self._state.pending_entries.get(key)
        if pending is not None:
            return pending
        same_symbol = [
            x for x in self._state.pending_entries.values() if x.symbol == symbol
        ]
        if len(same_symbol) == 1:
            return same_symbol[0]
        return None

    def _pending_key(self, symbol: str, side: PositionSide) -> str:
        return f"{symbol}:{side.value}"

    def _activate_time(self, signal_time: datetime) -> datetime:
        if self._cfg.entry_activation_mode == "same_bar":
            return signal_time
        return signal_time + timedelta(seconds=self._timeframe_seconds())

    def _timeframe_seconds(self) -> int:
        mapping = {
            "1m": 60,
            "3m": 180,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "2h": 7200,
            "4h": 14400,
            "6h": 21600,
            "12h": 43200,
            "1d": 86400,
        }
        return mapping.get(self._cfg.timeframe, 3600)

    def _get_last_price(self, symbol: str) -> Optional[float]:
        try:
            fetch = getattr(self._exchange, "fetch_price", None)
            if callable(fetch):
                val = fetch(symbol)
                if val is not None:
                    return float(val)
        except Exception:
            return None
        return None

    @staticmethod
    def _ensure_aware(moment: datetime) -> datetime:
        if moment.tzinfo is None:
            return moment.replace(tzinfo=timezone.utc)
        return moment

    def _notify_trade_opened(
        self, position: PositionRecord, pending: PendingEntryRecord
    ) -> None:
        if not self._telegram:
            return
        try:
            lines = [
                f"[PINBAR OPEN] {position.symbol}",
                f"Side: {position.side.value.upper()}",
                f"Entry: {position.entry_price:.8g}",
                f"Qty: {position.quantity:.8g}",
                f"Leverage: {position.leverage}x",
                f"Stop: {pending.stop_for_risk:.8g}",
                f"Mode: {position.margin_mode.value}",
                f"Time: {datetime.now(timezone.utc).isoformat()}",
            ]
            self._telegram.send_message("\n".join(lines))
        except Exception as exc:  # pragma: no cover - defensive logging
            self._log.warning(
                "Failed to send Telegram open notification for %s: %s",
                position.symbol,
                exc,
            )

    def _notify_trade_closed(
        self, position: PositionRecord, *, reason: str, exit_price: Optional[float]
    ) -> None:
        if not self._telegram:
            return
        try:
            pnl = position.pnl
            lines = [
                f"[PINBAR CLOSE] {position.symbol}",
                f"Side: {position.side.value.upper()}",
                f"Entry: {position.entry_price:.8g}",
                f"Exit: {exit_price:.8g}" if exit_price is not None else "Exit: n/a",
                f"Qty: {position.quantity:.8g}",
                f"PnL: {pnl:.8g}" if pnl is not None else "PnL: n/a",
                f"Reason: {reason}",
                f"Time: {datetime.now(timezone.utc).isoformat()}",
            ]
            self._telegram.send_message("\n".join(lines))
        except Exception as exc:  # pragma: no cover - defensive logging
            self._log.warning(
                "Failed to send Telegram close notification for %s: %s",
                position.symbol,
                exc,
            )
