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


class _InsufficientBalanceError(RuntimeError):
    """Raised when exchange rejects an action due to insufficient balance/margin."""


class _EntryDeferredError(RuntimeError):
    """Raised when a stop-entry trigger has not been reached yet."""


@dataclass(frozen=True)
class PinBarMagicCoordinatorV2Config:
    symbols: tuple[str, ...] = ("ETHUSDT",)
    timeframe: str = "1h"
    trailing_tick_timeframe: str = "15m"
    use_trailing_tick_emulation: bool = False
    poll_interval_seconds: float = 5.0
    trailing_check_interval_seconds: float = 5.0
    leverage: int = 10
    margin_mode: MarginMode = MarginMode.ISOLATED
    max_concurrent_positions: int = 1
    max_entry_notional_usdt: float = 15.0
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
            trailing_tick_timeframe=cfg.trailing_tick_timeframe,
            use_trailing_tick_emulation=cfg.use_trailing_tick_emulation,
            enable_friday_close=cfg.enable_friday_close,
            friday_close_hour_utc=cfg.friday_close_hour_utc,
            enable_ema_cross_close=cfg.enable_ema_cross_close,
            risk_equity_include_unrealized=cfg.risk_equity_include_unrealized,
            risk_equity_mark_source=cfg.risk_equity_mark_source,  # type: ignore[arg-type]
            use_stop_fill_open_gap=cfg.use_stop_fill_open_gap,
            max_concurrent_positions=cfg.max_concurrent_positions,
            max_entry_notional_usdt=cfg.max_entry_notional_usdt,
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
                executor.submit(
                    self._fetch_latest_closed_candle, symbol_info.symbol
                ): symbol_info.symbol
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
                # TODO: Should we create fake candles based on last price if fetch fails? For now we just skip processing.
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
        price = None
        try:
            price = self._exchange.fetch_price(symbol)
        except Exception:
            pass
        return SymbolInfo(
            symbol=symbol.upper(),
            current_price=price if price else 0.0,
            price_change_pct=0.0,
            volume=0.0,
            quote_volume=0.0,
        )

    def _fetch_latest_closed_candle(self, symbol: str) -> Candle | None:
        """Return the most-recently CLOSED candle for the configured timeframe.

        Tries the exchange adapter first; falls back to the legacy Binance path.
        Returns None if fewer than 2 rows are available (can't identify closed bar).
        """
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
            try:
                rows = self._strategy._fetch_binance_klines(  # noqa: SLF001
                    symbol=symbol,
                    interval=self._cfg.timeframe,
                    limit=3,
                )
            except Exception as exc:
                self._log.warning(
                    "PinBarMagicCoordinatorV2: legacy klines fallback failed for %s (%s): %s",
                    symbol,
                    self._cfg.timeframe,
                    exc,
                )

        if len(rows) < 2:
            return None
        # rows[-1] is the still-open candle; rows[-2] is the last closed one.
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
        if entry_price <= 0:
            return False
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
        entry_notional = entry_price * qty
        max_notional = float(self._cfg.max_entry_notional_usdt)
        if entry_notional > max_notional:
            qty = max_notional / entry_price
            if qty <= 0:
                return False
            self._log.info(
                "Clamped entry notional for %s from %.6f to %.6f USDT "
                "(leverage=%sx, effective exposure=%.6f)",
                symbol,
                entry_notional,
                max_notional,
                signal.leverage,
                max_notional * float(signal.leverage),
            )

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
        """Reconcile local state with the exchange's live position list.

        - Any exchange position that has no local record but matches a pending entry
          is promoted to an active PositionRecord.
        - Any local active position that no longer appears on the exchange is marked
          CLOSED (e.g. hit SL/TP directly on the exchange while we weren't watching).
        """
        try:
            exchange_positions = self._exchange.get_current_positions()
        except Exception as exc:
            self._log.warning(
                "PinBarMagicCoordinatorV2: get_current_positions failed: %s", exc
            )
            return

        by_symbol = {p.symbol: p for p in exchange_positions}

        # Promote newly filled pending entries → active positions.
        for symbol, ex_pos in by_symbol.items():
            if symbol in self._state.active_positions:
                continue
            pending = self._find_matching_pending(symbol, ex_pos.side)
            if pending is None:
                continue
            if self._state.active_positions.get(symbol) is not None:
                self._log.warning(
                    "Position for %s already exists in state",
                    symbol,
                )
            # TODO: Is this correct? We may
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
            self._log.info(
                "Position opened for %s (%s) at entry price %.6f",
                symbol,
                ex_pos.side.value,
                ex_pos.entry_price or pending.entry_price,
            )
            self._notify_trade_opened(self._state.active_positions[symbol], pending)
            self._state.pending_entries.pop(pending.order_key, None)

        # Detect positions closed externally (SL/TP hit on exchange).
        for symbol, pos in list(self._state.active_positions.items()):
            if symbol not in by_symbol:
                pos.status = "CLOSED"
                pos.exit_time = now
                self._state.active_positions.pop(symbol, None)
                self._log.info(
                    "Position for %s no longer on exchange – marking closed", symbol
                )
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
            try:
                order = self._place_stop_entry(pending)
            except _InsufficientBalanceError as exc:
                pending.status = "PENDING"
                pending.notes = (
                    f"{pending.notes}; insufficient balance: {exc}"
                ).strip("; ")
                self._log.warning(
                    "Insufficient balance for %s %s entry; keeping pending for retry: %s",
                    pending.symbol,
                    pending.side.value,
                    exc,
                )
                continue
            except _EntryDeferredError as exc:
                pending.status = "PENDING"
                pending.notes = f"{pending.notes}; {exc}".strip("; ")
                self._log.info(
                    "Stop-entry trigger not reached for %s %s; keeping pending",
                    pending.symbol,
                    pending.side.value,
                )
                continue
            if order is None:
                pending.status = "ERROR"
                continue
            pending.order_id = order.order_id
            pending.status = "PLACED"

    def _apply_bar_close_rules(
        self, snapshot: PinBarMagicSnapshot, now: datetime
    ) -> None:
        """Apply Pine's bar-close exit rules: Friday market-close and EMA cross.

        Pine reference:
            strategy.close_all(when = hour==16 and dayofweek==friday)
            strategy.close_all(when = crossunder(fastEMA, medmEMA))
            strategy.close_all(when = crossover(fastEMA, medmEMA))

        Note: crossunder closes ALL positions (including longs), and crossover
        closes ALL positions (including shorts) – this matches Pine's close_all.
        """
        position = self._state.active_positions.get(snapshot.symbol)
        if position is None:
            return
        if position.strategy != "pinbar_magic_v2":
            self._log.warning(
                "Bar close rules skipped for %s due to strategy mismatch: %s",
                snapshot.symbol,
                position.strategy,
            )
            return
        close_reason: str | None = None
        if snapshot.friday_close:
            close_reason = "Market close"
        elif self._cfg.enable_ema_cross_close and (
            snapshot.crossunder or snapshot.crossover
        ):
            close_reason = "EMA cross close"
        if close_reason is not None:
            self._log.info(
                "Bar close rules triggered for %s: %s "
                "(crossunder=%s crossover=%s friday_close=%s)",
                snapshot.symbol,
                close_reason,
                snapshot.crossunder,
                snapshot.crossover,
                snapshot.friday_close,
            )
            self._close_position(position, now, close_reason)

    def _apply_bar_trailing(self, snapshot: PinBarMagicSnapshot, now: datetime) -> None:
        """Bar-resolution trailing stop update, simulating intra-bar price movement.

        Because we only receive OHLC data per bar, we reconstruct an approximate
        intra-bar path as: open → (high or low first, based on which wick is longer)
        → (the other extreme) → close.  This matches the common backtest convention.

        The trailing stop is only ever MOVED in the favourable direction (ratcheted).
        When price crosses the trailing stop level within a segment, the position is
        closed at that bar and the exchange stop-loss order is cancelled implicitly
        by _close_position.
        """
        position = self._state.active_positions.get(snapshot.symbol)
        if position is None:
            return
        if position.strategy != "pinbar_magic_v2":
            self._log.warning(
                "Bar trailing skipped for %s due to strategy mismatch: %s",
                snapshot.symbol,
                position.strategy,
            )
            return

        exchange_position = self._exchange.get_position(snapshot.symbol)
        if exchange_position is None:
            self._log.warning(
                "Bar trailing skipped for %s due to missing exchange position",
                snapshot.symbol,
            )
            return
        entry_price = exchange_position.entry_price or position.entry_price
        if entry_price <= 0:
            self._log.warning(
                "Bar trailing skipped for %s due to invalid entry price: %.6f",
                snapshot.symbol,
                entry_price,
            )
            return

        # Approximate intra-bar path: open → first extreme → second extreme → close.
        # Visit the longer wick first (more likely to be the early move).
        if abs(snapshot.bar.open - snapshot.bar.high) < abs(
            snapshot.bar.open - snapshot.bar.low
        ):
            nodes = ("high", "low", "close")
        else:
            nodes = ("low", "high", "close")

        start_price = snapshot.bar.open
        stop_updated = False
        triggered = False
        end_price = start_price  # kept in scope for the triggered-log below

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
                    stop_updated = True
                if position.trailing_active:
                    if (
                        position.extreme_since_activation is None
                        or seg_high > position.extreme_since_activation
                    ):
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
            else:  # SHORT
                activation = entry_price - self._cfg.trail_points
                if not position.trailing_active and seg_low <= activation:
                    position.trailing_active = True
                    position.extreme_since_activation = seg_low
                    position.trailing_stop = seg_low + self._cfg.trail_offset
                    stop_updated = True
                if position.trailing_active:
                    if (
                        position.extreme_since_activation is None
                        or seg_low < position.extreme_since_activation
                    ):
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

        # Push updated stop to exchange only when the level actually changed.
        if stop_updated and not triggered and position.trailing_stop is not None:
            self._log.info(
                "Bar trailing update for %s: entry=%.6f trail=%.6f -> updating stop on exchange",
                snapshot.symbol,
                entry_price,
                position.trailing_stop,
            )
            self._update_stop_loss_on_exchange(
                exchange_position, position.trailing_stop
            )

        if triggered:
            self._log.info(
                "Bar trailing triggered for %s at price %.6f (entry %.6f, trail %.6f)",
                snapshot.symbol,
                end_price,
                entry_price,
                position.trailing_stop,
            )
            self._close_position(position, now, "Trailing stop")

    def _manage_tick_trailing(self, now: datetime) -> None:
        """Tick-resolution trailing stop update, run on every poll cycle.

        Uses the live mark price (or the close of the shortest configured kline
        when use_trailing_tick_emulation is True) to ratchet the trailing stop
        between bar closes.  Logic mirrors _apply_bar_trailing but operates on a
        single price point rather than an OHLC segment.

        When the trailing stop is hit, the position is closed via market order and
        removed from local state.  The exchange stop-loss order will either have
        already triggered or will be cancelled by the close_position call.
        """
        for symbol, position in list(self._state.active_positions.items()):
            if position.strategy != "pinbar_magic_v2":
                continue
            price = self._latest_price_for_trailing(symbol)
            if price is None or price <= 0:
                continue
            entry = position.entry_price
            if entry <= 0:
                continue

            stop_updated = False
            triggered = False

            if position.side == PositionSide.LONG:
                activation = entry + self._cfg.trail_points
                if not position.trailing_active and price >= activation:
                    position.trailing_active = True
                    position.extreme_since_activation = price
                    position.trailing_stop = price - self._cfg.trail_offset
                    stop_updated = True
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
                        stop_updated = True
                    triggered = (
                        position.trailing_stop is not None
                        and price <= position.trailing_stop
                    )
            else:  # SHORT
                activation = entry - self._cfg.trail_points
                if not position.trailing_active and price <= activation:
                    position.trailing_active = True
                    position.extreme_since_activation = price
                    position.trailing_stop = price + self._cfg.trail_offset
                    stop_updated = True
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
                        stop_updated = True
                    triggered = (
                        position.trailing_stop is not None
                        and price >= position.trailing_stop
                    )

            # Push updated stop to exchange only when the level actually changed and
            # the position has not already been triggered (avoid a redundant API call
            # immediately before the close_position market order).
            if stop_updated and not triggered and position.trailing_stop is not None:
                ex_pos = self._exchange.get_position(symbol)
                if ex_pos is not None:
                    self._log.info(
                        "Tick trailing update for %s: price=%.6f entry=%.6f "
                        "trail=%.6f -> updating stop on exchange",
                        symbol,
                        price,
                        entry,
                        position.trailing_stop,
                    )
                    self._update_stop_loss_on_exchange(ex_pos, position.trailing_stop)

            if triggered:
                self._log.info(
                    "Tick trailing triggered for %s at price %.6f "
                    "(entry %.6f, trail %.6f)",
                    symbol,
                    price,
                    entry,
                    position.trailing_stop,
                )
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
        try:
            realized_equity = max(float(self._exchange.get_account_balance()), 0.0)
        except Exception as exc:
            self._log.warning(
                "PinBarMagicCoordinatorV2: get_account_balance failed: %s", exc
            )
            return 0.0

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
        """Compute position size matching Pine's unit calculation.

        Pine:
            units = risk / (entryPrice - stopLoss)   # for longs
            units = risk / (stopLoss - entryPrice)   # for shorts

        Leverage is applied at the execution layer (margin allocation), not here.
        """
        distance = abs(entry_price - stop_price)
        if distance <= 0 or risk_amount <= 0:
            return 0.0
        return risk_amount / distance

    def _place_stop_entry(self, pending: PendingEntryRecord) -> Optional[OrderResult]:
        """Submit a stop-entry order to the exchange.

        Sets margin mode and leverage first; if that fails the entry is aborted so we
        never enter with wrong account settings.

        Prefers place_stop_entry_order if the exchange adapter supports it.
        Falls back to open_limit_position with gap-fill logic for exchanges that
        don't support native stop orders.

        stop_loss is intentionally omitted from the initial stop-entry order and is
        applied in the first _sync_positions / _apply_bar_trailing cycle after fill.
        This avoids race conditions where the exchange rejects a combined stop+SL order
        if the SL is too close to the current price at submission time.
        """
        try:
            self._exchange.set_margin_mode(pending.symbol, pending.margin_mode)
            self._exchange.set_leverage(pending.symbol, pending.leverage)
        except Exception as exc:
            if self._is_insufficient_balance_error(exc):
                raise _InsufficientBalanceError(str(exc)) from exc
            self._log.error(
                "Failed to set account config for %s before entry "
                "(mode=%s, leverage=%sx): %s",
                pending.symbol,
                pending.margin_mode.value,
                pending.leverage,
                exc,
            )
            return None

        placer = getattr(self._exchange, "place_stop_entry_order", None)
        if callable(placer):
            try:
                return placer(
                    symbol=pending.symbol,
                    side=pending.side,
                    quantity=pending.quantity,
                    stop_price=pending.entry_price,
                    leverage=pending.leverage,
                    margin_mode=pending.margin_mode,
                    stop_loss=None,  # applied after fill in the next sync cycle
                )
            except Exception as exc:
                if self._is_insufficient_balance_error(exc):
                    raise _InsufficientBalanceError(str(exc)) from exc
                if self._is_entry_not_triggered_error(exc):
                    raise _EntryDeferredError(str(exc)) from exc
                self._log.warning(
                    "place_stop_entry_order failed for %s (%s): %s",
                    pending.symbol,
                    pending.side.value,
                    exc,
                )

        # Fallback to limit entry with optional gap-fill at current price.
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
                else:
                    raise _EntryDeferredError(
                        "stop entry trigger not reached (gap-fill fallback defers)"
                    )
            else:
                raise _EntryDeferredError(
                    "stop entry trigger not reached (missing last price)"
                )
        else:
            raise _EntryDeferredError(
                "stop entry trigger not reached (no native stop-entry support)"
            )

        try:
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
        except Exception as exc:
            if self._is_insufficient_balance_error(exc):
                raise _InsufficientBalanceError(str(exc)) from exc
            self._log.warning(
                "open_limit_position fallback failed for %s (%s): %s",
                pending.symbol,
                pending.side.value,
                exc,
            )
            return None

    @staticmethod
    def _is_insufficient_balance_error(exc: Exception) -> bool:
        text = str(exc).lower()
        markers = (
            "insufficient balance",
            "insufficient margin",
            "insufficient available",
            "not enough balance",
            "not enough margin",
            "balance not enough",
            "margin not enough",
            "available balance",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _is_entry_not_triggered_error(exc: Exception) -> bool:
        text = str(exc).lower()
        markers = (
            "trigger not reached",
            "stop entry trigger",
            "not triggered",
        )
        return any(marker in text for marker in markers)

    def _close_position(
        self, position: PositionRecord, now: datetime, reason: str
    ) -> None:
        try:
            result = self._exchange.close_position(position.symbol, side=position.side)
        except Exception as exc:
            self._log.error(
                "Failed to close position for %s (%s): %s",
                position.symbol,
                position.side.value,
                exc,
            )
            # Still mark locally as closed to prevent repeated close attempts;
            # operator must investigate the exchange state manually.
            result = None

        exit_price: Optional[float] = None
        if result is not None and result.price > 0:
            exit_price = result.price
        if exit_price is None:
            exit_price = self._get_last_price(position.symbol)

        position.status = "CLOSED"
        position.exit_time = now
        position.exit_price = exit_price
        if exit_price is not None:
            # NOTE: PnL here is notional (pre-leverage, pre-fee).
            # Leverage effect: multiply by position.leverage for levered PnL.
            # Fee and funding are not accounted for.
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
        try:
            return self._exchange.update_stop_loss(
                position=exchange_position,
                stop_price=stop_price,
            )
        except Exception:
            # FIX: was referencing bare `exc` which is not in scope here.
            self._log.warning(
                "Failed to update stop loss for %s to %.6f",
                exchange_position.symbol,
                stop_price,
                exc_info=True,
            )
            return False

    def _cancel_pending_entry(self, pending: PendingEntryRecord, reason: str) -> None:
        cancelled = False
        if pending.order_id:
            try:
                cancelled = bool(
                    self._exchange.cancel_order(
                        symbol=pending.symbol, order_id=pending.order_id
                    )
                )
            except Exception:
                self._log.warning(
                    "Failed to cancel pending order %s for %s (%s)",
                    pending.order_id,
                    pending.symbol,
                    reason,
                    exc_info=True,
                )
                cancelled = False
            if not cancelled:
                try:
                    self._exchange.cancel_all_orders(pending.symbol)
                except Exception:
                    self._log.warning(
                        "Failed to cancel all orders for %s during pending entry "
                        "cancellation (%s)",
                        pending.symbol,
                        reason,
                        exc_info=True,
                    )
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
        self._log.warning(
            "Multiple pending entries found for symbol %s, unable to match position",
            symbol,
        )
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
            price = self._exchange.fetch_price(symbol)
            if price is not None:
                return float(price)
        except Exception:
            return None
        return None

    def _latest_price_for_trailing(self, symbol: str) -> Optional[float]:
        """Resolve the price used for tick-level trailing logic.

        When use_trailing_tick_emulation is True, the close of the shortest
        configured kline interval is used instead of the live mark price.  This
        gives a slightly smoother price series on exchanges where fetch_price
        returns the last trade rather than the mark price.
        """
        if self._cfg.use_trailing_tick_emulation:
            try:
                rows = self._exchange.get_klines(
                    symbol=symbol,
                    interval=self._cfg.trailing_tick_timeframe,
                    limit=2,  # fetch 2 so rows[-2] is the last CLOSED kline
                )
                if rows and len(rows) >= 2:
                    # Use the last CLOSED candle's close, not the still-open one.
                    candle = Candle.from_binance(
                        symbol, self._cfg.trailing_tick_timeframe, rows[-2]
                    )
                    return float(candle.close)
            except Exception:
                pass
        return self._get_last_price(symbol)

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
        except Exception as exc:
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
        except Exception as exc:
            self._log.warning(
                "Failed to send Telegram close notification for %s: %s",
                position.symbol,
                exc,
            )
