"""Order, position, and stop management for EMA + AVWAP."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from candle_downloader.models import Candle

from ..exchange import Position, PositionSide
from ..models import PendingEntryRecord, PositionRecord
from .constants import POSITION_MISS_THRESHOLD
from ._mixin_typing import EmaAvwapMixinTyping
from .state import (
    _EntryCandidate,
    _ExitDecision,
    _PositionRuntime,
    _SymbolSnapshot,
)


class EmaAvwapPositionMixin(EmaAvwapMixinTyping):
    def _sync_positions(self, now: datetime) -> None:
        try:
            exchange_positions = self._exchange.get_current_positions()
        except Exception as exc:
            self._log.warning(
                "EmaAvwapPullback: get_current_positions failed; local state "
                "left unchanged: %s",
                exc,
            )
            return

        by_symbol = {position.symbol: position for position in exchange_positions}

        for symbol, ex_pos in by_symbol.items():
            if symbol in self._state.active_positions:
                self._position_miss_count_by_symbol[symbol] = 0
                continue
            pending = self._find_matching_pending(symbol, ex_pos.side)
            if pending is None:
                self._claim_untracked_exchange_position(ex_pos, now)
                continue
            meta = self._pending_meta_by_key.get(pending.order_key)
            if meta is None:
                self._log.warning(
                    "EmaAvwapPullback: filled pending %s has no metadata; "
                    "claiming position with pending stop only",
                    pending.order_key,
                )
                self._claim_pending_without_runtime(pending, ex_pos, now)
                continue

            entry_price = ex_pos.entry_price or pending.entry_price
            runtime = self._runtime_from_fill(meta.candidate, entry_price)
            stop_price = self._protective_stop_price(
                direction=runtime.direction,
                dynamic_stop=runtime.dynamic_stop_at_entry,
                rigid_stop=runtime.rigid_stop_level,
                trailing_stop=runtime.trailing_stop,
            )
            record = PositionRecord(
                position_id=ex_pos.position_id or pending.order_id or pending.order_key,
                symbol=symbol,
                side=ex_pos.side,
                entry_time=now,
                entry_price=entry_price,
                quantity=ex_pos.size if ex_pos.size > 0 else pending.quantity,
                leverage=pending.leverage,
                margin_mode=pending.margin_mode,
                take_profit=None,
                stop_loss=None,
                risk_amount=pending.risk_amount,
                strategy="ema_avwap_pullback",
                status="OPEN",
                notes=f"Filled from pending {pending.order_key}",
            )
            self._state.active_positions[symbol] = record
            self._position_runtime_by_symbol[symbol] = runtime
            self._position_miss_count_by_symbol[symbol] = 0
            self._state.pending_entries.pop(pending.order_key, None)
            self._pending_meta_by_key.pop(pending.order_key, None)

            stop_ok = self._update_protective_stop(
                record,
                ex_pos,
                stop_price,
                now,
                reason="Initial stop loss",
                allow_widen=True,
            )
            if not stop_ok and self._cfg.emergency_close_on_stop_failure:
                self._log.critical(
                    "EmaAvwapPullback: initial stop failed for %s at %.8f; "
                    "forcing emergency close",
                    symbol,
                    stop_price,
                )
                self._close_position(record, now, "Initial stop placement failed")
            else:
                self._notify_trade_opened(record, runtime, stop_price)
                self._log.info(
                    "EmaAvwapPullback: position opened for %s %s entry=%.8f qty=%.8f stop=%.8f",
                    symbol,
                    ex_pos.side.value,
                    entry_price,
                    record.quantity,
                    stop_price,
                )
                self._save_position_to_db(record)
                self._save_state()

        for symbol, pos in list(self._state.active_positions.items()):
            if symbol in by_symbol:
                self._position_miss_count_by_symbol[symbol] = 0
                continue
            misses = self._position_miss_count_by_symbol.get(symbol, 0) + 1
            self._position_miss_count_by_symbol[symbol] = misses
            if misses < POSITION_MISS_THRESHOLD:
                self._log.warning(
                    "EmaAvwapPullback: position for %s absent from exchange "
                    "(miss %d/%d); waiting before marking closed",
                    symbol,
                    misses,
                    POSITION_MISS_THRESHOLD,
                )
                continue
            self._position_miss_count_by_symbol[symbol] = 0
            pos.status = "CLOSED"
            pos.exit_time = now
            self._state.active_positions.pop(symbol, None)
            self._position_runtime_by_symbol.pop(symbol, None)
            self._state.disable_symbol(symbol, now, self._cfg.disable_symbol_hours)
            self._notify_trade_closed(
                pos,
                reason="Position no longer present on exchange",
                exit_price=None,
            )
            self._save_position_to_db(pos)
            self._save_state()
            self._log.info(
                "EmaAvwapPullback: position for %s confirmed absent for %d polls; "
                "marked closed",
                symbol,
                POSITION_MISS_THRESHOLD,
            )

    def _claim_pending_without_runtime(
        self, pending: PendingEntryRecord, ex_pos: Position, now: datetime
    ) -> None:
        entry_price = ex_pos.entry_price or pending.entry_price
        record = PositionRecord(
            position_id=ex_pos.position_id or pending.order_id or pending.order_key,
            symbol=pending.symbol,
            side=ex_pos.side,
            entry_time=now,
            entry_price=entry_price,
            quantity=ex_pos.size if ex_pos.size > 0 else pending.quantity,
            leverage=pending.leverage,
            margin_mode=pending.margin_mode,
            take_profit=None,
            stop_loss=None,
            risk_amount=pending.risk_amount,
            strategy="ema_avwap_pullback",
            status="OPEN",
            notes=(
                f"Recovered filled pending {pending.order_key} without runtime metadata"
            ),
        )
        self._state.active_positions[pending.symbol] = record
        self._position_miss_count_by_symbol[pending.symbol] = 0
        self._state.pending_entries.pop(pending.order_key, None)
        self._pending_meta_by_key.pop(pending.order_key, None)
        self._update_protective_stop(
            record,
            ex_pos,
            pending.stop_for_risk,
            now,
            reason="Recovered pending stop",
            allow_widen=True,
        )
        self._save_position_to_db(record)
        self._save_state()

    def _claim_untracked_exchange_position(
        self, ex_pos: Position, now: datetime
    ) -> None:
        if ex_pos.symbol not in self._cfg.symbols:
            return
        entry_price = ex_pos.entry_price or self._safe_fetch_price(ex_pos.symbol) or 0.0
        if entry_price <= 0:
            self._log.warning(
                "EmaAvwapPullback: cannot recover untracked exchange position for "
                "%s because entry price is unavailable",
                ex_pos.symbol,
            )
            return
        record = PositionRecord(
            position_id=ex_pos.position_id or f"{ex_pos.symbol}:{ex_pos.side.value}",
            symbol=ex_pos.symbol,
            side=ex_pos.side,
            entry_time=now,
            entry_price=entry_price,
            quantity=ex_pos.size,
            leverage=int(ex_pos.leverage) if ex_pos.leverage else self._cfg.leverage,
            margin_mode=ex_pos.margin_mode,
            take_profit=None,
            stop_loss=None,
            risk_amount=None,
            strategy="ema_avwap_pullback",
            status="OPEN",
            notes=(
                "Recovered from exchange without local EMA+AVWAP runtime metadata; "
                "preserving existing exchange stop only"
            ),
        )
        self._state.active_positions[ex_pos.symbol] = record
        self._position_miss_count_by_symbol[ex_pos.symbol] = 0
        self._log.warning(
            "EmaAvwapPullback: claimed untracked exchange position for %s %s; "
            "runtime metadata is unavailable, so AVWAP trailing is suspended "
            "until this position is closed",
            ex_pos.symbol,
            ex_pos.side.value,
        )
        self._save_position_to_db(record)
        self._save_state()

    def _runtime_from_fill(
        self, candidate: _EntryCandidate, actual_entry_price: float
    ) -> _PositionRuntime:
        rigid_stop = self._rigid_stop_level(candidate.direction, actual_entry_price)
        return _PositionRuntime(
            direction=candidate.direction,
            anchor_time=candidate.anchor_time,
            setup_detected_time=candidate.setup_detected_time,
            entry_signal_time=candidate.signal_time,
            raw_entry_price=candidate.raw_entry_price,
            dynamic_stop_at_entry=candidate.dynamic_stop_at_entry,
            rigid_stop_level=rigid_stop,
            trailing_activation_at_entry=candidate.trailing_activation_at_entry,
            entry_trigger_mode=candidate.entry_trigger_mode,
            risk_amount_interpretation=candidate.risk_amount_interpretation,
            position_sizing_mode=self._cfg.position_sizing_mode,
            last_avwap=candidate.avwap,
        )

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _manage_position_on_bar(self, snapshot: _SymbolSnapshot, now: datetime) -> None:
        record = self._state.active_positions.get(snapshot.symbol)
        if record is None or record.strategy != "ema_avwap_pullback":
            return
        runtime = self._position_runtime_by_symbol.get(snapshot.symbol)
        if runtime is None:
            self._log.warning(
                "EmaAvwapPullback: missing runtime metadata for %s; preserving "
                "existing exchange stop only",
                snapshot.symbol,
            )
            return
        exchange_position = self._safe_get_position(snapshot.symbol)
        if exchange_position is None:
            self._log.warning(
                "EmaAvwapPullback: bar management skipped for %s because exchange "
                "position is not visible",
                snapshot.symbol,
            )
            return

        try:
            anchor_index = self._find_anchor_index_by_time(
                snapshot.candles, runtime.anchor_time
            )
            avwap = self._build_avwap_snapshot(
                candles=snapshot.candles,
                anchor_index=anchor_index,
                candle_index=snapshot.candle_index,
                tpv_prefix=snapshot.tpv_prefix,
                vol_prefix=snapshot.vol_prefix,
                tpv2_prefix=snapshot.tpv2_prefix,
            )
        except Exception as exc:
            self._log.warning(
                "EmaAvwapPullback: could not rebuild AVWAP for active %s; "
                "keeping previous stop: %s",
                snapshot.symbol,
                exc,
            )
            return

        runtime.last_avwap = avwap
        stop_level = self._dynamic_stop_from_avwap(runtime.direction, avwap)
        activation_level = self._trailing_activation_level(runtime.direction, avwap)
        exit_decision = self._process_position_for_candle(
            runtime=runtime,
            candle=snapshot.candle,
            prev_close=snapshot.previous_candle.close,
            stop_level=stop_level,
            activation_level=activation_level,
        )
        self._sync_runtime_to_record(record, runtime)
        self._save_state()
        if exit_decision is not None:
            self._log.warning(
                "EmaAvwapPullback: bar stop condition detected for %s via %s "
                "(raw level %.8f); closing market",
                snapshot.symbol,
                exit_decision.reason,
                exit_decision.raw_exit_price,
            )
            self._close_position(record, now, exit_decision.reason)
            return

        protective_stop = self._protective_stop_price(
            direction=runtime.direction,
            dynamic_stop=stop_level,
            rigid_stop=runtime.rigid_stop_level,
            trailing_stop=runtime.trailing_stop if runtime.trailing_active else None,
        )
        self._update_protective_stop(
            record,
            exchange_position,
            protective_stop,
            now,
            reason="Bar stop update",
            allow_widen=self._cfg.allow_dynamic_stop_widening,
        )

    def _manage_tick_trailing(self, now: datetime) -> None:
        for symbol, record in list(self._state.active_positions.items()):
            if record.strategy != "ema_avwap_pullback":
                continue
            runtime = self._position_runtime_by_symbol.get(symbol)
            if runtime is None or runtime.last_avwap is None:
                continue
            price = self._latest_price_for_trailing(symbol)
            if price is None or price <= 0:
                continue
            activation = self._trailing_activation_level(
                runtime.direction, runtime.last_avwap
            )
            stop_updated = False
            triggered = False
            if runtime.direction == "long":
                if not runtime.trailing_active and price >= activation:
                    runtime.trailing_active = True
                    runtime.extreme_price = price
                    runtime.trailing_stop = self._constrain_trailing_stop(
                        runtime, price * (1.0 - self._cfg.trailing_gap_pct / 100.0)
                    )
                    stop_updated = True
                if runtime.trailing_active:
                    if runtime.extreme_price is None or price > runtime.extreme_price:
                        runtime.extreme_price = price
                        runtime.trailing_stop = self._constrain_trailing_stop(
                            runtime,
                            price * (1.0 - self._cfg.trailing_gap_pct / 100.0),
                        )
                        stop_updated = True
                    triggered = (
                        runtime.trailing_stop is not None
                        and price <= runtime.trailing_stop
                    )
            else:
                if not runtime.trailing_active and price <= activation:
                    runtime.trailing_active = True
                    runtime.extreme_price = price
                    runtime.trailing_stop = self._constrain_trailing_stop(
                        runtime, price * (1.0 + self._cfg.trailing_gap_pct / 100.0)
                    )
                    stop_updated = True
                if runtime.trailing_active:
                    if runtime.extreme_price is None or price < runtime.extreme_price:
                        runtime.extreme_price = price
                        runtime.trailing_stop = self._constrain_trailing_stop(
                            runtime,
                            price * (1.0 + self._cfg.trailing_gap_pct / 100.0),
                        )
                        stop_updated = True
                    triggered = (
                        runtime.trailing_stop is not None
                        and price >= runtime.trailing_stop
                    )

            self._sync_runtime_to_record(record, runtime)
            self._save_state()
            if triggered:
                self._log.warning(
                    "EmaAvwapPullback: tick trailing triggered for %s price=%.8f stop=%.8f",
                    symbol,
                    price,
                    runtime.trailing_stop or 0.0,
                )
                self._close_position(record, now, "Trailing stop")
                continue
            if not stop_updated or runtime.trailing_stop is None:
                continue
            exchange_position = self._safe_get_position(symbol)
            if exchange_position is None:
                continue
            dynamic_stop = self._dynamic_stop_from_avwap(
                runtime.direction, runtime.last_avwap
            )
            protective_stop = self._protective_stop_price(
                direction=runtime.direction,
                dynamic_stop=dynamic_stop,
                rigid_stop=runtime.rigid_stop_level,
                trailing_stop=runtime.trailing_stop,
            )
            self._update_protective_stop(
                record,
                exchange_position,
                protective_stop,
                now,
                reason="Tick trailing update",
                allow_widen=False,
            )

    def _process_position_for_candle(
        self,
        *,
        runtime: _PositionRuntime,
        candle: Candle,
        prev_close: float,
        stop_level: float,
        activation_level: float,
    ) -> _ExitDecision | None:
        rigid_stop = runtime.rigid_stop_level
        if runtime.direction == "long":
            gap_exit = self._check_long_gap_exit(
                prev_close=prev_close,
                open_price=candle.open,
                stop_level=stop_level,
                rigid_stop_level=rigid_stop,
                trailing_stop=runtime.trailing_stop
                if runtime.trailing_active
                else None,
            )
            if gap_exit is not None:
                return _ExitDecision(
                    gap_exit[0], gap_exit[1], stop_level, activation_level
                )
            open_exit = self._check_long_open_exit(
                open_price=candle.open,
                stop_level=stop_level,
                rigid_stop_level=rigid_stop,
                trailing_stop=runtime.trailing_stop
                if runtime.trailing_active
                else None,
            )
            if open_exit is not None:
                return _ExitDecision(
                    open_exit[0], open_exit[1], stop_level, activation_level
                )
            if not runtime.trailing_active and candle.open >= activation_level:
                self._activate_long_trailing(runtime, candle.open)

            start_price = candle.open
            for end_price in self._price_path(candle):
                if end_price >= start_price:
                    if (
                        not runtime.trailing_active
                        and start_price <= activation_level <= end_price
                    ):
                        self._activate_long_trailing(runtime, end_price)
                    elif runtime.trailing_active:
                        self._update_long_trailing(runtime, end_price)
                else:
                    adverse_exit = self._first_long_downside_exit(
                        start_price=start_price,
                        end_price=end_price,
                        stop_level=stop_level,
                        rigid_stop_level=rigid_stop,
                        trailing_stop=runtime.trailing_stop
                        if runtime.trailing_active
                        else None,
                    )
                    if adverse_exit is not None:
                        return _ExitDecision(
                            adverse_exit[0],
                            adverse_exit[1],
                            stop_level,
                            activation_level,
                        )
                start_price = end_price
            return None

        gap_exit = self._check_short_gap_exit(
            prev_close=prev_close,
            open_price=candle.open,
            stop_level=stop_level,
            rigid_stop_level=rigid_stop,
            trailing_stop=runtime.trailing_stop if runtime.trailing_active else None,
        )
        if gap_exit is not None:
            return _ExitDecision(gap_exit[0], gap_exit[1], stop_level, activation_level)
        open_exit = self._check_short_open_exit(
            open_price=candle.open,
            stop_level=stop_level,
            rigid_stop_level=rigid_stop,
            trailing_stop=runtime.trailing_stop if runtime.trailing_active else None,
        )
        if open_exit is not None:
            return _ExitDecision(
                open_exit[0], open_exit[1], stop_level, activation_level
            )
        if not runtime.trailing_active and candle.open <= activation_level:
            self._activate_short_trailing(runtime, candle.open)

        start_price = candle.open
        for end_price in self._price_path(candle):
            if end_price <= start_price:
                if (
                    not runtime.trailing_active
                    and start_price >= activation_level >= end_price
                ):
                    self._activate_short_trailing(runtime, end_price)
                elif runtime.trailing_active:
                    self._update_short_trailing(runtime, end_price)
            else:
                adverse_exit = self._first_short_upside_exit(
                    start_price=start_price,
                    end_price=end_price,
                    stop_level=stop_level,
                    rigid_stop_level=rigid_stop,
                    trailing_stop=runtime.trailing_stop
                    if runtime.trailing_active
                    else None,
                )
                if adverse_exit is not None:
                    return _ExitDecision(
                        adverse_exit[0],
                        adverse_exit[1],
                        stop_level,
                        activation_level,
                    )
            start_price = end_price
        return None

    # ------------------------------------------------------------------
    # Stop / close helpers
    # ------------------------------------------------------------------

    def _update_protective_stop(
        self,
        record: PositionRecord,
        exchange_position: Position,
        stop_price: float,
        now: datetime,
        *,
        reason: str,
        allow_widen: bool,
    ) -> bool:
        mark_price = self._safe_fetch_price(record.symbol)
        if mark_price is not None and self._is_stop_breached_by_price(
            direction=self._direction_from_side(record.side),
            price=mark_price,
            stop_price=stop_price,
        ):
            self._log.warning(
                "EmaAvwapPullback: %s stop %.8f already breached by mark %.8f "
                "for %s; closing market",
                reason,
                stop_price,
                mark_price,
                record.symbol,
            )
            self._close_position(record, now, reason)
            return False

        previous = record.stop_loss
        if previous is not None:
            direction = self._direction_from_side(record.side)
            if not allow_widen and self._is_less_protective(
                direction, stop_price, previous
            ):
                self._log.info(
                    "EmaAvwapPullback: skipping less protective stop update for %s "
                    "(previous=%.8f candidate=%.8f)",
                    record.symbol,
                    previous,
                    stop_price,
                )
                return True
            if self._cfg.min_stop_update_pct > 0:
                change_pct = abs(stop_price - previous) / max(abs(previous), 1e-12)
                if change_pct < self._cfg.min_stop_update_pct / 100.0:
                    return True

        ok = self._update_stop_loss_on_exchange(exchange_position, stop_price)
        if ok:
            record.stop_loss = stop_price
            self._save_position_to_db(record)
            self._save_state()
            return True
        if previous is None and self._cfg.emergency_close_on_stop_failure:
            self._log.critical(
                "EmaAvwapPullback: no protective stop is confirmed for %s after "
                "%s failure",
                record.symbol,
                reason,
            )
        return False

    def _update_stop_loss_on_exchange(
        self, exchange_position: Position, stop_price: float
    ) -> bool:
        if not exchange_position.position_id:
            self._log.warning(
                "EmaAvwapPullback: cannot update stop for %s without position_id",
                exchange_position.symbol,
            )
            return False

        updater = getattr(self._exchange, "update_position_stop_loss", None)
        if callable(updater):
            try:
                if bool(updater(exchange_position, stop_price)):
                    self._log.info(
                        "EmaAvwapPullback: updated position stop for %s to %.8f",
                        exchange_position.symbol,
                        stop_price,
                    )
                    return True
            except Exception:
                self._log.warning(
                    "EmaAvwapPullback: update_position_stop_loss failed for %s",
                    exchange_position.symbol,
                    exc_info=True,
                )

        placer = getattr(self._exchange, "place_position_tpsl_order", None)
        if callable(placer):
            try:
                result = placer(
                    symbol=exchange_position.symbol,
                    position_id=exchange_position.position_id,
                    sl_price=stop_price,
                    sl_stop_type="MARK_PRICE",
                )
                if result:
                    self._log.info(
                        "EmaAvwapPullback: placed/updated position stop for %s to %.8f",
                        exchange_position.symbol,
                        stop_price,
                    )
                    return True
            except Exception:
                self._log.warning(
                    "EmaAvwapPullback: place_position_tpsl_order failed for %s",
                    exchange_position.symbol,
                    exc_info=True,
                )

        order_placer = getattr(self._exchange, "place_stop_loss_order", None)
        if callable(order_placer):
            try:
                order_id = order_placer(exchange_position, stop_price)
                if order_id:
                    self._log.info(
                        "EmaAvwapPullback: placed order-level stop for %s to %.8f order=%s",
                        exchange_position.symbol,
                        stop_price,
                        order_id,
                    )
                    return True
            except Exception:
                self._log.warning(
                    "EmaAvwapPullback: place_stop_loss_order failed for %s",
                    exchange_position.symbol,
                    exc_info=True,
                )
        return False

    def _close_position(
        self, position: PositionRecord, now: datetime, reason: str
    ) -> None:
        try:
            result = self._retry(
                lambda: self._exchange.close_position(
                    position.symbol, side=position.side
                ),
                f"close_position {position.symbol}",
            )
        except Exception as exc:
            self._log.error(
                "EmaAvwapPullback: failed to close %s %s for %s: %s",
                position.symbol,
                position.side.value,
                reason,
                exc,
            )
            position.notes = f"{position.notes}; close failed: {reason}".strip("; ")
            self._save_position_to_db(position)
            self._save_state()
            return

        exit_price: Optional[float] = result.price if result.price > 0 else None
        if exit_price is None:
            exit_price = self._safe_fetch_price(position.symbol)
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
        self._position_runtime_by_symbol.pop(position.symbol, None)
        self._state.disable_symbol(position.symbol, now, self._cfg.disable_symbol_hours)
        self._notify_trade_closed(position, reason=reason, exit_price=exit_price)
        self._save_position_to_db(position)
        self._save_state()
        self._log.info("EmaAvwapPullback: closed %s due to %s", position.symbol, reason)

    # ------------------------------------------------------------------
    # Stale pending entries
    # ------------------------------------------------------------------

    def _cancel_stale_entries(self, snapshot: _SymbolSnapshot, now: datetime) -> None:
        tf_seconds = max(self._timeframe_seconds(self._cfg.timeframe), 1)
        for key, pending in list(self._state.pending_entries.items()):
            if pending.symbol != snapshot.symbol:
                continue
            reference = snapshot.candle.close_time
            bars_since = int(
                max(0.0, (reference - pending.signal_time).total_seconds())
                // tf_seconds
            )
            if bars_since > self._cfg.entry_cancel_bars:
                self._cancel_pending_entry(
                    pending, f"entry timeout after {bars_since} bars"
                )

    def _cancel_pending_entry(self, pending: PendingEntryRecord, reason: str) -> None:
        if pending.order_id:
            try:
                cancelled = bool(
                    self._exchange.cancel_order(
                        symbol=pending.symbol, order_id=pending.order_id
                    )
                )
            except Exception:
                self._log.warning(
                    "EmaAvwapPullback: failed to cancel pending order %s for %s (%s)",
                    pending.order_id,
                    pending.symbol,
                    reason,
                    exc_info=True,
                )
                cancelled = False
            if not cancelled:
                self._log.warning(
                    "EmaAvwapPullback: cancel by order id failed for %s order=%s "
                    "(%s); skipping cancel_all_orders to avoid removing protective stops",
                    pending.symbol,
                    pending.order_id,
                    reason,
                )
        pending.status = "CANCELLED"
        pending.notes = f"{pending.notes}; {reason}".strip("; ")
        self._state.pending_entries.pop(pending.order_key, None)
        self._pending_meta_by_key.pop(pending.order_key, None)
        self._save_state()

    # ------------------------------------------------------------------
    # Backtest-aligned math helpers
