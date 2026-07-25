"""Order, position, and stop management for EMA + AVWAP."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from candle_downloader.models import Candle

from ..exchange import Position, PositionSide
from ..models import PendingEntryRecord, PositionRecord
from .constants import POSITION_MISS_THRESHOLD
from ._mixin_typing import EmaAvwapMixinTyping
from .config import ExitMode
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
            self._position_sync_healthy = False
            self._log.warning(
                "EmaAvwapPullback: get_current_positions failed; local state "
                "left unchanged and new entries are blocked: %s",
                exc,
            )
            return

        by_symbol: dict[str, Position] = {}
        for position in exchange_positions:
            if position.symbol in by_symbol:
                # This should be prevented at startup on Bitunix, but never
                # silently discard one side if the account changes underneath us.
                self._log.critical(
                    "EmaAvwapPullback: multiple exchange positions found for %s; "
                    "refusing to manage an ambiguous hedge state",
                    position.symbol,
                )
                self._position_sync_healthy = False
                return
            by_symbol[position.symbol] = position

        for symbol, ex_pos in by_symbol.items():
            if symbol in self._state.active_positions:
                self._sync_active_exchange_position(
                    self._state.active_positions[symbol], ex_pos, now
                )
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
            initial_rigid_stop = meta.candidate.rigid_stop_at_entry
            if initial_rigid_stop is None and pending.stop_for_risk > 0:
                initial_rigid_stop = pending.stop_for_risk
            required_stop = initial_rigid_stop
            actual_stop = self._confirm_initial_stop(ex_pos, required_stop)
            runtime = self._runtime_from_fill(meta.candidate, actual_stop)
            stop_price = actual_stop
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
                # Never mirror the requested stop: retain only the native stop
                # read back from the exchange after the fill.
                stop_loss=stop_price,
                risk_amount=pending.risk_amount,
                strategy="ema_avwap_pullback",
                status="OPEN",
                notes=(
                    f"Filled from pending {pending.order_key}; "
                    f"entry_mode={runtime.entry_mode.value}; "
                    f"exit_mode={runtime.exit_mode.value}; "
                    f"exit_band={runtime.exit_band.value}"
                ),
            )
            self._state.active_positions[symbol] = record
            self._position_runtime_by_symbol[symbol] = runtime
            self._position_miss_count_by_symbol[symbol] = 0
            order_status = self._reconcile_pending_order(pending)
            if self._order_is_terminal(order_status):
                self._remove_pending_entry(pending)

            if actual_stop is None:
                self._log.critical(
                    "EmaAvwapPullback: exchange protective stop could not be "
                    "confirmed for newly filled %s; cancelling any remainder and "
                    "emergency-closing the filled exposure",
                    symbol,
                )
                if pending.order_id:
                    self._cancel_pending_entry(
                        pending, "emergency stop-confirmation failure"
                    )
                self._save_position_to_db(record)
                self._save_state()
                self._close_position(
                    record,
                    now,
                    "Emergency close: protective stop could not be confirmed",
                )
                continue

            self._notify_trade_opened(record, runtime, stop_price)
            self._log.info(
                "EmaAvwapPullback: ENTRY EXECUTION filled entry_mode=%s symbol=%s "
                "side=%s entry=%.8f qty=%.8f rigid_stop=%s trigger=%s",
                runtime.entry_mode.value,
                symbol,
                ex_pos.side.value,
                entry_price,
                record.quantity,
                f"{stop_price:.8f}" if stop_price is not None else "disabled",
                runtime.entry_trigger_mode,
            )
            self._save_position_to_db(record)
            self._save_state()

        self._clear_terminal_unfilled_entries(by_symbol)

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
            runtime = self._position_runtime_by_symbol.pop(symbol, None)
            self._state.disable_symbol(symbol, now, self._cfg.disable_symbol_hours)
            self._notify_trade_closed(
                pos,
                reason="Position no longer present on exchange",
                exit_price=None,
                runtime=runtime,
            )
            self._save_position_to_db(pos)
            self._save_state()
            self._log.info(
                "EmaAvwapPullback: position for %s confirmed absent for %d polls; "
                "marked closed",
                symbol,
                POSITION_MISS_THRESHOLD,
            )
        self._position_sync_healthy = True

    def _sync_active_exchange_position(
        self, record: PositionRecord, exchange_position: Position, now: datetime
    ) -> None:
        changed = False
        if exchange_position.position_id and record.position_id != exchange_position.position_id:
            record.position_id = exchange_position.position_id
            changed = True
        if exchange_position.size > 0 and record.quantity != exchange_position.size:
            self._log.warning(
                "EmaAvwapPullback: exchange position size changed for %s "
                "(local=%.8f exchange=%.8f); retaining the entry order until its "
                "authoritative terminal state is known",
                record.symbol,
                record.quantity,
                exchange_position.size,
            )
            record.quantity = exchange_position.size
            changed = True
        if exchange_position.entry_price > 0 and record.entry_price != exchange_position.entry_price:
            record.entry_price = exchange_position.entry_price
            changed = True

        pending = self._find_matching_pending(record.symbol, exchange_position.side)
        if pending is not None:
            status = self._reconcile_pending_order(pending)
            if self._order_is_terminal(status):
                self._remove_pending_entry(pending)
                changed = True

        runtime = self._position_runtime_by_symbol.get(record.symbol)
        required_stop = runtime.rigid_stop_level if runtime is not None else None
        if record.stop_loss is None and required_stop is not None:
            actual_stop = self._confirm_initial_stop(exchange_position, required_stop)
            if actual_stop is None:
                self._log.critical(
                    "EmaAvwapPullback: active %s has no confirmed native stop; "
                    "emergency-closing it",
                    record.symbol,
                )
                if pending is not None and pending.order_id:
                    self._cancel_pending_entry(
                        pending, "emergency stop-confirmation failure"
                    )
                if changed:
                    self._save_position_to_db(record)
                    self._save_state()
                self._close_position(
                    record,
                    now,
                    "Emergency close: protective stop could not be confirmed",
                )
                return
            record.stop_loss = actual_stop
            runtime.rigid_stop_level = actual_stop
            changed = True

        if changed:
            self._save_position_to_db(record)
            self._save_state()

    @staticmethod
    def _order_status_name(order: Optional[dict[str, Any]]) -> str:
        if not order:
            return ""
        return str(order.get("status") or order.get("orderStatus") or "").upper()

    @classmethod
    def _order_is_live(cls, status: str) -> bool:
        return status in {"INIT", "NEW", "OPEN", "PENDING", "PART_FILLED", "PARTIALLY_FILLED"}

    @classmethod
    def _order_is_terminal(cls, status: str) -> bool:
        return status in {"FILLED", "CANCELLED", "CANCELED", "REJECTED", "EXPIRED"}

    def _reconcile_pending_order(self, pending: PendingEntryRecord) -> str:
        lookup = getattr(self._exchange, "get_order_status", None)
        if not callable(lookup):
            return ""
        try:
            order = lookup(
                symbol=pending.symbol,
                order_id=pending.order_id,
                client_id=pending.client_id,
            )
        except Exception:
            self._log.warning(
                "EmaAvwapPullback: order reconciliation failed for %s",
                pending.order_key,
                exc_info=True,
            )
            return ""
        if not isinstance(order, dict) or not order:
            return ""
        exchange_order_id = str(order.get("orderId") or order.get("id") or "").strip()
        changed = False
        if exchange_order_id and pending.order_id != exchange_order_id:
            pending.order_id = exchange_order_id
            changed = True
        status = self._order_status_name(order)
        if status and pending.status != "PLACED":
            pending.status = "PLACED"
            changed = True
        if changed:
            self._save_state()
        return status

    def _clear_terminal_unfilled_entries(
        self, positions_by_symbol: dict[str, Position]
    ) -> None:
        for pending in list(self._state.pending_entries.values()):
            if pending.symbol in positions_by_symbol:
                continue
            status = self._reconcile_pending_order(pending)
            if status in {"CANCELLED", "CANCELED", "REJECTED", "EXPIRED"}:
                self._log.info(
                    "EmaAvwapPullback: removing terminal unfilled order %s (%s)",
                    pending.order_key,
                    status,
                )
                self._remove_pending_entry(pending)
                self._save_state()

    def _remove_pending_entry(self, pending: PendingEntryRecord) -> None:
        self._state.pending_entries.pop(pending.order_key, None)
        self._pending_meta_by_key.pop(pending.order_key, None)

    def _confirm_initial_stop(
        self, exchange_position: Position, required_stop: Optional[float]
    ) -> Optional[float]:
        if required_stop is None or required_stop <= 0:
            return None
        confirmer = getattr(self._exchange, "ensure_position_stop_loss", None)
        if not callable(confirmer):
            self._log.critical(
                "EmaAvwapPullback: exchange cannot confirm a native position stop "
                "for %s",
                exchange_position.symbol,
            )
            return None
        try:
            actual = confirmer(exchange_position, required_stop)
        except Exception:
            self._log.critical(
                "EmaAvwapPullback: native stop confirmation failed for %s",
                exchange_position.symbol,
                exc_info=True,
            )
            return None
        try:
            actual_value = float(actual) if actual is not None else 0.0
        except (TypeError, ValueError):
            actual_value = 0.0
        if actual_value <= 0:
            return None
        if exchange_position.side == PositionSide.LONG and actual_value < required_stop:
            return None
        if exchange_position.side == PositionSide.SHORT and actual_value > required_stop:
            return None
        return actual_value

    def _claim_pending_without_runtime(
        self, pending: PendingEntryRecord, ex_pos: Position, now: datetime
    ) -> None:
        entry_price = ex_pos.entry_price or pending.entry_price
        actual_stop = self._confirm_initial_stop(
            ex_pos, pending.stop_for_risk if pending.stop_for_risk > 0 else None
        )
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
            stop_loss=actual_stop,
            risk_amount=pending.risk_amount,
            strategy="ema_avwap_pullback",
            status="OPEN",
            notes=(
                f"Recovered filled pending {pending.order_key} without runtime metadata"
            ),
        )
        self._state.active_positions[pending.symbol] = record
        self._position_miss_count_by_symbol[pending.symbol] = 0
        order_status = self._reconcile_pending_order(pending)
        if self._order_is_terminal(order_status):
            self._remove_pending_entry(pending)
        self._save_position_to_db(record)
        self._save_state()
        if actual_stop is not None:
            return
        self._log.critical(
            "EmaAvwapPullback: recovered %s without a confirmed native stop; "
            "cancelling any remainder and emergency-closing it",
            pending.symbol,
        )
        if pending.order_id:
            self._cancel_pending_entry(pending, "emergency stop-confirmation failure")
        self._close_position(
            record,
            now,
            "Emergency close: protective stop could not be confirmed",
        )

    def _claim_untracked_exchange_position(
        self, ex_pos: Position, now: datetime
    ) -> None:
        # Account-wide concurrency limits must include manual/other-bot
        # positions, even when their symbol is outside this strategy's scan.
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
            "runtime metadata is unavailable, so AVWAP target management is suspended "
            "until this position is closed",
            ex_pos.symbol,
            ex_pos.side.value,
        )
        self._save_position_to_db(record)
        self._save_state()

    def _runtime_from_fill(
        self, candidate: _EntryCandidate, initial_rigid_stop: float | None
    ) -> _PositionRuntime:
        return _PositionRuntime(
            direction=candidate.direction,
            anchor_time=candidate.anchor_time,
            setup_detected_time=candidate.setup_detected_time,
            entry_signal_time=candidate.signal_time,
            raw_entry_price=candidate.raw_entry_price,
            dynamic_stop_at_entry=candidate.dynamic_stop_at_entry,
            # Preserve the stop submitted with the opening order. A better limit
            # fill must not cause the rigid stop to be recalculated or replaced.
            rigid_stop_level=initial_rigid_stop,
            trailing_activation_at_entry=candidate.trailing_activation_at_entry,
            entry_trigger_mode=candidate.entry_trigger_mode,
            last_avwap=candidate.avwap,
            entry_mode=candidate.entry_mode,
            exit_mode=candidate.exit_mode,
            exit_band=candidate.exit_band,
            last_ema_value=candidate.ema_value,
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
                "EmaAvwapPullback: exit_mode=%s could not refresh closed-bar "
                "AVWAP for trailing management on %s: %s",
                runtime.exit_mode.value,
                snapshot.symbol,
                exc,
            )
            return

        runtime.last_avwap = avwap
        runtime.last_ema_value = snapshot.ema_value
        self._sync_runtime_to_record(record, runtime)
        self._save_state()
        self._log.info(
            "EmaAvwapPullback: trailing AVWAP refresh exit_mode=%s symbol=%s "
            "closed_candle=%s middle=%.8f stdev=%.8f "
            "upper1=%.8f lower1=%.8f upper2=%.8f lower2=%.8f "
            "target_band=%d target=%.8f rigid_stop=%s",
            runtime.exit_mode.value,
            snapshot.symbol,
            snapshot.candle.close_time.isoformat(),
            avwap.vwap,
            avwap.stdev,
            avwap.upper1,
            avwap.lower1,
            avwap.upper2,
            avwap.lower2,
            runtime.exit_band.number,
            self._target_band_level(
                runtime.direction, avwap, runtime.exit_band
            ),
            f"{runtime.rigid_stop_level:.8f}"
            if runtime.rigid_stop_level is not None
            else "disabled",
        )
        if runtime.exit_mode is not ExitMode.CLOSE:
            return

        target_band = runtime.exit_band.number
        target = self._target_band_level(runtime.direction, avwap, runtime.exit_band)
        closed_price = snapshot.candle.close
        if not self._target_is_touched(
            direction=runtime.direction,
            price=closed_price,
            target=target,
        ):
            return

        reason = f"AVWAP band {target_band} target on candle close"
        self._log.info(
            "EmaAvwapPullback: EXIT SIGNAL exit_mode=%s symbol=%s reason=%s "
            "closed_price=%.8f target=%.8f middle=%.8f candle=%s",
            runtime.exit_mode.value,
            snapshot.symbol,
            reason,
            closed_price,
            target,
            avwap.vwap,
            snapshot.candle.close_time.isoformat(),
        )
        self._notify_exit_signal(
            record,
            runtime=runtime,
            reason=reason,
            live_price=closed_price,
            target_price=target,
            avwap=avwap,
        )
        self._close_position(record, now, reason)

    def _manage_live_position_exits(self, now: datetime) -> None:
        for symbol, record in list(self._state.active_positions.items()):
            if record.strategy != "ema_avwap_pullback":
                continue
            runtime = self._position_runtime_by_symbol.get(symbol)
            if runtime is None:
                continue
            if runtime.exit_mode is ExitMode.CLOSE:
                # CLOSE governs only the AVWAP profit target. Protective stops
                # remain live; check the confirmed rigid stop without building
                # an on-the-fly AVWAP snapshot.
                price = self._safe_fetch_price(symbol)
                if (
                    price is not None
                    and price > 0
                    and runtime.rigid_stop_level is not None
                    and self._is_stop_breached_by_price(
                        direction=runtime.direction,
                        price=price,
                        stop_price=runtime.rigid_stop_level,
                    )
                ):
                    reason = "Rigid stop loss"
                    if runtime.last_avwap is not None:
                        self._notify_exit_signal(
                            record,
                            runtime=runtime,
                            reason=reason,
                            live_price=price,
                            target_price=runtime.rigid_stop_level,
                            avwap=runtime.last_avwap,
                        )
                    self._close_position(record, now, reason)
                continue
            snapshot = self._last_snapshot_by_symbol.get(symbol)
            if snapshot is None:
                continue
            try:
                live_snapshot, avwap = self._build_live_avwap_snapshot(
                    snapshot, runtime
                )
            except Exception as exc:
                self._log.warning(
                    "EmaAvwapPullback: exit_mode=%s live exit indicator refresh "
                    "failed for %s: %s",
                    runtime.exit_mode.value,
                    symbol,
                    exc,
                )
                continue
            runtime.last_ema_value = live_snapshot.ema_value
            price = self._safe_fetch_price(symbol)
            if price is None or price <= 0:
                continue
            target_band = runtime.exit_band.number
            target = self._target_band_level(
                runtime.direction, avwap, runtime.exit_band
            )
            self._log.info(
                "EmaAvwapPullback: live exit evaluation exit_mode=%s symbol=%s "
                "direction=%s live=%.8f forming_close=%s ema=%.8f "
                "middle=%.8f target_band=%d target=%.8f upper1=%.8f "
                "lower1=%.8f upper2=%.8f lower2=%.8f rigid_stop=%s",
                runtime.exit_mode.value,
                symbol,
                runtime.direction,
                price,
                live_snapshot.candle.close_time.isoformat(),
                live_snapshot.ema_value,
                avwap.vwap,
                target_band,
                target,
                avwap.upper1,
                avwap.lower1,
                avwap.upper2,
                avwap.lower2,
                f"{runtime.rigid_stop_level:.8f}"
                if runtime.rigid_stop_level is not None
                else "disabled",
            )
            self._save_state()
            if runtime.exit_mode is ExitMode.LIVE and self._target_is_touched(
                direction=runtime.direction, price=price, target=target
            ):
                reason = f"AVWAP band {target_band} target"
                self._log.info(
                    "EmaAvwapPullback: EXIT SIGNAL exit_mode=%s symbol=%s "
                    "reason=%s live=%.8f target=%.8f middle=%.8f",
                    runtime.exit_mode.value,
                    symbol,
                    reason,
                    price,
                    target,
                    avwap.vwap,
                )
                self._notify_exit_signal(
                    record,
                    runtime=runtime,
                    reason=reason,
                    live_price=price,
                    target_price=target,
                    avwap=avwap,
                )
                self._close_position(record, now, reason)
                continue
            if runtime.rigid_stop_level is not None and self._is_stop_breached_by_price(
                direction=runtime.direction,
                price=price,
                stop_price=runtime.rigid_stop_level,
            ):
                reason = "Rigid stop loss"
                self._log.warning(
                    "EmaAvwapPullback: EXIT SIGNAL exit_mode=%s symbol=%s "
                    "reason=%s live=%.8f rigid_stop=%.8f",
                    runtime.exit_mode.value,
                    symbol,
                    reason,
                    price,
                    runtime.rigid_stop_level,
                )
                self._notify_exit_signal(
                    record,
                    runtime=runtime,
                    reason=reason,
                    live_price=price,
                    target_price=runtime.rigid_stop_level,
                    avwap=avwap,
                )
                self._close_position(record, now, reason)

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
                    "EmaAvwapPullback: tick trailing triggered for %s "
                    "price=%.8f stop=%.8f",
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
            protective_stop = self._protective_stop_price(
                direction=runtime.direction,
                rigid_stop=runtime.rigid_stop_level,
                trailing_stop=runtime.trailing_stop,
            )
            if protective_stop is None:
                continue
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

        confirmer = getattr(self._exchange, "ensure_position_stop_loss", None)
        actual_stop: Optional[float] = None
        if callable(confirmer):
            actual_stop = self._confirm_initial_stop(exchange_position, stop_price)
            ok = actual_stop is not None
        else:
            ok = self._update_stop_loss_on_exchange(exchange_position, stop_price)
            actual_stop = stop_price if ok else None
        if ok:
            # The venue may round to a tick.  Keep the read-back stop locally
            # so local risk/exit logic never relies on a requested-only value.
            record.stop_loss = actual_stop
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
        runtime = self._position_runtime_by_symbol.get(position.symbol)
        exit_mode = runtime.exit_mode if runtime is not None else self._cfg.exit_mode
        self._log.info(
            "EmaAvwapPullback: EXIT EXECUTION requested exit_mode=%s symbol=%s "
            "side=%s reason=%s",
            exit_mode.value,
            position.symbol,
            position.side.value,
            reason,
        )
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
        self._notify_trade_closed(
            position,
            reason=reason,
            exit_price=exit_price,
            runtime=runtime,
        )
        self._save_position_to_db(position)
        self._save_state()
        self._log.info(
            "EmaAvwapPullback: EXIT EXECUTION filled exit_mode=%s symbol=%s "
            "side=%s exit=%s qty=%.8f pnl=%s reason=%s",
            exit_mode.value,
            position.symbol,
            position.side.value,
            f"{exit_price:.8f}" if exit_price is not None else "unavailable",
            position.quantity,
            f"{position.pnl:.8f}" if position.pnl is not None else "unavailable",
            reason,
        )

    # ------------------------------------------------------------------
    # Stale pending entries
    # ------------------------------------------------------------------

    def _cancel_stale_entries(self, snapshot: _SymbolSnapshot, now: datetime) -> None:
        tf_seconds = max(self._timeframe_seconds(self._cfg.timeframe), 1)
        for key, pending in list(self._state.pending_entries.items()):
            if pending.symbol != snapshot.symbol:
                continue
            if pending.status == "ERROR" and not pending.order_id:
                # A legacy ambiguous submission without a recoverable exchange
                # identifier must remain visible for manual reconciliation.
                continue
            reference = snapshot.candle.close_time
            bars_since = int(
                max(0.0, (reference - pending.signal_time).total_seconds())
                // tf_seconds
            )
            if bars_since >= self._cfg.entry_cancel_bars:
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
            status = self._reconcile_pending_order(pending)
            if self._order_is_terminal(status):
                self._remove_pending_entry(pending)
                self._save_state()
                return
            if not cancelled:
                self._log.warning(
                    "EmaAvwapPullback: cancel by order id failed for %s order=%s "
                    "(%s); retaining it for reconciliation and avoiding cancel_all_orders "
                    "so protective stops cannot be removed",
                    pending.symbol,
                    pending.order_id,
                    reason,
                )
                pending.status = "PLACED"
                pending.notes = f"{pending.notes}; cancel failed: {reason}".strip("; ")
                self._save_state()
                return
            # Even a successful cancel acknowledgement can race the order
            # detail endpoint.  Do not forget a GTC order until its terminal
            # state can be observed in a later reconciliation pass.
            pending.status = "PLACED"
            pending.notes = f"{pending.notes}; cancel submitted: {reason}".strip("; ")
            self._save_state()
            return
        pending.status = "CANCELLED"
        pending.notes = f"{pending.notes}; {reason}".strip("; ")
        self._remove_pending_entry(pending)
        self._save_state()

    # ------------------------------------------------------------------
    # Backtest-aligned math helpers
