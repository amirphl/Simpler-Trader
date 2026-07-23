"""Setup detection and entry queuing for EMA + AVWAP."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Optional
from uuid import uuid4

from candle_downloader.models import Candle

from ..exchange import OrderResult
from ..models import PendingEntryRecord
from .config import Direction
from ._mixin_typing import EmaAvwapMixinTyping
from .state import (
    _AvwapSnapshot,
    _CrossDecision,
    _EntryCandidate,
    _InsufficientBalanceError,
    _PendingEntryMeta,
    _SetupState,
    _SymbolSnapshot,
)


class EmaAvwapSignalMixin(EmaAvwapMixinTyping):
    def _process_signal_state(self, snapshot: _SymbolSnapshot, now: datetime) -> None:
        symbol = snapshot.symbol
        if self._state.is_symbol_disabled(symbol, now, self._cfg.disable_symbol_hours):
            return
        if symbol in self._state.active_positions or self._has_pending_for_symbol(symbol):
            return

        queued = self._process_pending_setup("long", snapshot, now)
        if queued:
            self._clear_setups_for_symbol(symbol)
            return

        queued = self._process_pending_setup("short", snapshot, now)
        if queued:
            self._clear_setups_for_symbol(symbol)
            return

        maybe_long = self._detect_setup("long", snapshot)
        if maybe_long is not None:
            self._replace_or_store_setup(maybe_long, snapshot)

        maybe_short = self._detect_setup("short", snapshot)
        if maybe_short is not None:
            self._replace_or_store_setup(maybe_short, snapshot)

    def _detect_setup(
        self, direction: Direction, snapshot: _SymbolSnapshot
    ) -> _SetupState | None:
        anchor_index = snapshot.candle_index - self._cfg.consecutive_count + 1
        if anchor_index < 0:
            return None
        window = snapshot.candles[anchor_index : snapshot.candle_index + 1]
        if direction == "long":
            if not all(candle.is_bullish() for candle in window):
                return None
        else:
            if not all(candle.is_bearish() for candle in window):
                return None
        if not self._validate_ema_position(
            candle=snapshot.candle,
            ema_value=snapshot.ema_value,
            direction=direction,
        ):
            return None
        try:
            detected_avwap = self._build_avwap_snapshot(
                candles=snapshot.candles,
                anchor_index=anchor_index,
                candle_index=snapshot.candle_index,
                tpv_prefix=snapshot.tpv_prefix,
                vol_prefix=snapshot.vol_prefix,
                tpv2_prefix=snapshot.tpv2_prefix,
            )
        except Exception as exc:
            self._log.warning(
                "EmaAvwapPullback: skipped %s setup for %s because detected "
                "AVWAP could not be built: %s",
                direction,
                snapshot.symbol,
                exc,
            )
            return None
        return _SetupState(
            symbol=snapshot.symbol,
            direction=direction,
            anchor_time=snapshot.candles[anchor_index].open_time,
            detected_time=snapshot.candle.close_time,
            consecutive_count=self._cfg.consecutive_count,
            detected_avwap=detected_avwap,
        )

    def _validate_ema_position(
        self, *, candle: Candle, ema_value: float, direction: Direction
    ) -> bool:
        if direction == "long":
            if self._cfg.ema_validation_mode == "wick":
                return candle.low > ema_value
            return min(candle.open, candle.close) > ema_value
        if self._cfg.ema_validation_mode == "wick":
            return candle.high < ema_value
        return max(candle.open, candle.close) < ema_value

    def _replace_or_store_setup(
        self, new_setup: _SetupState, snapshot: _SymbolSnapshot
    ) -> _SetupState:
        key = self._setup_key(new_setup.symbol, new_setup.direction)
        current = self._active_setups.get(key)
        if current is not None and current.is_waiting_for_cross:
            if self._cfg.setup_waiting_replacement_mode == "keep_waiting":
                if current.detected_avwap is None and snapshot is not None:
                    recovered = self._recover_setup_detected_avwap(current, snapshot)
                    if recovered is not current:
                        self._active_setups[key] = recovered
                        self._save_state()
                        return recovered
                self._log.info(
                    "EmaAvwapPullback: keeping waiting %s setup for %s "
                    "(anchor=%s), ignoring newer anchor=%s",
                    new_setup.direction,
                    new_setup.symbol,
                    current.anchor_time.isoformat(),
                    new_setup.anchor_time.isoformat(),
                )
                return current
            self._log.info(
                "EmaAvwapPullback: replacing waiting %s setup for %s "
                "(old anchor=%s, new anchor=%s)",
                new_setup.direction,
                new_setup.symbol,
                current.anchor_time.isoformat(),
                new_setup.anchor_time.isoformat(),
            )
        elif current is not None:
            self._log.info(
                "EmaAvwapPullback: replacing active %s setup for %s "
                "(old anchor=%s, new anchor=%s)",
                new_setup.direction,
                new_setup.symbol,
                current.anchor_time.isoformat(),
                new_setup.anchor_time.isoformat(),
            )

        self._last_price_by_setup_key.pop(key, None)
        self._last_middle_by_setup_key.pop(key, None)
        self._active_setups[key] = new_setup
        self._save_state()
        try:
            avwap = new_setup.detected_avwap
            if avwap is None:
                anchor_index = self._find_anchor_index(snapshot.candles, new_setup)
                avwap = self._build_avwap_snapshot(
                    candles=snapshot.candles,
                    anchor_index=anchor_index,
                    candle_index=snapshot.candle_index,
                    tpv_prefix=snapshot.tpv_prefix,
                    vol_prefix=snapshot.vol_prefix,
                    tpv2_prefix=snapshot.tpv2_prefix,
                )
            self._log_avwap_levels(
                context="detected_setup",
                setup=new_setup,
                snapshot=snapshot,
                avwap=avwap,
            )
            self._log.info(
                "EmaAvwapPullback: detected %s setup for %s anchor=%s vwap=%.8f",
                new_setup.direction,
                new_setup.symbol,
                new_setup.anchor_time.isoformat(),
                avwap.vwap,
            )
        except Exception:
            self._log.info(
                "EmaAvwapPullback: detected %s setup for %s anchor=%s",
                new_setup.direction,
                new_setup.symbol,
                new_setup.anchor_time.isoformat(),
        )
        return new_setup

    def _recover_setup_detected_avwap(
        self, setup: _SetupState, snapshot: _SymbolSnapshot
    ) -> _SetupState:
        if setup.detected_avwap is not None:
            return setup
        try:
            anchor_index = self._find_anchor_index(snapshot.candles, setup)
            detected_index = next(
                idx
                for idx, candle in enumerate(snapshot.candles)
                if candle.close_time == setup.detected_time
            )
            avwap = self._build_avwap_snapshot(
                candles=snapshot.candles,
                anchor_index=anchor_index,
                candle_index=detected_index,
                tpv_prefix=snapshot.tpv_prefix,
                vol_prefix=snapshot.vol_prefix,
                tpv2_prefix=snapshot.tpv2_prefix,
            )
        except Exception as exc:
            self._log.warning(
                "EmaAvwapPullback: could not recover frozen %s setup AVWAP "
                "for %s: %s",
                setup.direction,
                setup.symbol,
                exc,
            )
            return setup
        recovered = replace(setup, detected_avwap=avwap)
        self._active_setups[self._setup_key(setup.symbol, setup.direction)] = recovered
        self._save_state()
        self._log.info(
            "EmaAvwapPullback: recovered frozen %s setup AVWAP for %s "
            "anchor=%s detected=%s vwap=%.8f",
            setup.direction,
            setup.symbol,
            setup.anchor_time.isoformat(),
            setup.detected_time.isoformat(),
            avwap.vwap,
        )
        return recovered

    def _process_pending_setup(
        self, direction: Direction, snapshot: _SymbolSnapshot, now: datetime
    ) -> bool:
        setup = self._active_setups.get(self._setup_key(snapshot.symbol, direction))
        if setup is None:
            return False
        if snapshot.candle.close_time <= setup.detected_time:
            return False

        if self._cfg.entry_exit_mode.uses_closed_candle_entry:
            return self._process_closed_candle_setup(setup, snapshot, now)

        self._active_setups[self._setup_key(setup.symbol, setup.direction)] = (
            replace(setup, is_waiting_for_cross=True)
        )
        self._save_state()
        self._log.info(
            "EmaAvwapPullback: entry evaluation mode=%s symbol=%s "
            "direction=%s trigger=live_middle_touch close=%.8f ema=%.8f "
            "result=waiting_for_live_tick",
            self._cfg.entry_exit_mode.value,
            setup.symbol,
            setup.direction,
            snapshot.candle.close,
            snapshot.ema_value,
        )
        return False

    def _process_closed_candle_setup(
        self, setup: _SetupState, snapshot: _SymbolSnapshot, now: datetime
    ) -> bool:
        """Evaluate mode-2 entry using only the candle that actually closed."""
        try:
            anchor_index = self._find_anchor_index(snapshot.candles, setup)
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
                "EmaAvwapPullback: mode=%s cannot build closed-candle "
                "indicators for %s %s: %s",
                self._cfg.entry_exit_mode.value,
                setup.direction,
                setup.symbol,
                exc,
            )
            return False

        self._log_avwap_levels(
            context="closed_candle_entry_evaluation",
            setup=setup,
            snapshot=snapshot,
            avwap=avwap,
        )

        expects_pullback = (
            snapshot.candle.is_bearish()
            if setup.direction == "long"
            else snapshot.candle.is_bullish()
        )
        if not expects_pullback:
            self._log.info(
                "EmaAvwapPullback: entry evaluation mode=%s symbol=%s "
                "setup_direction=%s candle_open=%.8f candle_close=%.8f "
                "closed_ema=%.8f closed_middle=%.8f result=waiting_for_pullback",
                self._cfg.entry_exit_mode.value,
                setup.symbol,
                setup.direction,
                snapshot.candle.open,
                snapshot.candle.close,
                snapshot.ema_value,
                avwap.vwap,
            )
            return False

        closed_price = snapshot.candle.close
        crossed_middle = (
            closed_price < avwap.vwap
            if setup.direction == "long"
            else closed_price > avwap.vwap
        )
        self._log.info(
            "EmaAvwapPullback: entry evaluation mode=%s symbol=%s "
            "setup_direction=%s candle_open=%.8f closed_price=%.8f "
            "closed_ema=%.8f closed_middle=%.8f result=%s",
            self._cfg.entry_exit_mode.value,
            setup.symbol,
            setup.direction,
            snapshot.candle.open,
            closed_price,
            snapshot.ema_value,
            avwap.vwap,
            "entry" if crossed_middle else "waiting_for_middle_cross",
        )
        if not crossed_middle:
            self._active_setups[self._setup_key(setup.symbol, setup.direction)] = (
                replace(setup, is_waiting_for_cross=True)
            )
            self._save_state()
            return False
        current_price = self._safe_fetch_price(snapshot.symbol)
        if current_price is None or current_price <= 0:
            self._log.warning(
                "EmaAvwapPullback: mode=%s entry skipped for %s because live "
                "execution price is unavailable",
                self._cfg.entry_exit_mode.value,
                snapshot.symbol,
            )
            return False
        candidate = self._build_entry_candidate(
            setup=setup,
            snapshot=snapshot,
            avwap=avwap,
            cross=_CrossDecision(True, "candle_close"),
            signal_time=snapshot.candle.close_time,
            current_price=current_price,
            decision_price=closed_price,
        )
        if candidate is None:
            return False
        return self._queue_entry_candidate(candidate, now)

    def _log_avwap_levels(
        self,
        *,
        context: str,
        setup: _SetupState,
        snapshot: _SymbolSnapshot,
        avwap: _AvwapSnapshot,
    ) -> None:
        self._log.info(
            "EmaAvwapPullback: AVWAP levels mode=%s context=%s symbol=%s direction=%s "
            "anchor=%s candle_close=%s vwap=%.8f stdev=%.8f "
            "lower1=%.8f upper1=%.8f lower2=%.8f upper2=%.8f "
            "lower3=%.8f upper3=%.8f",
            self._cfg.entry_exit_mode.value,
            context,
            setup.symbol,
            setup.direction,
            avwap.anchor_time.isoformat(),
            snapshot.candle.close_time.isoformat(),
            avwap.vwap,
            avwap.stdev,
            avwap.lower1,
            avwap.upper1,
            avwap.lower2,
            avwap.upper2,
            avwap.lower3,
            avwap.upper3,
        )

    def _process_live_setup_crosses(self, now: datetime) -> None:
        if self._cfg.entry_exit_mode.uses_closed_candle_entry:
            return
        for key, setup in list(self._active_setups.items()):
            symbol = setup.symbol
            if self._state.is_symbol_disabled(
                symbol, now, self._cfg.disable_symbol_hours
            ):
                continue
            if symbol in self._state.active_positions or self._has_pending_for_symbol(
                symbol
            ):
                continue
            snapshot = self._last_snapshot_by_symbol.get(symbol)
            if snapshot is None:
                continue

            try:
                live_snapshot, avwap = self._build_live_avwap_snapshot(snapshot, setup)
            except Exception as exc:
                self._log.warning(
                    "EmaAvwapPullback: mode=%s live indicator refresh failed "
                    "for %s %s: %s",
                    self._cfg.entry_exit_mode.value,
                    setup.direction,
                    setup.symbol,
                    exc,
                )
                continue

            current_price = self._safe_fetch_price(symbol)
            if current_price is None or current_price <= 0:
                continue

            last_price = self._last_price_by_setup_key.get(key)
            last_middle = self._last_middle_by_setup_key.get(key)
            first_observation = last_price is None or last_middle is None
            if first_observation:
                crossed = self._price_is_past_entry_line(
                    direction=setup.direction,
                    price=current_price,
                    entry_price=avwap.vwap,
                )
                if crossed:
                    self._log.info(
                        "EmaAvwapPullback: triggering %s setup for %s because live "
                        "price is at or past the AVWAP entry line at first "
                        "observation (current=%.8f entry=%.8f)",
                        setup.direction,
                        symbol,
                        current_price,
                        avwap.vwap,
                    )
                else:
                    self._active_setups[key] = replace(
                        setup, is_waiting_for_cross=True
                    )
                    self._last_price_by_setup_key[key] = current_price
                    self._last_middle_by_setup_key[key] = avwap.vwap
                    self._save_state()
                    continue
            else:
                assert last_price is not None and last_middle is not None
                crossed = self._price_crossed_entry_line(
                    direction=setup.direction,
                    previous_price=last_price,
                    previous_entry_price=last_middle,
                    current_price=current_price,
                    current_entry_price=avwap.vwap,
                )

            if not crossed:
                self._last_price_by_setup_key[key] = current_price
                self._last_middle_by_setup_key[key] = avwap.vwap
                self._save_state()
                continue

            self._log.info(
                "EmaAvwapPullback: ENTRY SIGNAL mode=%s symbol=%s direction=%s "
                "reason=live_price_touched_avwap_middle previous=%.8f live=%.8f "
                "ema=%.8f middle=%.8f upper1=%.8f lower1=%.8f "
                "upper2=%.8f lower2=%.8f forming_close=%s",
                self._cfg.entry_exit_mode.value,
                symbol,
                setup.direction,
                current_price if last_price is None else last_price,
                current_price,
                live_snapshot.ema_value,
                avwap.vwap,
                avwap.upper1,
                avwap.lower1,
                avwap.upper2,
                avwap.lower2,
                live_snapshot.candle.close_time.isoformat(),
            )

            candidate = self._build_entry_candidate(
                setup=setup,
                snapshot=live_snapshot,
                avwap=avwap,
                cross=_CrossDecision(True, "live_tick"),
                signal_time=now,
                current_price=current_price,
            )
            if candidate is None:
                self._log.warning(
                    "EmaAvwapPullback: preserving %s setup for %s after the live "
                    "middle trigger because an entry candidate could not be built",
                    setup.direction,
                    setup.symbol,
                )
                continue
            if self._queue_entry_candidate(candidate, now):
                self._clear_setups_for_symbol(symbol)

    def _price_is_past_entry_line(
        self, *, direction: Direction, price: float, entry_price: float
    ) -> bool:
        if direction == "long":
            return price <= entry_price
        return price >= entry_price

    def _price_crossed_entry_line(
        self,
        *,
        direction: Direction,
        previous_price: float,
        previous_entry_price: float,
        current_price: float,
        current_entry_price: float,
    ) -> bool:
        previous_distance = previous_price - previous_entry_price
        current_distance = current_price - current_entry_price
        if direction == "long":
            return previous_distance > 0 and current_distance <= 0
        return previous_distance < 0 and current_distance >= 0

    def _build_entry_candidate(
        self,
        *,
        setup: _SetupState,
        snapshot: _SymbolSnapshot,
        avwap: _AvwapSnapshot,
        cross: _CrossDecision,
        signal_time: datetime | None = None,
        current_price: float | None = None,
        decision_price: float | None = None,
    ) -> _EntryCandidate | None:
        if current_price is None:
            current_price = self._safe_fetch_price(snapshot.symbol)
        if current_price is None or current_price <= 0:
            self._log.warning(
                "EmaAvwapPullback: mode=%s entry skipped for %s because current "
                "price is unavailable",
                self._cfg.entry_exit_mode.value,
                snapshot.symbol,
            )
            return None

        risk_amount = self._compute_risk_amount(snapshot.symbol)
        if risk_amount <= 0:
            self._log.warning(
                "EmaAvwapPullback: entry skipped for %s %s due to non-positive "
                "risk amount",
                snapshot.symbol,
                setup.direction,
            )
            return None

        raw_entry_price = (
            current_price
            if self._cfg.entry_exit_mode.uses_closed_candle_entry
            else avwap.vwap
        )
        rigid_stop_level = self._rigid_stop_level(
            setup.direction, raw_entry_price
        )
        sizing = self._build_sizing_decision(
            direction=setup.direction,
            raw_entry_price=raw_entry_price,
            stop_level=rigid_stop_level,
            risk_amount=risk_amount,
        )
        if sizing is None or sizing.qty <= 0:
            self._log.warning(
                "EmaAvwapPullback: mode=%s entry skipped for %s %s because "
                "rigid-stop sizing is unavailable (entry=%.8f rigid_stop=%s "
                "sizing_mode=%s qty=%s)",
                self._cfg.entry_exit_mode.value,
                snapshot.symbol,
                setup.direction,
                raw_entry_price,
                f"{rigid_stop_level:.8f}" if rigid_stop_level is not None else "disabled",
                self._cfg.position_sizing_mode,
                sizing.qty if sizing is not None else "n/a",
            )
            return None

        qty = sizing.qty
        entry_notional = raw_entry_price * qty
        if entry_notional > self._cfg.max_entry_notional_usdt:
            qty = self._cfg.max_entry_notional_usdt / raw_entry_price
            if qty <= 0:
                return None
            self._log.info(
                "EmaAvwapPullback: clamped %s %s notional from %.8f to %.8f USDT",
                snapshot.symbol,
                setup.direction,
                entry_notional,
                self._cfg.max_entry_notional_usdt,
            )

        if (
            not self._cfg.entry_exit_mode.uses_closed_candle_entry
            and not self._entry_price_is_marketable(
                direction=setup.direction,
                current_price=current_price,
                entry_price=raw_entry_price,
            )
        ):
            self._log.warning(
                "EmaAvwapPullback: entry skipped for %s %s because the live "
                "AVWAP middle order is no longer marketable "
                "(current=%.8f entry=%.8f)",
                snapshot.symbol,
                setup.direction,
                current_price,
                raw_entry_price,
            )
            return None

        rigid_stop_level = self._rigid_stop_level(
            setup.direction, sizing.entry_price
        )
        if rigid_stop_level is not None and self._is_stop_breached_by_price(
            direction=setup.direction,
            price=current_price,
            stop_price=rigid_stop_level,
        ):
            self._log.warning(
                "EmaAvwapPullback: entry skipped for %s %s because price %.8f is "
                "already beyond protective stop %.8f",
                snapshot.symbol,
                setup.direction,
                current_price,
                rigid_stop_level,
            )
            return None

        side = self._side_from_direction(setup.direction)
        target_level = self._target_band_level(setup.direction, avwap)
        self._log.info(
            "EmaAvwapPullback: ENTRY SIGNAL mode=%s symbol=%s side=%s "
            "trigger=%s live_price=%.8f order_price=%.8f ema=%.8f "
            "middle=%.8f target_band=%d target=%.8f rigid_stop=%s "
            "upper1=%.8f lower1=%.8f upper2=%.8f lower2=%.8f",
            self._cfg.entry_exit_mode.value,
            snapshot.symbol,
            side.value,
            cross.mode or "intrabar",
            current_price,
            raw_entry_price,
            snapshot.ema_value,
            avwap.vwap,
            self._target_band_number(),
            target_level,
            f"{rigid_stop_level:.8f}" if rigid_stop_level is not None else "disabled",
            avwap.upper1,
            avwap.lower1,
            avwap.upper2,
            avwap.lower2,
        )
        return _EntryCandidate(
            symbol=snapshot.symbol,
            side=side,
            direction=setup.direction,
            signal_time=signal_time or snapshot.candle.close_time,
            anchor_time=setup.anchor_time,
            setup_detected_time=setup.detected_time,
            candle_index=snapshot.candle_index,
            raw_entry_price=raw_entry_price,
            order_price=raw_entry_price,
            stop_for_risk=rigid_stop_level or 0.0,
            dynamic_stop_at_entry=(
                avwap.lower1 if setup.direction == "long" else avwap.upper1
            ),
            rigid_stop_at_entry=rigid_stop_level,
            trailing_activation_at_entry=self._trailing_activation_level(
                setup.direction, avwap
            ),
            quantity=qty,
            risk_amount=risk_amount,
            risk_amount_interpretation=sizing.risk_amount_interpretation,
            entry_trigger_mode=cross.mode or "intrabar",
            sizing=sizing,
            avwap=avwap,
            entry_exit_mode=self._cfg.entry_exit_mode,
            ema_value=snapshot.ema_value,
            decision_price=(
                decision_price if decision_price is not None else current_price
            ),
        )

    def _entry_price_is_marketable(
        self, *, direction: Direction, current_price: float, entry_price: float
    ) -> bool:
        if direction == "long":
            return current_price <= entry_price
        return current_price >= entry_price

    # ------------------------------------------------------------------
    # Entry queue / activation
    # ------------------------------------------------------------------

    def _queue_entry_candidate(
        self, candidate: _EntryCandidate, now: datetime
    ) -> bool:
        symbol = candidate.symbol
        if symbol in self._state.active_positions:
            self._log.warning(
                "EmaAvwapPullback: rejected %s %s entry because symbol already "
                "has an active position",
                symbol,
                candidate.side.value,
            )
            return False
        if self._has_pending_for_symbol(symbol):
            self._log.warning(
                "EmaAvwapPullback: rejected %s %s entry because symbol already "
                "has a pending entry",
                symbol,
                candidate.side.value,
            )
            return False
        if len(self._state.active_positions) >= self._cfg.max_concurrent_positions:
            self._log.warning(
                "EmaAvwapPullback: rejected %s %s entry because max concurrent "
                "positions is reached (%d)",
                symbol,
                candidate.side.value,
                self._cfg.max_concurrent_positions,
            )
            return False

        key = self._pending_key(symbol, candidate.side)
        pending = PendingEntryRecord(
            order_key=key,
            symbol=symbol,
            side=candidate.side,
            entry_price=candidate.order_price,
            quantity=candidate.quantity,
            leverage=self._cfg.leverage,
            margin_mode=self._cfg.margin_mode,
            risk_amount=candidate.risk_amount,
            stop_for_risk=candidate.stop_for_risk,
            created_time=now,
            signal_time=self._ensure_aware(candidate.signal_time),
            activate_time=now,
            order_id=None,
            status="PENDING",
            notes=(
                f"EMA+AVWAP {candidate.direction} "
                f"mode={candidate.entry_exit_mode.value} "
                f"anchor={candidate.anchor_time.isoformat()} "
                f"trigger={candidate.entry_trigger_mode}"
            ),
            # Keep this value stable through retries and restarts.  Bitunix uses
            # ``clientId`` to identify an order independently of its server id.
            client_id=f"emaavwap-{uuid4().hex}",
        )
        self._state.pending_entries[key] = pending
        self._pending_meta_by_key[key] = _PendingEntryMeta(candidate=candidate)
        self._state.last_pinbar_signal_times[key] = pending.signal_time
        self._save_state()
        self._log.info(
            "EmaAvwapPullback: ENTRY EXECUTION queued mode=%s %s %s @ %.8f "
            "qty=%.8f rigid_stop=%s risk=%.8f trigger=%s",
            candidate.entry_exit_mode.value,
            candidate.side.value,
            symbol,
            candidate.order_price,
            candidate.quantity,
            f"{candidate.stop_for_risk:.8f}" if candidate.stop_for_risk > 0 else "disabled",
            candidate.risk_amount,
            candidate.entry_trigger_mode,
        )
        self._notify_entry_signal(candidate)
        self._activate_due_entries(now)
        return True

    def _activate_due_entries(self, now: datetime) -> None:
        for pending in list(self._state.pending_entries.values()):
            if pending.status not in {"PENDING"}:
                continue
            if now < pending.activate_time:
                continue
            if not pending.client_id and not pending.order_id:
                # A state created before client-id persistence cannot be safely
                # retried: its last POST may have reached the venue with no
                # identifier we can reconcile.  Leave it visible for manual
                # review rather than risk a duplicate entry.
                pending.status = "ERROR"
                pending.notes = (
                    f"{pending.notes}; manual review required: missing client id "
                    "for an ambiguous legacy submission"
                ).strip("; ")
                self._log.critical(
                    "EmaAvwapPullback: refusing to retry legacy pending entry %s "
                    "without an order id or client id",
                    pending.order_key,
                )
                self._save_state()
                continue
            # Recover an order accepted before a timeout/crash by its stable
            # client id.  Never submit another entry while the venue can name
            # the original order.
            recovered_status = self._reconcile_pending_order(pending)
            if pending.order_id:
                if recovered_status in {"CANCELLED", "CANCELED", "REJECTED", "EXPIRED"}:
                    self._remove_pending_entry(pending)
                    self._save_state()
                    continue
                pending.status = "PLACED"
                self._save_state()
                self._log.info(
                    "EmaAvwapPullback: recovered pending entry %s as exchange "
                    "order %s status=%s",
                    pending.order_key,
                    pending.order_id,
                    recovered_status or "unknown",
                )
                continue
            try:
                order = self._place_limit_entry(pending)
            except _InsufficientBalanceError as exc:
                pending.status = "PENDING"
                pending.notes = f"{pending.notes}; insufficient balance: {exc}".strip(
                    "; "
                )
                self._log.warning(
                    "EmaAvwapPullback: insufficient balance for %s %s entry; "
                    "keeping pending until stale: %s",
                    pending.symbol,
                    pending.side.value,
                    exc,
                )
                self._save_state()
                continue
            if order is None:
                # A POST can have reached the exchange even when the response
                # was lost.  Reconcile before allowing any future attempt; the
                # same client id also makes the exchange-side retry idempotent.
                recovered_status = self._reconcile_pending_order(pending)
                if pending.order_id:
                    pending.status = "PLACED"
                    self._save_state()
                    self._log.warning(
                        "EmaAvwapPullback: placement response was ambiguous for %s; "
                        "recovered exchange order %s status=%s",
                        pending.order_key,
                        pending.order_id,
                        recovered_status or "unknown",
                    )
                    continue
                pending.status = "PENDING"
                pending.notes = f"{pending.notes}; last placement attempt failed".strip(
                    "; "
                )
                self._save_state()
                continue
            pending.order_id = order.order_id
            pending.status = "PLACED"
            self._save_state()
            self._log.info(
                "EmaAvwapPullback: ENTRY EXECUTION placed mode=%s order=%s "
                "symbol=%s side=%s price=%.8f qty=%.8f rigid_stop=%s",
                self._pending_meta_by_key.get(
                    pending.order_key
                ).candidate.entry_exit_mode.value
                if self._pending_meta_by_key.get(pending.order_key)
                else self._cfg.entry_exit_mode.value,
                order.order_id,
                pending.symbol,
                pending.side.value,
                pending.entry_price,
                pending.quantity,
                f"{pending.stop_for_risk:.8f}" if pending.stop_for_risk > 0 else "disabled",
            )

    def _place_limit_entry(self, pending: PendingEntryRecord) -> Optional[OrderResult]:
        try:
            validator = getattr(self._exchange, "validate_ema_avwap_execution", None)
            if callable(validator):
                # Recheck immediately before a state-changing order in case the
                # account mode changed after startup.
                validator()
            self._retry(
                lambda: self._exchange.set_margin_mode(
                    pending.symbol, pending.margin_mode
                ),
                f"set_margin_mode {pending.symbol}",
            )
            self._retry(
                lambda: self._exchange.set_leverage(pending.symbol, pending.leverage),
                f"set_leverage {pending.symbol}",
            )
        except Exception as exc:
            if self._is_insufficient_balance_error(exc):
                raise _InsufficientBalanceError(str(exc)) from exc
            self._log.error(
                "EmaAvwapPullback: failed to set account config for %s before "
                "entry (mode=%s leverage=%sx): %s",
                pending.symbol,
                pending.margin_mode.value,
                pending.leverage,
                exc,
            )
            return None

        try:
            # Do not wrap a state-changing POST in the generic retry helper.
            # Bitunix receives the persisted client id and may retry its own
            # request safely; an ambiguous result is reconciled above.
            return self._exchange.open_limit_position(
                symbol=pending.symbol,
                side=pending.side,
                quantity=pending.quantity,
                price=pending.entry_price,
                leverage=pending.leverage,
                margin_mode=pending.margin_mode,
                take_profit=None,
                stop_loss=(
                    pending.stop_for_risk if pending.stop_for_risk > 0 else None
                ),
                client_id=pending.client_id,
            )
        except Exception as exc:
            if self._is_insufficient_balance_error(exc):
                raise _InsufficientBalanceError(str(exc)) from exc
            self._log.error(
                "EmaAvwapPullback: open_limit_position failed for %s %s: %s",
                pending.symbol,
                pending.side.value,
                exc,
            )
