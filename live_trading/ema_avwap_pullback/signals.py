"""Setup detection and entry queuing for EMA + AVWAP."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Literal, Optional

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

        setup = self._recover_setup_detected_avwap(setup, snapshot)
        avwap = setup.detected_avwap
        if avwap is None:
            self._log.warning(
                "EmaAvwapPullback: invalidating %s setup for %s because frozen "
                "AVWAP is unavailable",
                setup.direction,
                setup.symbol,
            )
            self._remove_setup(setup.symbol, setup.direction)
            return False

        self._log_avwap_levels(
            context="active_setup_update",
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
            return False

        cross_direction: Literal["up", "down"] = (
            "down" if setup.direction == "long" else "up"
        )
        cross = self._detect_level_cross(
            candle=snapshot.candle,
            prev_close=snapshot.previous_candle.close,
            level=avwap.vwap,
            direction=cross_direction,
        )
        if not cross.crossed:
            self._active_setups[self._setup_key(setup.symbol, setup.direction)] = (
                replace(setup, is_waiting_for_cross=True)
            )
            self._save_state()
            self._log.info(
                "EmaAvwapPullback: %s setup for %s is waiting for AVWAP cross "
                "(vwap=%.8f)",
                setup.direction,
                setup.symbol,
                avwap.vwap,
            )
            return False

        self._log.info(
            "EmaAvwapPullback: skipping %s setup for %s because AVWAP was crossed "
            "on an already-closed candle (vwap=%.8f); waiting for next setup",
            setup.direction,
            setup.symbol,
            avwap.vwap,
        )
        self._remove_setup(setup.symbol, setup.direction)
        return False

    def _log_avwap_levels(
        self,
        *,
        context: str,
        setup: _SetupState,
        snapshot: _SymbolSnapshot,
        avwap: _AvwapSnapshot,
    ) -> None:
        self._log.info(
            "EmaAvwapPullback: AVWAP levels context=%s symbol=%s direction=%s "
            "anchor=%s candle_close=%s vwap=%.8f stdev=%.8f "
            "lower1=%.8f upper1=%.8f lower2=%.8f upper2=%.8f "
            "lower3=%.8f upper3=%.8f",
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

            setup = self._recover_setup_detected_avwap(setup, snapshot)
            avwap = setup.detected_avwap
            if avwap is None:
                self._log.warning(
                    "EmaAvwapPullback: invalidating %s setup for %s because frozen "
                    "AVWAP is unavailable for live cross tracking",
                    setup.direction,
                    setup.symbol,
                )
                self._remove_setup(setup.symbol, setup.direction)
                continue
            live_snapshot = snapshot

            current_price = self._safe_fetch_price(symbol)
            if current_price is None or current_price <= 0:
                continue

            last_price = self._last_price_by_setup_key.get(key)
            if last_price is None:
                if self._price_is_past_entry_line(
                    direction=setup.direction,
                    price=current_price,
                    entry_price=avwap.vwap,
                ):
                    self._log.info(
                        "EmaAvwapPullback: skipping %s setup for %s because live "
                        "price is already past the AVWAP entry line at first "
                        "observation (current=%.8f entry=%.8f); waiting for next setup",
                        setup.direction,
                        symbol,
                        current_price,
                        avwap.vwap,
                    )
                    self._remove_setup(setup.symbol, setup.direction)
                    continue
                self._active_setups[key] = replace(setup, is_waiting_for_cross=True)
                self._last_price_by_setup_key[key] = current_price
                self._save_state()
                continue

            if not self._price_crossed_entry_line(
                direction=setup.direction,
                previous_price=last_price,
                current_price=current_price,
                entry_price=avwap.vwap,
            ):
                self._last_price_by_setup_key[key] = current_price
                self._save_state()
                continue

            candidate = self._build_entry_candidate(
                setup=setup,
                snapshot=live_snapshot,
                avwap=avwap,
                cross=_CrossDecision(True, "live_tick"),
                signal_time=now,
                current_price=current_price,
            )
            if candidate is None:
                self._remove_setup(setup.symbol, setup.direction)
                continue
            if self._queue_entry_candidate(candidate, now):
                self._clear_setups_for_symbol(symbol)

    def _price_is_past_entry_line(
        self, *, direction: Direction, price: float, entry_price: float
    ) -> bool:
        if direction == "long":
            return price < entry_price
        return price > entry_price

    def _price_crossed_entry_line(
        self,
        *,
        direction: Direction,
        previous_price: float,
        current_price: float,
        entry_price: float,
    ) -> bool:
        if direction == "long":
            return previous_price > entry_price >= current_price
        return previous_price < entry_price <= current_price

    def _build_entry_candidate(
        self,
        *,
        setup: _SetupState,
        snapshot: _SymbolSnapshot,
        avwap: _AvwapSnapshot,
        cross: _CrossDecision,
        signal_time: datetime | None = None,
        current_price: float | None = None,
    ) -> _EntryCandidate | None:
        risk_amount = self._compute_risk_amount(snapshot.symbol)
        if risk_amount <= 0:
            self._log.warning(
                "EmaAvwapPullback: entry skipped for %s %s due to non-positive "
                "risk amount",
                snapshot.symbol,
                setup.direction,
            )
            return None

        raw_entry_price = avwap.vwap
        stop_level = avwap.lower2 if setup.direction == "long" else avwap.upper2
        sizing = self._build_sizing_decision(
            direction=setup.direction,
            raw_entry_price=raw_entry_price,
            stop_level=stop_level,
            risk_amount=risk_amount,
        )
        if sizing is None or sizing.qty <= 0:
            self._log.warning(
                "EmaAvwapPullback: entry skipped for %s %s due to invalid stop "
                "distance (entry=%.8f stop=%.8f qty=%s)",
                snapshot.symbol,
                setup.direction,
                raw_entry_price,
                stop_level,
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

        if current_price is None:
            current_price = self._safe_fetch_price(snapshot.symbol)
        if current_price is None or current_price <= 0:
            self._log.warning(
                "EmaAvwapPullback: entry skipped for %s because current price is unavailable",
                snapshot.symbol,
            )
            return None
        if not self._entry_price_is_marketable(
            direction=setup.direction,
            current_price=current_price,
            entry_price=raw_entry_price,
        ):
            self._log.warning(
                "EmaAvwapPullback: entry skipped for %s %s because the closed-bar "
                "AVWAP cross is no longer marketable (current=%.8f entry=%.8f)",
                snapshot.symbol,
                setup.direction,
                current_price,
                raw_entry_price,
            )
            return None

        rigid_stop_level = self._rigid_stop_level(setup.direction, sizing.entry_price)
        protective_stop = self._protective_stop_price(
            direction=setup.direction,
            dynamic_stop=stop_level,
            rigid_stop=rigid_stop_level,
            trailing_stop=None,
        )
        if self._is_stop_breached_by_price(
            direction=setup.direction,
            price=current_price,
            stop_price=protective_stop,
        ):
            self._log.warning(
                "EmaAvwapPullback: entry skipped for %s %s because price %.8f is "
                "already beyond protective stop %.8f",
                snapshot.symbol,
                setup.direction,
                current_price,
                protective_stop,
            )
            return None

        side = self._side_from_direction(setup.direction)
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
            stop_for_risk=protective_stop,
            dynamic_stop_at_entry=stop_level,
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
                f"anchor={candidate.anchor_time.isoformat()} "
                f"trigger={candidate.entry_trigger_mode}"
            ),
        )
        self._state.pending_entries[key] = pending
        self._pending_meta_by_key[key] = _PendingEntryMeta(candidate=candidate)
        self._state.last_pinbar_signal_times[key] = pending.signal_time
        self._save_state()
        self._log.info(
            "EmaAvwapPullback: queued limit entry %s %s @ %.8f qty=%.8f "
            "stop=%.8f risk=%.8f",
            candidate.side.value,
            symbol,
            candidate.order_price,
            candidate.quantity,
            candidate.stop_for_risk,
            candidate.risk_amount,
        )
        self._activate_due_entries(now)
        return True

    def _activate_due_entries(self, now: datetime) -> None:
        for pending in list(self._state.pending_entries.values()):
            if pending.status not in {"PENDING"}:
                continue
            if now < pending.activate_time:
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
                "EmaAvwapPullback: placed limit entry order %s for %s %s",
                order.order_id,
                pending.symbol,
                pending.side.value,
            )

    def _place_limit_entry(self, pending: PendingEntryRecord) -> Optional[OrderResult]:
        try:
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
            return self._retry(
                lambda: self._exchange.open_limit_position(
                    symbol=pending.symbol,
                    side=pending.side,
                    quantity=pending.quantity,
                    price=pending.entry_price,
                    leverage=pending.leverage,
                    margin_mode=pending.margin_mode,
                    take_profit=None,
                    stop_loss=pending.stop_for_risk,
                ),
                f"open_limit_position {pending.symbol}",
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
