"""Strong-trend percent-based stair-trailing live coordinator (single symbol)."""

from __future__ import annotations

import logging
import time
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from candle_downloader.models import Candle
from candle_downloader.binance import BinanceClient, BinanceClientConfig, MAX_BATCH
from signal_notifier import TelegramClient

from backtest.scalping_FVG_strategy import ema as calc_ema
from .exchange import Exchange, MarginMode, Position, PositionSide


@dataclass(frozen=True)
class StrongTrendStairConfig:
    symbol: str = "ETHUSDT"
    timeframe: str = "4h"
    tick_interval_seconds: float = 1.0
    leverage: int = 100
    margin_mode: MarginMode = MarginMode.CROSS

    trade_notional_usd: float = 100.0
    hard_stop_loss_pct: float = 5.0
    trail_start_pct: float = 2.0
    trail_offset_pct: float = 1.0

    ema_fast_len: int = 50
    ema_mid_len: int = 100
    ema_slow_len: int = 200
    slope_lookback: int = 10

    st_atr_len: int = 10
    st_factor: float = 3.0

    di_len: int = 14
    adx_smooth: int = 14
    adx_min: float = 20.0
    reverse_on_opposite_signal: bool = False

    klines_limit: int = 320
    entry_limit_offset_bps: float = 0.0
    fill_wait_timeout_seconds: float = 20.0
    fill_wait_poll_seconds: float = 0.4
    api_retries: int = 3
    api_retry_delay_seconds: float = 1.0

    # Order entry mode: False = limit (default), True = market
    use_market_entry: bool = False


class StrongTrendStairCoordinator:
    """Runs the Strong Trend stair trailing strategy in live mode."""

    _LOW_PROFIT_THRESHOLD = 1.0
    _MID_PROFIT_THRESHOLD = 2.0
    _MID_TRAIL_OFFSET_PCT = 0.5
    _HIGH_TRAIL_OFFSET_PCT = 1.0

    def __init__(
        self,
        exchange: Exchange,
        config: Optional[StrongTrendStairConfig] = None,
        telegram_client: Optional[TelegramClient] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize coordinator dependencies and in-memory runtime state."""
        self._exchange = exchange
        self._cfg = config or StrongTrendStairConfig()
        self._telegram = telegram_client
        self._log = logger or logging.getLogger(self.__class__.__name__)
        self._running = False
        self._active_stop: Optional[float] = None
        self._last_stop_sent: Optional[float] = None
        self._stop_order_id: Optional[str] = None
        self._last_seen_position_id: Optional[str] = None
        self._last_seen_position_side: Optional[PositionSide] = None
        self._symbol_meta_cache: Dict[str, Dict[str, Any]] = {}
        self._last_processed_candle_close_time: Optional[datetime] = None
        self._binance_client: Optional[BinanceClient] = None

    def run_forever(self) -> None:
        """Run the strategy loop continuously until ``stop()`` is called."""
        self._running = True
        self._log.info(
            "StrongTrendStair started symbol=%s timeframe=%s leverage=%sx margin_mode=%s",
            self._cfg.symbol,
            self._cfg.timeframe,
            self._cfg.leverage,
            self._cfg.margin_mode.value,
        )
        while self._running:
            try:
                self._tick()
            except Exception as exc:
                self._log.error("StrongTrendStair tick error: %s", exc, exc_info=True)
            time.sleep(max(self._cfg.tick_interval_seconds, 0.2))

    def stop(self) -> None:
        """Request graceful termination of the main loop."""
        self._running = False

    def _tick(self) -> None:
        """Process one closed-candle cycle for signal, entry, and position management."""
        snapshot = self._evaluate_snapshot()
        if snapshot is None:
            self._log.info("StrongTrendStair no data for snapshot; skipping tick")
            return
        candle, signal_side, indicators = snapshot
        if (
            self._last_processed_candle_close_time is not None
            and candle.close_time <= self._last_processed_candle_close_time
        ):
            return
        # Use Binance candles for signals, but Bitunix mark for pricing decisions.
        mark_price = self._safe_fetch_price(self._cfg.symbol)
        if mark_price is None or mark_price <= 0:
            self._log.warning(
                "StrongTrendStair price unavailable from exchange; skipping tick symbol=%s",
                self._cfg.symbol,
            )
            return
        self._log.info(
            "StrongTrendStair tick snapshot symbol=%s close_time=%s signal_side=%s indicators=%s mark=%.8f",
            self._cfg.symbol,
            candle.close_time.isoformat(),
            signal_side.value if signal_side is not None else "n/a",
            indicators if indicators is not None else "n/a",
            mark_price,
        )
        self._last_processed_candle_close_time = candle.close_time

        position, position_known = self._safe_get_position_with_status(self._cfg.symbol)
        if not position_known:
            self._log.warning(
                "StrongTrendStair position state unknown (all get_position retries failed); skipping tick symbol=%s",
                self._cfg.symbol,
            )
            return
        if position is not None:
            self._last_seen_position_id = position.position_id or self._cfg.symbol
            self._last_seen_position_side = position.side
            self._manage_open_position(position, mark_price)
            if self._cfg.reverse_on_opposite_signal:
                self._maybe_reverse_position(
                    position, mark_price, signal_side, candle
                )
            return

        if self._last_seen_position_id is not None:
            self._notify(
                "[STRONG CLOSE] %s\nSide: %s\nReason: position closed on exchange "
                "(likely stop-loss trigger)\nTime: %s"
                % (
                    self._cfg.symbol,
                    self._last_seen_position_side.value
                    if self._last_seen_position_side is not None
                    else "n/a",
                    datetime.now(timezone.utc).isoformat(),
                )
            )
            self._last_seen_position_id = None
            self._last_seen_position_side = None

        # No open position: reset in-memory trailing state.
        self._active_stop = None
        self._last_stop_sent = None
        self._stop_order_id = None
        if signal_side is None or indicators is None:
            return

        side = signal_side
        self._log.info(
            "StrongTrendStair trend signal symbol=%s side=%s "
            "close=%.8f o=%.8f h=%.8f l=%.8f v=%.8f "
            "ema_fast=%.8f ema_mid=%.8f ema_slow=%.8f st_line=%.8f "
            "di_plus=%.8f di_minus=%.8f adx=%.8f",
            self._cfg.symbol,
            side.value,
            candle.close,
            candle.open,
            candle.high,
            candle.low,
            candle.volume,
            indicators[0],
            indicators[1],
            indicators[2],
            indicators[3],
            indicators[4],
            indicators[5],
            indicators[6],
        )
        if not self._has_sufficient_balance_for_entry(mark_price):
            self._log.info(
                "StrongTrendStair skipping entry due to insufficient balance symbol=%s required≈%.8f (notional=%.2f leverage=%s)",
                self._cfg.symbol,
                self._cfg.trade_notional_usd / max(float(self._cfg.leverage), 1.0),
                self._cfg.trade_notional_usd,
                self._cfg.leverage,
            )
            return
        self._open_new_position(side=side, reference_price=mark_price)

    def _maybe_reverse_position(
        self,
        position: Position,
        reference_price: float,
        signal_side: Optional[PositionSide],
        candle: Candle,
    ) -> None:
        """Close and reopen in the opposite direction when reversal mode is enabled."""
        if signal_side is None:
            return
        if signal_side == position.side:
            return

        self._log.info(
            "StrongTrendStair reversal signal symbol=%s from=%s to=%s binance_close=%.8f mark_price=%.8f",
            self._cfg.symbol,
            position.side.value,
            signal_side.value,
            candle.close,
            reference_price,
        )
        self._notify(
            "[STRONG REVERSAL] %s\nFrom: %s\nTo: %s\nSignal Close: %.8f\nMark: %.8f\nTime: %s"
            % (
                self._cfg.symbol,
                position.side.value,
                signal_side.value,
                candle.close,
                reference_price,
                datetime.now(timezone.utc).isoformat(),
            )
        )

        closed = self._safe_close_position(self._cfg.symbol)
        if not closed:
            self._log.warning(
                "StrongTrendStair reversal close failed symbol=%s from=%s to=%s",
                self._cfg.symbol,
                position.side.value,
                signal_side.value,
            )
            return

        if not self._wait_until_flat(
            self._cfg.symbol, timeout_seconds=5.0, poll_seconds=0.2
        ):
            self._log.warning(
                "StrongTrendStair reversal close submitted but position still visible; skip reopen this tick symbol=%s",
                self._cfg.symbol,
            )
            return

        self._last_seen_position_id = None
        self._last_seen_position_side = None
        self._open_new_position(side=signal_side, reference_price=reference_price)

    def _manage_open_position(self, position: Position, last_price: float) -> None:
        """Update trailing stop state and enforce emergency close on hard-stop breach."""
        qty = abs(float(position.size))
        if qty <= 0:
            return
        entry = float(position.entry_price)
        favorable_mark = last_price
        if position.side == PositionSide.LONG:
            open_pnl = (last_price - entry) * qty
        else:
            open_pnl = (entry - last_price) * qty
        hard_stop = self._hard_stop_price(entry, position.side)

        if self._active_stop is None:
            self._active_stop = hard_stop

        candidate = hard_stop
        trailing_candidate = self._trailing_candidate_price(
            entry_price=entry,
            favorable_mark=favorable_mark,
            side=position.side,
            qty=qty,
        )
        if trailing_candidate is not None:
            candidate = trailing_candidate

        if position.side == PositionSide.LONG:
            self._active_stop = max(self._active_stop, candidate)
        else:
            self._active_stop = min(self._active_stop, candidate)
        self._active_stop = self._normalize_price_for_symbol(
            position.symbol, self._active_stop, position.side
        )

        self._log.info(
            "StrongTrendStair position symbol=%s side=%s entry=%.8f price=%.8f qty=%.8f "
            "open_pnl=%.8f hard_stop=%.8f active_stop=%.8f",
            position.symbol,
            position.side.value,
            entry,
            last_price,
            qty,
            open_pnl,
            hard_stop,
            self._active_stop,
        )

        if self._should_send_stop(self._active_stop, position.side):
            ok = self._safe_update_stop_loss(position, self._active_stop)
            self._log.info(
                "StrongTrendStair stop update symbol=%s side=%s new_stop=%.8f sent=%s",
                position.symbol,
                position.side.value,
                self._active_stop,
                ok,
            )
            if ok:
                self._last_stop_sent = self._active_stop

        if self._is_hard_stop_breached(entry, last_price, position.side):
            self._log.warning(
                "StrongTrendStair hard-stop breach symbol=%s price=%.8f entry=%.8f, forcing close",
                position.symbol,
                last_price,
                entry,
            )
            ok = self._safe_close_position(position.symbol)
            self._log.info(
                "StrongTrendStair hard-stop close executed symbol=%s side=%s qty=%.8f sent=%s",
                position.symbol,
                position.side.value,
                qty,
                ok,
            )
            if ok:
                self._last_seen_position_id = position.position_id or self._cfg.symbol
                self._last_seen_position_side = position.side

    def _hard_stop_price(self, entry_price: float, side: PositionSide) -> float:
        """Return absolute hard-stop price derived from entry and configured loss percent."""
        hard_stop_fraction = self._cfg.hard_stop_loss_pct / 100.0
        if side == PositionSide.LONG:
            return entry_price * (1.0 - hard_stop_fraction)
        return entry_price * (1.0 + hard_stop_fraction)

    def _current_return(
        self, entry_price: float, qty: float, mark_price: float, side: PositionSide
    ) -> float:
        """Return multiple on margin (1.0 = 100% return)."""
        leverage = max(float(self._cfg.leverage), 1.0)
        margin = (entry_price * qty) / leverage if leverage > 0 else 0.0
        if margin <= 0:
            return 0.0
        if side == PositionSide.LONG:
            pnl = (mark_price - entry_price) * qty
        else:
            pnl = (entry_price - mark_price) * qty
        return pnl / margin

    def _trailing_candidate_price(
        self, entry_price: float, favorable_mark: float, side: PositionSide, qty: float
    ) -> Optional[float]:
        """Compute trailing stop candidate once activation threshold is reached."""
        if entry_price <= 0:
            return None
        if side == PositionSide.LONG:
            favorable_move_pct = ((favorable_mark - entry_price) / entry_price) * 100.0
        else:
            favorable_move_pct = ((entry_price - favorable_mark) / entry_price) * 100.0
        if favorable_move_pct < self._cfg.trail_start_pct:
            return None
        trail_offset_pct = self._cfg.trail_offset_pct
        ret = self._current_return(entry_price, qty, favorable_mark, side)
        if ret > self._LOW_PROFIT_THRESHOLD:
            if ret <= self._MID_PROFIT_THRESHOLD:
                trail_offset_pct = self._MID_TRAIL_OFFSET_PCT
            else:
                trail_offset_pct = self._HIGH_TRAIL_OFFSET_PCT
        offset_fraction = trail_offset_pct / 100.0
        if side == PositionSide.LONG:
            return favorable_mark * (1.0 - offset_fraction)
        return favorable_mark * (1.0 + offset_fraction)

    def _is_hard_stop_breached(
        self, entry_price: float, mark_price: float, side: PositionSide
    ) -> bool:
        """Check whether current adverse move breaches the configured hard-stop percent."""
        if entry_price <= 0:
            return False
        if side == PositionSide.LONG:
            move_pct = ((mark_price - entry_price) / entry_price) * 100.0
        else:
            move_pct = ((entry_price - mark_price) / entry_price) * 100.0
        return move_pct <= -self._cfg.hard_stop_loss_pct

    def _open_new_position(
        self, side: PositionSide, reference_price: Optional[float]
    ) -> None:
        """Submit protected entry (limit or market) and handle full/partial fill outcomes safely."""
        if reference_price is None or reference_price <= 0:
            return
        qty = self._normalize_quantity_for_symbol(
            self._cfg.symbol,
            self._cfg.trade_notional_usd / reference_price,
            reference_price=reference_price,
        )
        if qty is None or qty <= 0:
            return

        if not self._safe_set_margin_mode(self._cfg.symbol, self._cfg.margin_mode):
            self._log.warning("StrongTrendStair: abort entry, set_margin_mode failed")
            return
        if not self._safe_set_leverage(self._cfg.symbol, self._cfg.leverage):
            self._log.warning("StrongTrendStair: abort entry, set_leverage failed")
            return

        if self._cfg.use_market_entry:
            result = self._safe_open_market_position(
                symbol=self._cfg.symbol,
                side=side,
                qty=qty,
                stop_loss=None,
                leverage=self._cfg.leverage,
                margin_mode=self._cfg.margin_mode,
            )
        else:
            offset = max(0.0, float(self._cfg.entry_limit_offset_bps)) / 10_000.0
            if side == PositionSide.LONG:
                limit_price = reference_price * (1.0 + offset)
            else:
                limit_price = reference_price * (1.0 - offset)
            if limit_price <= 0:
                self._log.warning(
                    "StrongTrendStair: abort entry, invalid limit price %.8f",
                    limit_price,
                )
                return
            try:
                limit_price = self._normalize_price_for_symbol(
                    self._cfg.symbol, limit_price, side
                )
            except Exception as exc:
                self._log.warning(
                    "StrongTrendStair: abort entry, limit price normalization failed symbol=%s raw=%.8f err=%s",
                    self._cfg.symbol,
                    limit_price,
                    exc,
                )
                return
            initial_attached_stop = self._hard_stop_price(limit_price, side)
            try:
                initial_attached_stop = self._normalize_price_for_symbol(
                    self._cfg.symbol, initial_attached_stop, side
                )
            except Exception as exc:
                self._log.warning(
                    "StrongTrendStair: abort entry, attached stop normalization failed symbol=%s raw=%.8f err=%s",
                    self._cfg.symbol,
                    initial_attached_stop,
                    exc,
                )
                return

            result = self._safe_open_limit_position(
                symbol=self._cfg.symbol,
                side=side,
                qty=qty,
                price=limit_price,
                stop_loss=initial_attached_stop,
                leverage=self._cfg.leverage,
                margin_mode=self._cfg.margin_mode,
            )
        if result is None:
            return

        if self._cfg.use_market_entry:
            self._log.info(
                "StrongTrendStair submitted MARKET entry symbol=%s side=%s order_id=%s qty=%.8f",
                self._cfg.symbol,
                side.value,
                result.order_id,
                result.quantity,
            )
        else:
            self._log.info(
                "StrongTrendStair submitted LIMIT entry symbol=%s side=%s order_id=%s qty=%.8f limit=%.8f attached_stop=%.8f",
                self._cfg.symbol,
                side.value,
                result.order_id,
                result.quantity,
                result.price,
                initial_attached_stop,
            )

        timeout_seconds = self._cfg.fill_wait_timeout_seconds
        if not self._cfg.use_market_entry:
            timeout_seconds = max(timeout_seconds, 45.0)

        position = self._wait_for_full_fill(
            symbol=self._cfg.symbol,
            side=side,
            expected_qty=abs(float(result.quantity)),
            order_id=result.order_id,
            timeout_seconds=timeout_seconds,
            poll_seconds=self._cfg.fill_wait_poll_seconds,
        )
        if position is None:
            entry_mode_label = "market" if self._cfg.use_market_entry else "limit"
            self._log.warning(
                "StrongTrendStair %s entry order=%s not observed as filled in time",
                entry_mode_label,
                result.order_id,
            )
            if not self._cfg.use_market_entry:
                if not self._safe_cancel_order(self._cfg.symbol, result.order_id):
                    self._log.warning(
                        "StrongTrendStair failed to cancel unfilled order=%s",
                        result.order_id,
                    )
            self._notify(
                "[STRONG ENTRY ABORTED] %s\nSide: %s\nOrder: %s\n"
                "Reason: %s order not fully confirmed in %.1fs\nTime: %s"
                % (
                    self._cfg.symbol,
                    side.value,
                    result.order_id,
                    entry_mode_label,
                    float(timeout_seconds),
                    datetime.now(timezone.utc).isoformat(),
                )
            )

            partial_position = self._safe_get_position(self._cfg.symbol)
            if partial_position is None:
                return

            partial_qty = abs(float(partial_position.size))
            if partial_qty <= 0:
                return

            if partial_position.side != side:
                self._log.error(
                    "StrongTrendStair unexpected partial position side symbol=%s expected=%s actual=%s",
                    self._cfg.symbol,
                    side.value,
                    partial_position.side.value,
                )
                return

            partial_entry = float(partial_position.entry_price)
            initial_stop = self._hard_stop_price(partial_entry, partial_position.side)
            try:
                initial_stop = self._normalize_price_for_symbol(
                    partial_position.symbol, initial_stop, partial_position.side
                )
            except Exception as exc:
                self._log.warning(
                    "StrongTrendStair partial stop normalization failed symbol=%s raw=%.8f err=%s",
                    partial_position.symbol,
                    initial_stop,
                    exc,
                )
            self._active_stop = initial_stop
            self._last_seen_position_id = (
                partial_position.position_id or self._cfg.symbol
            )
            self._last_seen_position_side = partial_position.side

            ok = self._safe_update_stop_loss(partial_position, initial_stop)
            if ok:
                self._last_stop_sent = initial_stop
                self._notify(
                    "[STRONG PARTIAL FILLED] %s\nSide: %s\nOrder: %s\nQty: %.8f\n"
                    "Entry: %.8f\nInitial Stop: %.8f\nAction: protected and continue managing\nTime: %s"
                    % (
                        self._cfg.symbol,
                        partial_position.side.value,
                        result.order_id,
                        partial_qty,
                        partial_entry,
                        initial_stop,
                        datetime.now(timezone.utc).isoformat(),
                    )
                )
                return

            self._log.error(
                "StrongTrendStair partial fill is unprotected; forcing immediate close symbol=%s qty=%.8f",
                self._cfg.symbol,
                partial_qty,
            )
            ok = self._safe_close_position(self._cfg.symbol)
            if not ok:
                self._notify(
                    "[STRONG CRITICAL] %s\nSide: %s\nOrder: %s\nQty: %.8f\n"
                    "Reason: partial fill and failed to set stop or close position\nTime: %s"
                    % (
                        self._cfg.symbol,
                        partial_position.side.value,
                        result.order_id,
                        partial_qty,
                        datetime.now(timezone.utc).isoformat(),
                    )
                )
            else:
                self._notify(
                    "[STRONG PARTIAL CLOSED] %s\nSide: %s\nOrder: %s\nQty: %.8f\n"
                    "Reason: partial fill timed out and stop update failed, forced close executed\nTime: %s"
                    % (
                        self._cfg.symbol,
                        partial_position.side.value,
                        result.order_id,
                        partial_qty,
                        datetime.now(timezone.utc).isoformat(),
                    )
                )
            return

        actual_qty = abs(float(position.size))
        if actual_qty <= 0:
            return
        entry = float(position.entry_price)
        initial_stop = self._hard_stop_price(entry, position.side)
        try:
            initial_stop = self._normalize_price_for_symbol(
                position.symbol, initial_stop, position.side
            )
        except Exception as exc:
            self._log.warning(
                "StrongTrendStair initial stop normalization failed symbol=%s raw=%.8f err=%s",
                position.symbol,
                initial_stop,
                exc,
            )
        self._active_stop = initial_stop
        ok = self._safe_update_stop_loss(position, initial_stop)
        if not ok:
            self._log.error(
                "StrongTrendStair full-fill position is unprotected; forcing immediate close symbol=%s side=%s qty=%.8f",
                position.symbol,
                position.side.value,
                actual_qty,
            )
            ok = self._safe_close_position(position.symbol)
            if not ok:
                self._notify(
                    "[STRONG CRITICAL] %s\nSide: %s\nQty: %.8f\nEntry: %.8f\n"
                    "Reason: full fill but failed to set initial stop and failed to close position\nTime: %s"
                    % (
                        position.symbol,
                        position.side.value,
                        actual_qty,
                        entry,
                        datetime.now(timezone.utc).isoformat(),
                    )
                )
            else:
                self._notify(
                    "[STRONG OPEN ABORTED] %s\nSide: %s\nQty: %.8f\nEntry: %.8f\n"
                    "Reason: full fill but failed to set initial stop, forced close executed\nTime: %s"
                    % (
                        position.symbol,
                        position.side.value,
                        actual_qty,
                        entry,
                        datetime.now(timezone.utc).isoformat(),
                    )
                )
            return

        self._last_stop_sent = initial_stop
        self._last_seen_position_id = position.position_id or self._cfg.symbol
        self._last_seen_position_side = position.side

        self._notify(
            "[STRONG OPEN] %s\nSide: %s\nEntry: %.8f\nQty: %.8f\n"
            "Notional: %.2f\nLeverage: %sx\nMode: %s\nInitial Stop: %.8f\nTime: %s"
            % (
                self._cfg.symbol,
                side.value,
                entry,
                actual_qty,
                self._cfg.trade_notional_usd,
                self._cfg.leverage,
                self._cfg.margin_mode.value,
                initial_stop,
                datetime.now(timezone.utc).isoformat(),
            )
        )

    def _evaluate_snapshot(
        self,
    ) -> Optional[
        Tuple[
            Candle,
            Optional[PositionSide],
            Optional[Tuple[float, float, float, float, float, float, float]],
        ]
    ]:
        """Build latest indicator snapshot and return directional signal for the last closed candle."""
        candles = self._safe_get_klines(
            symbol=self._cfg.symbol,
            interval=self._cfg.timeframe,
            limit=self._cfg.klines_limit,
        )
        if not candles:
            return None
        now_utc = datetime.now(timezone.utc)
        closed_candles = [c for c in candles if c.close_time <= now_utc]
        if not closed_candles:
            return None
        candles = closed_candles
        last_closed = candles[-1]
        if len(candles) < max(
            self._cfg.ema_slow_len + self._cfg.slope_lookback + 2, 230
        ):
            self._log.info(
                "StrongTrendStair insufficient data for snapshot; skipping tick"
            )
            return last_closed, None, None

        closes = [c.close for c in candles]
        ema_fast = calc_ema(closes, self._cfg.ema_fast_len)
        ema_mid = calc_ema(closes, self._cfg.ema_mid_len)
        ema_slow = calc_ema(closes, self._cfg.ema_slow_len)
        st_line = self._supertrend_line(
            candles, self._cfg.st_atr_len, self._cfg.st_factor
        )
        di_plus, di_minus, adx = self._dmi(
            candles, self._cfg.di_len, self._cfg.adx_smooth
        )

        idx = len(candles) - 1
        slope_idx = idx - self._cfg.slope_lookback
        values = (
            ema_fast[idx],
            ema_mid[idx],
            ema_slow[idx],
            st_line[idx],
            di_plus[idx],
            di_minus[idx],
            adx[idx],
            ema_slow[slope_idx] if slope_idx >= 0 else None,
        )
        if any(value is None for value in values):
            return last_closed, None, None
        fast, mid, slow, st, plus, minus, adx_now, slow_prev = (
            float(values[0]),
            float(values[1]),
            float(values[2]),
            float(values[3]),
            float(values[4]),
            float(values[5]),
            float(values[6]),
            float(values[7]),
        )
        candle = candles[idx]
        ema_slope_up = slow > slow_prev
        ema_slope_dn = slow < slow_prev
        bull = (
            candle.close > st
            and candle.close > slow
            and fast > mid > slow
            and ema_slope_up
            and adx_now >= self._cfg.adx_min
            and plus > minus
        )
        bear = (
            candle.close < st
            and candle.close < slow
            and fast < mid < slow
            and ema_slope_dn
            and adx_now >= self._cfg.adx_min
            and minus > plus
        )
        if bull:
            return (
                candle,
                PositionSide.LONG,
                (fast, mid, slow, st, plus, minus, adx_now),
            )
        if bear:
            return (
                candle,
                PositionSide.SHORT,
                (fast, mid, slow, st, plus, minus, adx_now),
            )
        return candle, None, (fast, mid, slow, st, plus, minus, adx_now)

    def _safe_get_account_balance(self) -> Optional[float]:
        """Fetch account balance with retry/backoff, returning ``None`` on repeated failure."""
        for attempt in range(1, self._cfg.api_retries + 1):
            try:
                return float(self._exchange.get_account_balance())
            except Exception as exc:
                self._log.warning(
                    "StrongTrendStair get_account_balance failed attempt=%s/%s err=%s",
                    attempt,
                    self._cfg.api_retries,
                    exc,
                )
                if attempt < self._cfg.api_retries:
                    time.sleep(self._cfg.api_retry_delay_seconds)
        return None

    def _has_sufficient_balance_for_entry(self, reference_price: float) -> bool:
        """Validate that available margin can support configured notional with a small safety buffer."""
        if reference_price <= 0:
            return False
        balance = self._safe_get_account_balance()
        if balance is None:
            self._log.warning(
                "StrongTrendStair skip entry on candle %s: account balance unavailable",
                datetime.now(timezone.utc).isoformat(),
            )
            return False
        required_margin = self._cfg.trade_notional_usd / max(
            float(self._cfg.leverage), 1.0
        )
        # small buffer to reduce edge rejections
        required_with_buffer = required_margin * 1.02
        if balance < required_with_buffer:
            self._log.warning(
                "StrongTrendStair skip entry: insufficient balance available=%.8f required≈%.8f (notional=%.2f leverage=%s)",
                balance,
                required_with_buffer,
                self._cfg.trade_notional_usd,
                self._cfg.leverage,
            )
            return False
        return True

    @staticmethod
    def _true_range(candles: List[Candle]) -> List[float]:
        """Compute True Range series for the candle sequence."""
        tr: List[float] = []
        for i, candle in enumerate(candles):
            if i == 0:
                tr.append(candle.high - candle.low)
                continue
            prev_close = candles[i - 1].close
            tr.append(
                max(
                    candle.high - candle.low,
                    abs(candle.high - prev_close),
                    abs(candle.low - prev_close),
                )
            )
        return tr

    @staticmethod
    def _rma(values: List[float], length: int) -> List[Optional[float]]:
        """Compute Wilder RMA, returning ``None`` until seed window is available."""
        if length <= 0 or len(values) < length:
            return [None] * len(values)
        out: List[Optional[float]] = [None] * len(values)
        seed = sum(values[:length]) / float(length)
        out[length - 1] = seed
        prev = seed
        for i in range(length, len(values)):
            prev = (prev * (length - 1) + values[i]) / float(length)
            out[i] = prev
        return out

    def _supertrend_line(
        self, candles: List[Candle], atr_len: int, factor: float
    ) -> List[Optional[float]]:
        """Compute Supertrend line values from candles, ATR length, and multiplier."""
        tr = self._true_range(candles)
        atr = self._rma(tr, atr_len)
        n = len(candles)
        out: List[Optional[float]] = [None] * n
        final_upper: List[Optional[float]] = [None] * n
        final_lower: List[Optional[float]] = [None] * n
        for i, candle in enumerate(candles):
            atr_i = atr[i]
            if atr_i is None:
                continue
            hl2 = (candle.high + candle.low) / 2.0
            upper = hl2 + factor * atr_i
            lower = hl2 - factor * atr_i
            if i == 0 or final_upper[i - 1] is None or final_lower[i - 1] is None:
                final_upper[i] = upper
                final_lower[i] = lower
                out[i] = lower
                continue
            prev_final_upper = float(final_upper[i - 1])
            prev_final_lower = float(final_lower[i - 1])
            prev_close = candles[i - 1].close
            final_upper[i] = (
                upper
                if upper < prev_final_upper or prev_close > prev_final_upper
                else prev_final_upper
            )
            final_lower[i] = (
                lower
                if lower > prev_final_lower or prev_close < prev_final_lower
                else prev_final_lower
            )
            prev_st = out[i - 1]
            if prev_st is None:
                out[i] = final_lower[i]
            elif abs(float(prev_st) - prev_final_upper) < 1e-12:
                out[i] = (
                    final_upper[i]
                    if candle.close <= float(final_upper[i])
                    else final_lower[i]
                )
            else:
                out[i] = (
                    final_lower[i]
                    if candle.close >= float(final_lower[i])
                    else final_upper[i]
                )
        return out

    def _dmi(
        self, candles: List[Candle], di_len: int, adx_smooth: int
    ) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
        """Compute +DI, -DI, and ADX series."""
        n = len(candles)
        plus_dm: List[float] = [0.0] * n
        minus_dm: List[float] = [0.0] * n
        tr = self._true_range(candles)
        for i in range(1, n):
            up_move = candles[i].high - candles[i - 1].high
            down_move = candles[i - 1].low - candles[i].low
            plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
            minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0

        tr_rma = self._rma(tr, di_len)
        plus_rma = self._rma(plus_dm, di_len)
        minus_rma = self._rma(minus_dm, di_len)
        plus_di: List[Optional[float]] = [None] * n
        minus_di: List[Optional[float]] = [None] * n
        dx_values: List[Optional[float]] = [None] * n
        for i in range(n):
            tr_val = tr_rma[i]
            p_val = plus_rma[i]
            m_val = minus_rma[i]
            if tr_val is None or p_val is None or m_val is None or tr_val <= 0:
                continue
            p_di = 100.0 * (p_val / tr_val)
            m_di = 100.0 * (m_val / tr_val)
            plus_di[i] = p_di
            minus_di[i] = m_di
            denom = p_di + m_di
            if denom > 0:
                dx_values[i] = 100.0 * abs(p_di - m_di) / denom

        adx: List[Optional[float]] = [None] * n
        valid_dx_points = [
            (idx, float(value))
            for idx, value in enumerate(dx_values)
            if value is not None
        ]
        if len(valid_dx_points) < adx_smooth:
            return plus_di, minus_di, adx

        seed_window = valid_dx_points[:adx_smooth]
        prev_adx = sum(point[1] for point in seed_window) / float(adx_smooth)
        seed_index = seed_window[-1][0]
        adx[seed_index] = prev_adx

        for idx, dx_val in valid_dx_points[adx_smooth:]:
            prev_adx = (prev_adx * (adx_smooth - 1) + dx_val) / float(adx_smooth)
            adx[idx] = prev_adx
        return plus_di, minus_di, adx

    def _should_send_stop(self, stop_price: float, side: PositionSide) -> bool:
        """Return ``True`` when stop value changed after exchange-precision quantization."""
        if self._last_stop_sent is None:
            return True
        try:
            # Keep comparison precision identical to the actual stop submission path.
            last_q = self._normalize_price_for_symbol(
                self._cfg.symbol, self._last_stop_sent, side
            )
            new_q = self._normalize_price_for_symbol(self._cfg.symbol, stop_price, side)
            return new_q != last_q
        except Exception:
            # If normalization fails, allow sending so the updater can log the concrete failure.
            return True

    @staticmethod
    def _quantize(value: float, decimals: int, *, rounding_mode: str) -> float:
        """Quantize ``value`` to fixed decimal places using the provided rounding mode."""
        precision = max(int(decimals), 0)
        quantum = Decimal("1").scaleb(-precision)
        return float(Decimal(str(value)).quantize(quantum, rounding=rounding_mode))

    def _get_symbol_meta(self, symbol: str) -> Dict[str, Any]:
        """Fetch and cache symbol trading metadata when supported by the exchange adapter."""
        normalized_symbol = str(symbol).strip().upper()
        cached = self._symbol_meta_cache.get(normalized_symbol)
        if cached is not None:
            return cached

        getter = getattr(self._exchange, "get_trading_pairs", None)
        if not callable(getter):
            return {}
        try:
            pairs = getter(symbols=[normalized_symbol])
            if isinstance(pairs, list):
                for item in pairs:
                    if (
                        isinstance(item, dict)
                        and str(item.get("symbol", "")).strip().upper()
                        == normalized_symbol
                    ):
                        self._symbol_meta_cache[normalized_symbol] = item
                        return item
        except Exception as exc:
            self._log.debug(
                "StrongTrendStair symbol meta fetch failed symbol=%s err=%s",
                normalized_symbol,
                exc,
            )
        return {}

    def _normalize_quantity_for_symbol(
        self, symbol: str, quantity: float, reference_price: Optional[float] = None
    ) -> Optional[float]:
        """Normalize quantity to symbol rules and reject below-min quantity/notional values."""
        qty = float(quantity)
        if qty <= 0:
            return None
        meta = self._get_symbol_meta(symbol)
        step_size = self._first_positive_float(
            meta,
            (
                "stepSize",
                "qtyStep",
                "quantityStep",
                "lotSize",
                "baseIncrement",
            ),
        )
        if step_size is not None:
            normalized = self._quantize_to_step(
                qty, step_size, rounding_mode=ROUND_DOWN
            )
        else:
            base_precision = int(meta.get("basePrecision", 8) or 8)
            normalized = self._quantize(qty, base_precision, rounding_mode=ROUND_DOWN)
        if normalized <= 0:
            return None

        min_qty = (
            self._first_positive_float(
                meta,
                (
                    "minQty",
                    "minQuantity",
                    "minTradeVolume",
                    "minOrderQty",
                ),
            )
            or 0.0
        )
        if min_qty > 0 and normalized < min_qty:
            self._log.warning(
                "StrongTrendStair normalized quantity below min qty symbol=%s raw=%.12f normalized=%.12f min=%.12f",
                symbol,
                qty,
                normalized,
                min_qty,
            )
            return None

        min_notional = (
            self._first_positive_float(
                meta,
                (
                    "minNotional",
                    "minOrderValue",
                    "minTradeAmount",
                    "minNotionalValue",
                ),
            )
            or 0.0
        )
        ref_px = (
            reference_price
            if reference_price is not None
            else self._safe_fetch_price(symbol)
        )
        if min_notional > 0 and ref_px is not None and ref_px > 0:
            notional = normalized * float(ref_px)
            if notional < min_notional:
                self._log.warning(
                    "StrongTrendStair normalized quantity below min notional symbol=%s qty=%.12f ref_price=%.12f notional=%.12f min_notional=%.12f",
                    symbol,
                    normalized,
                    float(ref_px),
                    notional,
                    min_notional,
                )
                return None
        return normalized

    def _normalize_price_for_symbol(
        self, symbol: str, price: float, side: PositionSide
    ) -> float:
        """Normalize price to symbol tick/precision using side-aware rounding direction."""
        px = float(price)
        if px <= 0:
            raise RuntimeError(f"Invalid price for {symbol}: {price}")
        meta = self._get_symbol_meta(symbol)
        rounding = ROUND_DOWN if side == PositionSide.LONG else ROUND_UP
        tick_size = self._first_positive_float(
            meta,
            (
                "tickSize",
                "priceStep",
                "quoteIncrement",
                "minPriceIncrement",
            ),
        )
        if tick_size is not None:
            normalized = self._quantize_to_step(px, tick_size, rounding_mode=rounding)
        else:
            quote_precision = int(meta.get("quotePrecision", 8) or 8)
            normalized = self._quantize(px, quote_precision, rounding_mode=rounding)
        if normalized <= 0:
            raise RuntimeError(
                f"Invalid normalized price for {symbol}: raw={px} normalized={normalized}"
            )
        return normalized

    @staticmethod
    def _first_positive_float(
        meta: Dict[str, Any], keys: Tuple[str, ...]
    ) -> Optional[float]:
        """Return the first positive float from ``meta`` for the ordered key list."""
        for key in keys:
            try:
                value = float(meta.get(key, 0) or 0)
                if value > 0:
                    return value
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _quantize_to_step(value: float, step: float, *, rounding_mode: str) -> float:
        """Quantize value to discrete step size using decimal-safe arithmetic."""
        if step <= 0:
            return float(value)
        d_value = Decimal(str(value))
        d_step = Decimal(str(step))
        units = (d_value / d_step).quantize(Decimal("1"), rounding=rounding_mode)
        return float(units * d_step)

    def _notify(self, message: str) -> None:
        """Send Telegram message if notifier exists; log warning instead of raising on failure."""
        if not self._telegram:
            return
        try:
            self._telegram.send_message(message)
        except Exception as exc:
            self._log.warning("StrongTrendStair telegram send failed: %s", exc)

    def _get_binance_client(self) -> BinanceClient:
        if self._binance_client is None:
            cfg = BinanceClientConfig(proxies=None)
            self._binance_client = BinanceClient(
                cfg, logger=logging.getLogger("binance.client")
            )
        return self._binance_client

    @staticmethod
    def _interval_to_ms(interval: str) -> int:
        mapping = {
            "1m": 60_000,
            "3m": 180_000,
            "5m": 300_000,
            "15m": 900_000,
            "30m": 1_800_000,
            "1h": 3_600_000,
            "2h": 7_200_000,
            "4h": 14_400_000,
            "6h": 21_600_000,
            "12h": 43_200_000,
            "1d": 86_400_000,
        }
        return mapping.get(interval, 0)

    def _fetch_binance_klines_recent(
        self, symbol: str, interval: str, limit: int
    ) -> List[List]:
        # Fetch most recent candles from Binance because Bitunix caps at 200.
        interval_ms = self._interval_to_ms(interval)
        if interval_ms <= 0:
            raise ValueError(f"Unsupported interval for Binance fetch: {interval}")
        capped_limit = min(limit, MAX_BATCH)
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = end_ms - capped_limit * interval_ms
        client = self._get_binance_client()
        candles = client.fetch_klines(
            symbol=symbol,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
            limit=capped_limit,
        )
        if not candles:
            return []
        # Return Binance payload shape (raw kline rows) to keep downstream parsing consistent.
        return [
            [
                c.open_time_ms,
                c.open,
                c.high,
                c.low,
                c.close,
                c.volume,
                int(c.close_time.timestamp() * 1000),
            ]
            for c in candles
        ]

    def _safe_fetch_price(self, symbol: str) -> Optional[float]:
        """Fetch symbol price with retry/backoff, returning ``None`` if all attempts fail."""
        for attempt in range(1, self._cfg.api_retries + 1):
            try:
                return self._exchange.fetch_price(symbol)
            except Exception as exc:
                self._log.warning(
                    "StrongTrendStair fetch_price failed symbol=%s attempt=%s/%s err=%s",
                    symbol,
                    attempt,
                    self._cfg.api_retries,
                    exc,
                )
                if attempt < self._cfg.api_retries:
                    time.sleep(self._cfg.api_retry_delay_seconds)
        return None

    def _safe_get_position(self, symbol: str) -> Optional[Position]:
        """Return current position only when retrieval status is known."""
        position, known = self._safe_get_position_with_status(symbol)
        if not known:
            return None
        return position

    def _safe_get_position_with_status(
        self, symbol: str
    ) -> Tuple[Optional[Position], bool]:
        """Fetch position with a success flag to distinguish flat state from API failure."""
        for attempt in range(1, self._cfg.api_retries + 1):
            try:
                return self._exchange.get_position(symbol), True
            except Exception as exc:
                self._log.warning(
                    "StrongTrendStair get_position failed symbol=%s attempt=%s/%s err=%s",
                    symbol,
                    attempt,
                    self._cfg.api_retries,
                    exc,
                )
                if attempt < self._cfg.api_retries:
                    time.sleep(self._cfg.api_retry_delay_seconds)
        return None, False

    def _wait_for_full_fill(
        self,
        symbol: str,
        side: PositionSide,
        expected_qty: float,
        order_id: str,
        timeout_seconds: float,
        poll_seconds: float,
    ) -> Optional[Position]:
        """Wait until expected quantity is filled, or abort when timeout/non-fillable status occurs."""
        min_full_qty = max(0.0, expected_qty) * 0.999
        deadline = time.monotonic() + max(timeout_seconds, 0.1)
        pause = max(poll_seconds, 0.1)

        while time.monotonic() <= deadline:
            position, known = self._safe_get_position_with_status(symbol)
            if known and position is not None and position.side == side:
                current_qty = abs(float(position.size))
                if current_qty >= min_full_qty:
                    self._log.info(
                        "StrongTrendStair entry fully filled symbol=%s side=%s qty=%.8f/%.8f",
                        symbol,
                        side.value,
                        current_qty,
                        expected_qty,
                    )
                    return position
                self._log.info(
                    "StrongTrendStair partial fill symbol=%s side=%s qty=%.8f/%.8f order_id=%s",
                    symbol,
                    side.value,
                    current_qty,
                    expected_qty,
                    order_id,
                )
            elif not known:
                self._log.warning(
                    "StrongTrendStair fill wait: position state unknown symbol=%s order_id=%s; continuing",
                    symbol,
                    order_id,
                )

            detail = self._safe_get_order_detail(order_id)
            if detail:
                status_raw = (
                    str(
                        detail.get("status")
                        or detail.get("orderStatus")
                        or detail.get("state")
                        or ""
                    )
                    .strip()
                    .upper()
                )
                if status_raw in {"CANCELED", "CANCELLED", "REJECTED", "EXPIRED"}:
                    self._log.warning(
                        "StrongTrendStair order became non-fillable order_id=%s status=%s",
                        order_id,
                        status_raw,
                    )
                    return None

            time.sleep(pause)
        return None

    def _wait_until_flat(
        self, symbol: str, timeout_seconds: float, poll_seconds: float
    ) -> bool:
        """Poll until no open position exists for ``symbol`` or until timeout."""
        deadline = time.monotonic() + max(timeout_seconds, 0.1)
        pause = max(poll_seconds, 0.1)
        while time.monotonic() <= deadline:
            pos, known = self._safe_get_position_with_status(symbol)
            if known and pos is None:
                return True
            if not known:
                self._log.warning(
                    "StrongTrendStair wait_until_flat: position state unknown symbol=%s; continuing to wait",
                    symbol,
                )
            time.sleep(pause)
        return False

    def _safe_get_klines(self, symbol: str, interval: str, limit: int) -> List[Candle]:
        """Fetch klines from Binance (Bitunix REST caps at 200)."""
        for attempt in range(1, self._cfg.api_retries + 1):
            try:
                rows = self._fetch_binance_klines_recent(symbol, interval, limit)
                return [Candle.from_binance(symbol, interval, row) for row in rows]
            except Exception as exc:
                self._log.warning(
                    "StrongTrendStair binance klines failed symbol=%s interval=%s attempt=%s/%s err=%s",
                    symbol,
                    interval,
                    attempt,
                    self._cfg.api_retries,
                    exc,
                )
                if attempt < self._cfg.api_retries:
                    time.sleep(self._cfg.api_retry_delay_seconds)
        return []

    def _safe_set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set symbol leverage with retries."""
        for attempt in range(1, self._cfg.api_retries + 1):
            try:
                self._exchange.set_leverage(symbol, leverage)
                return True
            except Exception as exc:
                self._log.warning(
                    "StrongTrendStair set_leverage failed symbol=%s leverage=%s attempt=%s/%s err=%s",
                    symbol,
                    leverage,
                    attempt,
                    self._cfg.api_retries,
                    exc,
                )
                if attempt < self._cfg.api_retries:
                    time.sleep(self._cfg.api_retry_delay_seconds)
        return False

    def _safe_set_margin_mode(self, symbol: str, margin_mode: MarginMode) -> bool:
        """Set symbol margin mode with retries."""
        for attempt in range(1, self._cfg.api_retries + 1):
            try:
                self._exchange.set_margin_mode(symbol, margin_mode)
                return True
            except Exception as exc:
                self._log.warning(
                    "StrongTrendStair set_margin_mode failed symbol=%s mode=%s attempt=%s/%s err=%s",
                    symbol,
                    margin_mode.value,
                    attempt,
                    self._cfg.api_retries,
                    exc,
                )
                if attempt < self._cfg.api_retries:
                    time.sleep(self._cfg.api_retry_delay_seconds)
        return False

    def _safe_open_market_position(
        self,
        symbol: str,
        side: PositionSide,
        qty: float,
        stop_loss: Optional[float],
        leverage: int,
        margin_mode: MarginMode,
    ):
        """Open a market position using retry/backoff semantics."""
        for attempt in range(1, self._cfg.api_retries + 1):
            try:
                return self._exchange.open_market_position(
                    symbol=symbol,
                    side=side,
                    quantity=qty,
                    leverage=leverage,
                    margin_mode=margin_mode,
                    take_profit=None,
                    stop_loss=stop_loss,
                )
            except Exception as exc:
                self._log.warning(
                    "StrongTrendStair open_market_position failed symbol=%s side=%s qty=%.8f stop=%.8f attempt=%s/%s err=%s",
                    symbol,
                    side.value,
                    qty,
                    stop_loss if stop_loss is not None else float('nan'),
                    attempt,
                    self._cfg.api_retries,
                    exc,
                )
                if attempt < self._cfg.api_retries:
                    time.sleep(self._cfg.api_retry_delay_seconds)
        return None

    def _safe_open_limit_position(
        self,
        symbol: str,
        side: PositionSide,
        qty: float,
        price: float,
        stop_loss: float,
        leverage: int,
        margin_mode: MarginMode,
    ):
        """Open a limit position with attached stop-loss using retry/backoff semantics."""
        for attempt in range(1, self._cfg.api_retries + 1):
            try:
                return self._exchange.open_limit_position(
                    symbol=symbol,
                    side=side,
                    quantity=qty,
                    price=price,
                    leverage=leverage,
                    margin_mode=margin_mode,
                    take_profit=None,
                    stop_loss=stop_loss,
                )
            except Exception as exc:
                self._log.warning(
                    "StrongTrendStair open_limit_position failed symbol=%s side=%s qty=%.8f price=%.8f stop=%.8f attempt=%s/%s err=%s",
                    symbol,
                    side.value,
                    qty,
                    price,
                    stop_loss,
                    attempt,
                    self._cfg.api_retries,
                    exc,
                )
                if attempt < self._cfg.api_retries:
                    time.sleep(self._cfg.api_retry_delay_seconds)
        return None

    def _safe_get_order_detail(self, order_id: str) -> Optional[dict]:
        """Fetch order detail when adapter exposes the method, with retries."""
        getter = getattr(self._exchange, "get_order_detail", None)
        if not callable(getter):
            return None
        for attempt in range(1, self._cfg.api_retries + 1):
            try:
                detail = getter(order_id=order_id)
                if isinstance(detail, dict):
                    return detail
                return None
            except Exception as exc:
                self._log.warning(
                    "StrongTrendStair get_order_detail failed order_id=%s attempt=%s/%s err=%s",
                    order_id,
                    attempt,
                    self._cfg.api_retries,
                    exc,
                )
                if attempt < self._cfg.api_retries:
                    time.sleep(self._cfg.api_retry_delay_seconds)
        return None

    def _safe_cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel order with retries and return cancellation success status."""
        for attempt in range(1, self._cfg.api_retries + 1):
            try:
                if self._exchange.cancel_order(symbol, order_id):
                    return True
            except Exception as exc:
                self._log.warning(
                    "StrongTrendStair cancel_order failed symbol=%s order_id=%s attempt=%s/%s err=%s",
                    symbol,
                    order_id,
                    attempt,
                    self._cfg.api_retries,
                    exc,
                )
            if attempt < self._cfg.api_retries:
                time.sleep(self._cfg.api_retry_delay_seconds)
        return False

    def _safe_update_stop_loss(self, position: Position, stop_price: float) -> bool:
        """Normalize and place/refresh a position-level stop-loss with retries."""
        if not position.position_id:
            self._log.warning(
                "StrongTrendStair cannot place position stop without position_id symbol=%s",
                position.symbol,
            )
            return False

        try:
            normalized_stop = self._normalize_price_for_symbol(
                position.symbol, stop_price, position.side
            )
        except Exception as exc:
            self._log.warning(
                "StrongTrendStair stop normalization failed symbol=%s raw_stop=%.8f err=%s",
                position.symbol,
                stop_price,
                exc,
            )
            return False

        for attempt in range(1, self._cfg.api_retries + 1):
            try:
                result = self._exchange.place_position_tpsl_order(
                    symbol=position.symbol,
                    position_id=position.position_id,
                    sl_price=normalized_stop,
                    sl_stop_type="MARK_PRICE",
                )
                if result:
                    self._stop_order_id = None
                    order_id = result.get("orderId") if isinstance(result, dict) else None
                    self._log.info(
                        "StrongTrendStair position stop placed/updated symbol=%s side=%s stop=%.8f order=%s",
                        position.symbol,
                        position.side.value,
                        normalized_stop,
                        order_id or "n/a",
                    )
                    return True

                self._log.warning(
                    "StrongTrendStair position stop placement returned empty response symbol=%s stop=%.8f attempt=%s/%s",
                    position.symbol,
                    normalized_stop,
                    attempt,
                    self._cfg.api_retries,
                )
            except Exception as exc:
                self._log.warning(
                    "StrongTrendStair position stop placement failed symbol=%s stop=%.8f attempt=%s/%s err=%s",
                    position.symbol,
                    normalized_stop,
                    attempt,
                    self._cfg.api_retries,
                    exc,
                )
            if attempt < self._cfg.api_retries:
                time.sleep(self._cfg.api_retry_delay_seconds)
        return False

    def _safe_close_position(self, symbol: str) -> bool:
        """Close position with retries and clear in-memory trailing stop state on success."""
        for attempt in range(1, self._cfg.api_retries + 1):
            try:
                self._exchange.close_position(symbol)
                self._active_stop = None
                self._last_stop_sent = None
                self._stop_order_id = None
                return True
            except Exception as exc:
                self._log.warning(
                    "StrongTrendStair close_position failed symbol=%s attempt=%s/%s err=%s",
                    symbol,
                    attempt,
                    self._cfg.api_retries,
                    exc,
                )
                if attempt < self._cfg.api_retries:
                    time.sleep(self._cfg.api_retry_delay_seconds)
        return False
