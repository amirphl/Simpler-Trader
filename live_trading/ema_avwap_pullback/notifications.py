"""Telegram notifications for EMA + AVWAP live trades."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from ..models import PositionRecord
from ._mixin_typing import EmaAvwapMixinTyping
from .config import EntryMode
from .state import _AvwapSnapshot, _EntryCandidate, _PositionRuntime


class EmaAvwapNotificationMixin(EmaAvwapMixinTyping):
    def _send_trade_notification(
        self, *, symbol: str, event: str, lines: list[str]
    ) -> None:
        if not self._telegram:
            return
        try:
            self._telegram.send_message(
                "\n".join([f"[EMA AVWAP {event}] {symbol}", *lines])
            )
        except Exception as exc:
            self._log.warning(
                "EmaAvwapPullback: failed to send %s notification for %s: %s",
                event.lower(),
                symbol,
                exc,
            )

    def _notify_entry_signal(self, candidate: _EntryCandidate) -> None:
        target_band = candidate.exit_band.number
        if candidate.entry_mode is EntryMode.CLOSE:
            reason = (
                "bearish candle closed below AVWAP middle"
                if candidate.direction == "long"
                else "bullish candle closed above AVWAP middle"
            )
        else:
            reason = "live price touched AVWAP middle"
        target = self._target_band_level(
            candidate.direction, candidate.avwap, candidate.exit_band
        )
        self._send_trade_notification(
            symbol=candidate.symbol,
            event="ENTRY SIGNAL",
            lines=[
                f"Entry Mode: {candidate.entry_mode.value}",
                f"Exit Mode: {candidate.exit_mode.value}",
                f"Timeframe: {self._cfg.timeframe}",
                f"Side: {candidate.side.value}",
                f"Trigger: {candidate.entry_trigger_mode}",
                f"Reason: {reason}",
                f"Entry: {candidate.order_price:.8g}",
                f"Decision Price: {candidate.decision_price:.8g}"
                if candidate.decision_price is not None
                else "Decision Price: unavailable",
                f"AVWAP Middle: {candidate.avwap.vwap:.8g}",
                f"EMA: {candidate.ema_value:.8g}"
                if candidate.ema_value is not None
                else "EMA: unavailable",
                f"Band 1: {candidate.avwap.upper1:.8g} / {candidate.avwap.lower1:.8g}",
                f"Band 2: {candidate.avwap.upper2:.8g} / {candidate.avwap.lower2:.8g}",
                f"Exit Band: {target_band} @ {target:.8g}",
                f"Rigid Stop: {candidate.rigid_stop_at_entry:.8g}"
                if candidate.rigid_stop_at_entry is not None
                else "Rigid Stop: disabled",
                f"Anchor: {candidate.anchor_time.isoformat()}",
                f"Signal Time: {candidate.signal_time.isoformat()}",
            ],
        )

    def _notify_exit_signal(
        self,
        position: PositionRecord,
        *,
        runtime: _PositionRuntime,
        reason: str,
        live_price: float,
        target_price: float,
        avwap: _AvwapSnapshot,
    ) -> None:
        self._send_trade_notification(
            symbol=position.symbol,
            event="EXIT SIGNAL",
            lines=[
                f"Entry Mode: {runtime.entry_mode.value}",
                f"Exit Mode: {runtime.exit_mode.value}",
                f"Exit Band: {runtime.exit_band.value}",
                f"Timeframe: {self._cfg.timeframe}",
                f"Side: {position.side.value}",
                f"Reason: {reason}",
                f"Observed Price: {live_price:.8g}",
                f"Trigger Level: {target_price:.8g}",
                f"AVWAP Middle: {avwap.vwap:.8g}",
                f"EMA: {runtime.last_ema_value:.8g}"
                if runtime.last_ema_value is not None
                else "EMA: unavailable",
                f"Band 1: {avwap.upper1:.8g} / {avwap.lower1:.8g}",
                f"Band 2: {avwap.upper2:.8g} / {avwap.lower2:.8g}",
                f"Rigid Stop: {runtime.rigid_stop_level:.8g}"
                if runtime.rigid_stop_level is not None
                else "Rigid Stop: disabled",
                f"Time: {datetime.now(timezone.utc).isoformat()}",
            ],
        )

    def _notify_trade_opened(
        self,
        position: PositionRecord,
        runtime: _PositionRuntime,
        stop_price: float | None,
    ) -> None:
        avwap = runtime.last_avwap
        target_band = runtime.exit_band.number
        lines = [
            f"Entry Mode: {runtime.entry_mode.value}",
            f"Exit Mode: {runtime.exit_mode.value}",
            f"Timeframe: {self._cfg.timeframe}",
            f"Side: {position.side.value}",
            f"Entry: {position.entry_price:.8g}",
            f"Qty: {position.quantity:.8g}",
            f"Leverage: {position.leverage}x",
            f"Rigid Stop: {stop_price:.8g}"
            if stop_price is not None
            else "Rigid Stop: disabled",
            f"Anchor: {runtime.anchor_time.isoformat()}",
            f"Trigger: {runtime.entry_trigger_mode}",
            "Reason: entry order filled",
            f"Time: {datetime.now(timezone.utc).isoformat()}",
        ]
        if avwap is not None:
            lines.extend(
                [
                    f"AVWAP Middle: {avwap.vwap:.8g}",
                    f"Band 1: {avwap.upper1:.8g} / {avwap.lower1:.8g}",
                    f"Band 2: {avwap.upper2:.8g} / {avwap.lower2:.8g}",
                    f"EMA: {runtime.last_ema_value:.8g}"
                    if runtime.last_ema_value is not None
                    else "EMA: unavailable",
                    f"Exit Band: {target_band} @ "
                    f"{self._target_band_level(runtime.direction, avwap, runtime.exit_band):.8g}",
                ]
            )
        self._send_trade_notification(
            symbol=position.symbol, event="ENTRY EXECUTED", lines=lines
        )

    def _notify_trade_closed(
        self,
        position: PositionRecord,
        *,
        reason: str,
        exit_price: Optional[float],
        runtime: _PositionRuntime | None = None,
    ) -> None:
        entry_mode = runtime.entry_mode if runtime is not None else self._cfg.entry_mode
        exit_mode = runtime.exit_mode if runtime is not None else self._cfg.exit_mode
        exit_band = runtime.exit_band if runtime is not None else self._cfg.exit_band
        lines = [
            f"Entry Mode: {entry_mode.value}",
            f"Exit Mode: {exit_mode.value}",
            f"Exit Band: {exit_band.value}",
            f"Timeframe: {self._cfg.timeframe}",
            f"Side: {position.side.value}",
            f"Entry: {position.entry_price:.8g}",
            f"Exit: {exit_price:.8g}" if exit_price is not None else "Exit: n/a",
            f"Qty: {position.quantity:.8g}",
            f"PnL: {position.pnl:.8g}"
            if position.pnl is not None
            else "PnL: n/a",
            f"Reason: {reason}",
            f"Time: {datetime.now(timezone.utc).isoformat()}",
        ]
        if runtime is not None and runtime.last_avwap is not None:
            avwap = runtime.last_avwap
            lines.extend(
                [
                    f"AVWAP Middle: {avwap.vwap:.8g}",
                    f"EMA: {runtime.last_ema_value:.8g}"
                    if runtime.last_ema_value is not None
                    else "EMA: unavailable",
                    f"Band 1: {avwap.upper1:.8g} / {avwap.lower1:.8g}",
                    f"Band 2: {avwap.upper2:.8g} / {avwap.lower2:.8g}",
                    f"Rigid Stop: {runtime.rigid_stop_level:.8g}"
                    if runtime.rigid_stop_level is not None
                    else "Rigid Stop: disabled",
                ]
            )
        self._send_trade_notification(
            symbol=position.symbol, event="EXIT EXECUTED", lines=lines
        )
