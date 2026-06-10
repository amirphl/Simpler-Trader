"""Telegram notifications for EMA + AVWAP live trades."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from ..models import PositionRecord
from ._mixin_typing import EmaAvwapMixinTyping
from .state import _PositionRuntime


class EmaAvwapNotificationMixin(EmaAvwapMixinTyping):
    def _notify_trade_opened(
        self, position: PositionRecord, runtime: _PositionRuntime, stop_price: float
    ) -> None:
        if not self._telegram:
            return
        try:
            lines = [
                f"[EMA AVWAP OPEN] {position.symbol}",
                f"Timeframe: {self._cfg.timeframe}",
                f"Side: {position.side.value}",
                f"Entry: {position.entry_price:.8g}",
                f"Qty: {position.quantity:.8g}",
                f"Leverage: {position.leverage}x",
                f"Stop: {stop_price:.8g}",
                f"Rigid Stop: {runtime.rigid_stop_level:.8g}"
                if runtime.rigid_stop_level is not None
                else "Rigid Stop: n/a",
                f"Anchor: {runtime.anchor_time.isoformat()}",
                f"Trigger: {runtime.entry_trigger_mode}",
                f"Time: {datetime.now(timezone.utc).isoformat()}",
            ]
            self._telegram.send_message("\n".join(lines))
        except Exception as exc:
            self._log.warning(
                "EmaAvwapPullback: failed to send open notification for %s: %s",
                position.symbol,
                exc,
            )

    def _notify_trade_closed(
        self, position: PositionRecord, *, reason: str, exit_price: Optional[float]
    ) -> None:
        if not self._telegram:
            return
        try:
            lines = [
                f"[EMA AVWAP CLOSE] {position.symbol}",
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
            self._telegram.send_message("\n".join(lines))
        except Exception as exc:
            self._log.warning(
                "EmaAvwapPullback: failed to send close notification for %s: %s",
                position.symbol,
                exc,
            )
