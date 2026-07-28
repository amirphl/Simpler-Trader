"""Candle fetching and snapshot construction for EMA + AVWAP."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from candle_downloader.binance import interval_to_milliseconds
from candle_downloader.models import Candle

from backtest.indicators import ema as calc_ema

from ._mixin_typing import EmaAvwapMixinTyping
from .state import _AvwapSnapshot, _PositionRuntime, _SetupState, _SymbolSnapshot


class EmaAvwapDataMixin(EmaAvwapMixinTyping):
    def _fetch_latest_closed_candle(self, symbol: str) -> Candle | None:
        rows = self._fetch_strategy_klines(symbol, self._cfg.timeframe, 3)
        if rows is None:
            return None
        # Bitunix REST is treated as history only. It may contain a forming row
        # or it may stop at the latest closed row, so select by close boundary
        # rather than assuming any position in this three-row response.
        closed = self._ready_closed_candles(symbol, self._cfg.timeframe, rows)
        return closed[-1] if closed else None

    def _build_snapshot(self, symbol: str) -> _SymbolSnapshot | None:
        min_history = max(self._cfg.ema_length, self._cfg.consecutive_count) + 2
        raw = self._fetch_strategy_klines(
            symbol=symbol,
            interval=self._cfg.timeframe,
            limit=max(min_history + 1, self._cfg.max_history_bars),
        )
        if raw is None:
            return None
        closed = self._ready_closed_candles(symbol, self._cfg.timeframe, raw)
        if len(closed) < min_history:
            self._log.debug(
                "EmaAvwapPullback: insufficient ready closed klines for %s "
                "(%s < %s)",
                symbol,
                len(closed),
                min_history,
            )
            return None
        closes = [candle.close for candle in closed]
        ema_values = calc_ema(closes, self._cfg.ema_length)
        idx = len(closed) - 1
        ema_value = ema_values[idx]
        if ema_value is None:
            return None
        tpv_prefix, vol_prefix, tpv2_prefix = self._build_avwap_prefixes(closed)
        return _SymbolSnapshot(
            symbol=symbol,
            timeframe=self._cfg.timeframe,
            timeframe_minutes=self._interval_to_minutes(self._cfg.timeframe),
            candles=closed,
            candle_index=idx,
            candle=closed[idx],
            previous_candle=closed[idx - 1],
            ema_value=float(ema_value),
            tpv_prefix=tpv_prefix,
            vol_prefix=vol_prefix,
            tpv2_prefix=tpv2_prefix,
        )

    def _build_live_avwap_snapshot(
        self,
        snapshot: _SymbolSnapshot,
        setup: _SetupState | _PositionRuntime,
    ) -> Tuple[_SymbolSnapshot, _AvwapSnapshot]:
        candles = self._live_avwap_candles(snapshot)
        if candles[-1].open_time <= snapshot.candle.open_time:
            raise ValueError(
                "latest forming candle is unavailable; refusing to use stale "
                "closed-candle indicators"
            )
        tpv_prefix, vol_prefix, tpv2_prefix = self._build_avwap_prefixes(candles)
        ema_values = calc_ema(
            [candle.close for candle in candles], self._cfg.ema_length
        )
        live_ema = ema_values[-1]
        live_snapshot = _SymbolSnapshot(
            symbol=snapshot.symbol,
            timeframe=snapshot.timeframe,
            timeframe_minutes=snapshot.timeframe_minutes,
            candles=candles,
            candle_index=len(candles) - 1,
            candle=candles[-1],
            previous_candle=candles[-2],
            ema_value=(
                float(live_ema) if live_ema is not None else snapshot.ema_value
            ),
            tpv_prefix=tpv_prefix,
            vol_prefix=vol_prefix,
            tpv2_prefix=tpv2_prefix,
        )
        anchor_index = self._find_anchor_index(candles, setup)
        avwap = self._build_avwap_snapshot(
            candles=candles,
            anchor_index=anchor_index,
            candle_index=live_snapshot.candle_index,
            tpv_prefix=tpv_prefix,
            vol_prefix=vol_prefix,
            tpv2_prefix=tpv2_prefix,
        )
        return live_snapshot, avwap

    def _live_avwap_candles(self, snapshot: _SymbolSnapshot) -> Tuple[Candle, ...]:
        forming = self._fresh_forming_kline(snapshot.symbol)
        expected_open_ms = (
            snapshot.candle.open_time_ms
            + interval_to_milliseconds(snapshot.timeframe)
        )
        if forming is None:
            # Do not turn a live signal or exit check into a blocking REST
            # request. The REST history endpoint cannot guarantee a forming
            # candle, so proceeding without a fresh WebSocket candle is unsafe.
            return tuple(snapshot.candles)
        if forming.open_time_ms <= snapshot.candle.open_time_ms:
            return tuple(snapshot.candles)
        if forming.open_time_ms > expected_open_ms:
            self._log.warning(
                "EmaAvwapPullback: live AVWAP stream for %s has a candle gap "
                "after %s; using closed snapshot until the bar snapshot is rebuilt",
                snapshot.symbol,
                snapshot.candle.close_time.isoformat(),
            )
            return tuple(snapshot.candles)
        return tuple(snapshot.candles) + (forming,)

    def _fetch_strategy_klines(
        self, symbol: str, interval: str, limit: int
    ) -> Optional[List[List]]:
        try:
            rows = self._exchange.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit,
            )
            if rows:
                return rows
            self._log.warning(
                "EmaAvwapPullback: Bitunix get_klines returned no data for %s (%s)",
                symbol,
                interval,
            )
        except Exception as exc:
            self._log.warning(
                "EmaAvwapPullback: Bitunix get_klines failed for %s (%s): %s",
                symbol,
                interval,
                exc,
            )
        return None

    def _ready_closed_candles(
        self,
        symbol: str,
        interval: str,
        rows: List[List],
        *,
        now: datetime | None = None,
    ) -> list[Candle]:
        """Return only bars whose close and configured readiness delay passed.

        Bitunix can return an already-closed last row, or a still-forming last
        row.  Selecting by time instead of by position avoids assuming that
        ``rows[-2]`` is always the most recent stable bar.
        """
        current_time = now or datetime.now(tz=timezone.utc)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        else:
            current_time = current_time.astimezone(timezone.utc)
        interval_ms = interval_to_milliseconds(interval)
        ready_delay = timedelta(seconds=self._cfg.candle_ready_delay_seconds)
        closed: list[Candle] = []
        for row in rows:
            try:
                candle = Candle.from_binance(symbol, interval, row)
            except (IndexError, TypeError, ValueError) as exc:
                self._log.warning(
                    "EmaAvwapPullback: skipping malformed Bitunix kline for %s "
                    "(%s): %s",
                    symbol,
                    interval,
                    exc,
                )
                continue
            close_boundary = candle.open_time + timedelta(milliseconds=interval_ms)
            if current_time >= close_boundary + ready_delay:
                closed.append(candle)
        return sorted(closed, key=lambda candle: candle.open_time)

    # ------------------------------------------------------------------
    # Setup / entry signal state machine
