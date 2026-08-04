"""Candle fetching and snapshot construction for EMA + AVWAP."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

from candle_downloader.binance import MAX_BATCH, interval_to_milliseconds
from candle_downloader.models import Candle

from backtest.indicators import ema as calc_ema

from ._mixin_typing import EmaAvwapMixinTyping
from .state import _AvwapSnapshot, _PositionRuntime, _SetupState, _SymbolSnapshot


class EmaAvwapDataMixin(EmaAvwapMixinTyping):
    def _fetch_latest_closed_candle(self, symbol: str) -> Candle | None:
        candles = self._fetch_binance_signal_candles(symbol, self._cfg.timeframe, 3)
        # Binance returns the active row as well as closed rows. Select by the
        # close boundary rather than assuming a position in the response.
        closed = self._ready_closed_signal_candles(candles)
        return closed[-1] if closed else None

    def _build_snapshot(self, symbol: str) -> _SymbolSnapshot | None:
        min_history = max(self._cfg.ema_length, self._cfg.consecutive_count) + 2
        requested_limit = max(min_history + 1, self._cfg.max_history_bars + 1)
        history_limit = min(requested_limit, MAX_BATCH)
        if history_limit < requested_limit:
            self._log.warning(
                "EmaAvwapPullback: Binance kline limit caps %s at %s "
                "closed history rows",
                symbol,
                MAX_BATCH - 1,
            )
        candles = self._fetch_binance_signal_candles(
            symbol=symbol,
            interval=self._cfg.timeframe,
            limit=history_limit,
        )
        closed = self._ready_closed_signal_candles(candles)
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
        forming = self._fresh_forming_binance_candle(snapshot.symbol)
        expected_open_ms = (
            snapshot.candle.open_time_ms
            + interval_to_milliseconds(snapshot.timeframe)
        )
        if forming is None:
            # Forming candles are fetched asynchronously. Do not turn a live
            # signal or exit check into a blocking REST request; fail closed
            # until a current Binance candle has been validated and cached.
            return tuple(snapshot.candles)
        if forming.open_time_ms <= snapshot.candle.open_time_ms:
            return tuple(snapshot.candles)
        if forming.open_time_ms > expected_open_ms:
            self._log.warning(
                "EmaAvwapPullback: Binance live AVWAP feed for %s has a candle gap "
                "after %s; using closed snapshot until the bar snapshot is rebuilt",
                snapshot.symbol,
                snapshot.candle.close_time.isoformat(),
            )
            return tuple(snapshot.candles)
        return tuple(snapshot.candles) + (forming,)

    def _fetch_binance_signal_candles(
        self, symbol: str, interval: str, limit: int
    ) -> list[Candle]:
        """Fetch one self-consistent Binance candle series for EMA/AVWAP.

        Bitunix data is deliberately excluded: it is reserved for execution
        prices and order specifications, whose venue-specific price can differ
        from the Binance signal price.
        """
        try:
            candles = self._binance_signal_client.fetch_recent_klines(
                symbol=symbol,
                interval=interval,
                limit=limit,
            )
            if candles:
                return self._validate_binance_signal_candles(
                    symbol=symbol,
                    interval=interval,
                    candles=candles,
                )
            self._log.warning(
                "EmaAvwapPullback: Binance kline API returned no data for %s (%s)",
                symbol,
                interval,
            )
        except Exception as exc:
            self._log.warning(
                "EmaAvwapPullback: Binance kline API failed for %s (%s): %s",
                symbol,
                interval,
                exc,
            )
        return []

    def _validate_binance_signal_candles(
        self,
        *,
        symbol: str,
        interval: str,
        candles: list[Candle],
    ) -> list[Candle]:
        """Reject malformed, duplicate, or gapped candles before signalling."""
        interval_ms = interval_to_milliseconds(interval)
        ordered = sorted(candles, key=lambda candle: candle.open_time_ms)
        previous_open_ms: int | None = None
        expected_symbol = symbol.strip().upper()
        for candle in ordered:
            values = (candle.open, candle.high, candle.low, candle.close, candle.volume)
            valid_ohlcv = (
                all(math.isfinite(value) for value in values)
                and candle.open > 0
                and candle.high > 0
                and candle.low > 0
                and candle.close > 0
                and candle.volume >= 0
                and candle.low <= min(candle.open, candle.close)
                and candle.high >= max(candle.open, candle.close)
            )
            expected_close_ms = candle.open_time_ms + interval_ms - 1
            # ``datetime.timestamp()`` is a float. Flooring can turn a valid
            # ``...:59.999`` Binance close into ``...:59.998`` on some
            # platforms, rejecting an otherwise canonical completed candle.
            # Round to the millisecond that the venue actually supplies.
            actual_close_ms = round(candle.close_time.timestamp() * 1000)
            if (
                candle.symbol != expected_symbol
                or candle.interval != interval
                or actual_close_ms != expected_close_ms
                or not valid_ohlcv
                or (
                    previous_open_ms is not None
                    and candle.open_time_ms != previous_open_ms + interval_ms
                )
            ):
                self._log.warning(
                    "EmaAvwapPullback: rejected invalid Binance signal candle series "
                    "for %s (%s)",
                    symbol,
                    interval,
                )
                return []
            previous_open_ms = candle.open_time_ms
        return ordered

    def _ready_closed_signal_candles(
        self,
        candles: list[Candle],
        *,
        now: datetime | None = None,
    ) -> list[Candle]:
        """Return ready closed candles from an already-validated Binance feed."""
        current_time = self._as_utc(now)
        interval_ms = interval_to_milliseconds(self._cfg.timeframe)
        ready_delay = timedelta(seconds=self._cfg.candle_ready_delay_seconds)
        return [
            candle
            for candle in candles
            if current_time
            >= candle.open_time + timedelta(milliseconds=interval_ms) + ready_delay
        ]

    def _fetch_latest_forming_binance_candle(
        self,
        symbol: str,
        *,
        now: datetime | None = None,
    ) -> Candle | None:
        """Return only the current Binance candle; never reuse a closed bar."""
        current_time = self._as_utc(now)
        interval_ms = interval_to_milliseconds(self._cfg.timeframe)
        expected_open_ms = (
            int(current_time.timestamp() * 1000) // interval_ms
        ) * interval_ms
        candles = self._fetch_binance_signal_candles(
            symbol=symbol,
            interval=self._cfg.timeframe,
            limit=2,
        )
        for candle in reversed(candles):
            if candle.open_time_ms == expected_open_ms:
                return candle
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

        A Binance-compatible REST response can return an already-closed last
        row, or a still-forming last row. Selecting by time avoids assuming that
        ``rows[-2]`` is always the most recent stable bar.
        """
        current_time = self._as_utc(now)
        interval_ms = interval_to_milliseconds(interval)
        ready_delay = timedelta(seconds=self._cfg.candle_ready_delay_seconds)
        closed: list[Candle] = []
        for row in rows:
            try:
                candle = Candle.from_binance(symbol, interval, row)
            except (IndexError, TypeError, ValueError) as exc:
                self._log.warning(
                    "EmaAvwapPullback: skipping malformed Binance-compatible kline for %s "
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

    @staticmethod
    def _as_utc(now: datetime | None) -> datetime:
        current_time = now or datetime.now(tz=timezone.utc)
        if current_time.tzinfo is None:
            return current_time.replace(tzinfo=timezone.utc)
        return current_time.astimezone(timezone.utc)

    # ------------------------------------------------------------------
    # Setup / entry signal state machine
