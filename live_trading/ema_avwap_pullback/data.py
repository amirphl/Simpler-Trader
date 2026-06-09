"""Candle fetching and snapshot construction for EMA + AVWAP."""

from __future__ import annotations

import time
from json import loads
from typing import List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request

from candle_downloader.binance import interval_to_milliseconds
from candle_downloader.models import Candle

from backtest.indicators import ema as calc_ema

from .constants import KLINES_RETRIES, KLINES_RETRY_DELAY_SECONDS
from ._mixin_typing import EmaAvwapMixinTyping
from .state import _AvwapSnapshot, _SetupState, _SymbolSnapshot


class EmaAvwapDataMixin(EmaAvwapMixinTyping):
    def _fetch_latest_closed_candle(self, symbol: str) -> Candle | None:
        rows = self._fetch_strategy_klines(symbol, self._cfg.timeframe, 3)
        if rows is None or len(rows) < 2:
            return None
        return Candle.from_binance(symbol, self._cfg.timeframe, rows[-2])

    def _build_snapshot(self, symbol: str) -> _SymbolSnapshot | None:
        min_history = max(self._cfg.ema_length, self._cfg.consecutive_count) + 2
        raw = self._fetch_strategy_klines(
            symbol=symbol,
            interval=self._cfg.timeframe,
            limit=max(min_history + 1, self._cfg.max_history_bars),
        )
        if raw is None:
            return None
        if len(raw) < min_history:
            self._log.debug(
                "EmaAvwapPullback: insufficient klines for %s (%s < %s)",
                symbol,
                len(raw),
                min_history,
            )
            return None

        candles = [Candle.from_binance(symbol, self._cfg.timeframe, row) for row in raw]
        closed = candles[:-1]
        if len(closed) < min_history:
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
        self, snapshot: _SymbolSnapshot, setup: _SetupState
    ) -> Tuple[_SymbolSnapshot, _AvwapSnapshot]:
        candles = self._live_avwap_candles(snapshot)
        tpv_prefix, vol_prefix, tpv2_prefix = self._build_avwap_prefixes(candles)
        live_snapshot = _SymbolSnapshot(
            symbol=snapshot.symbol,
            timeframe=snapshot.timeframe,
            timeframe_minutes=snapshot.timeframe_minutes,
            candles=candles,
            candle_index=len(candles) - 1,
            candle=candles[-1],
            previous_candle=candles[-2],
            ema_value=snapshot.ema_value,
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
        tail_limit = min(
            max(self._cfg.consecutive_count + 3, 10),
            self._cfg.max_history_bars,
        )
        raw = self._fetch_strategy_klines(
            symbol=snapshot.symbol,
            interval=snapshot.timeframe,
            limit=tail_limit,
        )
        if raw is None:
            return tuple(snapshot.candles)

        tail = [
            Candle.from_binance(snapshot.symbol, snapshot.timeframe, row) for row in raw
        ]
        newer = [
            candle for candle in tail if candle.open_time > snapshot.candle.open_time
        ]
        if not newer:
            return tuple(snapshot.candles)

        interval_ms = interval_to_milliseconds(snapshot.timeframe)
        expected_open_ms = snapshot.candle.open_time_ms + interval_ms
        appended: list[Candle] = []
        for candle in sorted(newer, key=lambda item: item.open_time):
            if candle.open_time_ms < expected_open_ms:
                continue
            if candle.open_time_ms > expected_open_ms:
                self._log.warning(
                    "EmaAvwapPullback: live AVWAP tail for %s has a candle gap "
                    "after %s; using closed snapshot until the bar snapshot is rebuilt",
                    snapshot.symbol,
                    snapshot.candle.close_time.isoformat(),
                )
                break
            appended.append(candle)
            expected_open_ms += interval_ms

        if not appended:
            return tuple(snapshot.candles)
        return tuple(snapshot.candles) + tuple(appended)

    def _fetch_strategy_klines(
        self, symbol: str, interval: str, limit: int
    ) -> Optional[List[List]]:
        last_exc: Optional[Exception] = None
        retries = max(int(getattr(self._cfg, "api_retries", KLINES_RETRIES)), 1)
        retry_delay = max(
            float(
                getattr(
                    self._cfg,
                    "api_retry_delay_seconds",
                    KLINES_RETRY_DELAY_SECONDS,
                )
            ),
            0.0,
        )
        for attempt in range(1, retries + 1):
            try:
                rows = self._exchange.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit,
                )
                if rows:
                    return rows
                self._log.warning(
                    "EmaAvwapPullback: exchange.get_klines returned no data for "
                    "%s (%s) (attempt %d/%d)",
                    symbol,
                    interval,
                    attempt,
                    retries,
                )
            except Exception as exc:
                last_exc = exc
                self._log.warning(
                    "EmaAvwapPullback: exchange.get_klines raised for %s (%s) "
                    "(attempt %d/%d): %s",
                    symbol,
                    interval,
                    attempt,
                    retries,
                    exc,
                )
            if attempt < retries and retry_delay > 0:
                time.sleep(retry_delay)

        self._log.info(
            "EmaAvwapPullback: falling back to Binance klines for %s (%s)",
            symbol,
            interval,
        )
        try:
            rows = self._fetch_binance_klines(symbol, interval, limit)
            if rows:
                return rows
        except Exception as exc:
            self._log.warning(
                "EmaAvwapPullback: Binance fallback failed for %s (%s): %s",
                symbol,
                interval,
                exc,
            )

        self._log.error(
            "EmaAvwapPullback: all klines fetch attempts failed for %s (%s). "
            "Last primary error: %s",
            symbol,
            interval,
            last_exc,
        )
        return None

    def _fetch_binance_klines(
        self, symbol: str, interval: str, limit: int
    ) -> List[list]:
        params = urlencode({"symbol": symbol, "interval": interval, "limit": limit})
        base_urls = (
            "https://api.binance.com",
            "https://api1.binance.com",
            "https://api-gcp.binance.com",
            "https://api2.binance.com",
        )
        for base in base_urls:
            url = f"{base}/api/v3/klines?{params}"
            delay = 1.0
            for attempt in range(1, 4):
                try:
                    opener = self._get_binance_opener()
                    with opener.open(Request(url), timeout=10) as resp:
                        rows = loads(resp.read())
                        return rows
                except (HTTPError, URLError) as exc:
                    self._log.warning(
                        "Binance klines attempt %s host=%s symbol=%s failed: %s",
                        attempt,
                        base,
                        symbol,
                        exc,
                    )
                    if attempt < 3:
                        time.sleep(delay)
                        delay = min(delay * 2, 8.0)
                except Exception as exc:
                    self._log.warning(
                        "Error parsing Binance klines host=%s symbol=%s: %s",
                        base,
                        symbol,
                        exc,
                    )
                    break
        return []

    # ------------------------------------------------------------------
    # Setup / entry signal state machine
