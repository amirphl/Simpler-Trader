"""Shared data-access utilities for live trading strategies."""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from json import loads
from typing import List
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener

from candle_downloader.models import Candle
from candle_downloader.storage import CandleStore, build_store

from .exchange import Exchange
from .models import LiveTradingConfig, TradingState


class BaseLiveTradingStrategy:
    """Shared helper methods for live trading strategies."""

    def __init__(
        self,
        config: LiveTradingConfig,
        exchange: Exchange,
        state: TradingState,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._exchange = exchange
        self._state = state
        self._log = logger or logging.getLogger(__name__)
        self._thread_local = threading.local()
        self._opener_lock = threading.Lock()
        self._binance_opener = build_opener(*self._build_proxy_handlers())
        self._one_min_cache: dict[str, List[Candle]] = {}
        self._cache_lock = threading.Lock()
        self._candle_store: CandleStore | None = None

    def close(self) -> None:
        """Close any owned resources."""
        if self._candle_store is not None:
            try:
                self._candle_store.close()
            except Exception:
                pass

    def _get_candle_store(self) -> CandleStore:
        if self._candle_store is None:
            self._candle_store = build_store("postgres", self._config.klines_db)
            if self._config.klines_db.suffix != ".env":
                self._log.info(
                    "Live strategy uses postgres candle store; --klines-db is treated as optional env file path."
                )
        return self._candle_store

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def _fetch_binance_klines(
        self, symbol: str, interval: str, limit: int
    ) -> List[list]:
        """Fetch klines directly from Binance with retry and backup hosts."""
        params = urlencode({"symbol": symbol, "interval": interval, "limit": limit})
        base_urls = [
            "https://api.binance.com",
            "https://api1.binance.com",
            "https://api-gcp.binance.com",
            "https://api2.binance.com",
        ]

        for base in base_urls:
            url = f"{base}/api/v3/klines?{params}"
            delay = 1.0
            for attempt in range(1, 4):
                try:
                    opener = self._get_thread_opener()
                    with opener.open(Request(url), timeout=10) as resp:
                        klines = loads(resp.read())
                        return klines
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

        self._log.error("All Binance kline hosts failed for %s", symbol)
        return []

    def _fetch_binance_klines_window(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1m",
        limit: int = 1000,
    ) -> List[list]:
        """Fetch klines in a time window, paging until end_time."""
        base_urls = [
            "https://api.binance.com",
            "https://api1.binance.com",
            "https://api-gcp.binance.com",
            "https://api2.binance.com",
        ]
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        results: List[list] = []

        next_start = start_ms
        while next_start < end_ms:
            params = urlencode(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": next_start,
                    "endTime": end_ms,
                    "limit": limit,
                }
            )
            success = False
            for base in base_urls:
                url = f"{base}/api/v3/klines?{params}"
                delay = 1.0
                for attempt in range(1, 4):
                    try:
                        opener = self._get_thread_opener()
                        with opener.open(Request(url), timeout=10) as resp:
                            batch = loads(resp.read())
                            if not batch:
                                return results
                            results.extend(batch)
                            last_close = int(batch[-1][6])
                            if last_close <= next_start:
                                return results
                            next_start = last_close + 1
                            success = True
                            if len(batch) < limit or last_close >= end_ms:
                                return results
                            break
                    except (HTTPError, URLError) as exc:
                        self._log.warning(
                            "Binance window klines attempt %s host=%s symbol=%s failed: %s",
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
                if success:
                    break
            if not success:
                self._log.error("Failed to fetch window klines for %s", symbol)
                break
        return results

    def _get_1m_candles(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> List[Candle]:
        """Return cached 1m candles, fetching missing segments and persisting to postgres."""
        with self._cache_lock:
            cache = list(self._one_min_cache.get(symbol, []))

        cache = [c for c in cache if c.close_time >= start_time - timedelta(minutes=5)]
        thirty_days_ago = end_time - timedelta(days=30)
        cache = [c for c in cache if c.close_time >= thirty_days_ago]

        if not cache:
            cache = self._load_1m_from_db(symbol, start_time, end_time)

        last_cached_close = cache[-1].close_time if cache else None
        fetch_start = start_time
        if last_cached_close and last_cached_close > fetch_start:
            fetch_start = last_cached_close + timedelta(milliseconds=1)

        if fetch_start < end_time:
            raw = self._fetch_binance_klines_window(symbol, fetch_start, end_time)
            new_candles = [
                Candle.from_binance(symbol, "1m", payload) for payload in raw
            ]
            if cache:
                last_open = cache[-1].open_time
                new_candles = [c for c in new_candles if c.open_time > last_open]
            if new_candles:
                cache.extend(new_candles)
                self._persist_1m_to_db(new_candles)

        cache_sorted = cache
        with self._cache_lock:
            self._one_min_cache[symbol] = cache_sorted

        now = datetime.now(timezone.utc)
        return [
            c
            for c in cache_sorted
            if c.open_time >= start_time
            and c.close_time <= end_time
            and c.close_time <= now
        ]

    def _persist_1m_to_db(self, candles: List[Candle]) -> None:
        """Persist 1m candles to postgres candle store."""
        if not candles:
            return
        self._get_candle_store().save(candles)

    def _load_1m_from_db(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> List[Candle]:
        """Load 1m candles from postgres candle store."""
        return self._get_candle_store().load(symbol, "1m", start_time, end_time)

    def _aggregate_candles(
        self,
        minute_candles: List[Candle],
        start_boundary: datetime,
        end_boundary: datetime,
        timeframe_minutes: int,
    ) -> List[Candle]:
        """Aggregate 1m candles into higher timeframe bars."""
        aggregated_rev: List[Candle] = []
        window = timedelta(minutes=timeframe_minutes)

        minute_sorted = sorted(minute_candles, key=lambda c: c.open_time)
        total = len(minute_sorted)
        idx = total - 1

        cursor_end = end_boundary
        cursor_start = end_boundary - window

        while cursor_start >= start_boundary:
            bucket: List[Candle] = []
            while idx >= 0 and minute_sorted[idx].open_time >= cursor_start:
                if minute_sorted[idx].open_time < cursor_end:
                    bucket.append(minute_sorted[idx])
                idx -= 1

            if bucket:
                bucket.reverse()
                open_price = bucket[0].open
                close_price = bucket[-1].close
                high_price = max(c.high for c in bucket)
                low_price = min(c.low for c in bucket)
                volume = sum(c.volume for c in bucket)
                aggregated_rev.append(
                    Candle(
                        symbol=bucket[0].symbol,
                        interval=f"{timeframe_minutes}m",
                        open_time=cursor_start,
                        close_time=cursor_end,
                        open=open_price,
                        high=high_price,
                        low=low_price,
                        close=close_price,
                        volume=volume,
                    )
                )

            cursor_end = cursor_start
            cursor_start = cursor_start - window

        return list(reversed(aggregated_rev))

    def _interval_to_minutes(self, interval: str) -> int:
        mapping = {
            "1m": 1,
            "3m": 3,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "2h": 120,
            "4h": 240,
            "6h": 360,
            "12h": 720,
            "1d": 1440,
        }
        return mapping.get(interval, 0)

    def _build_proxy_handlers(self):
        """Reuse environment proxy settings for Binance HTTP calls."""
        http_env = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        https_env = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        all_env = os.getenv("ALL_PROXY") or os.getenv("all_proxy")

        proxies = {}
        if all_env:
            proxies["http"] = all_env
            proxies["https"] = all_env
        else:
            if http_env:
                proxies["http"] = http_env
            if https_env:
                proxies["https"] = https_env

        return [ProxyHandler(proxies)] if proxies else []

    def _get_thread_opener(self):
        """Return a thread-local opener to avoid cross-thread races."""
        local = getattr(self, "_thread_local", None)
        if local is None:
            local = threading.local()
            self._thread_local = local
        if getattr(local, "opener", None) is None:
            handlers = self._build_proxy_handlers()
            local.opener = build_opener(*handlers)
        return local.opener
