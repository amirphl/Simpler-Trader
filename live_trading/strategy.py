"""Live trading strategy implementation."""

from __future__ import annotations

import logging
import os
import time
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from json import loads
from typing import List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener

from candle_downloader.models import Candle

from .exchange import Exchange, PositionSide
from .heiken_ashi import calculate_heiken_ashi, detect_reversal_signal
from .models import LiveTradingConfig, SymbolInfo, TradingSignal, TradingState


class LiveTradingStrategy:
    """Live trading strategy based on Heiken Ashi reversals."""

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
        self._db_lock = threading.Lock()
        self._klines_db_path = str(self._config.klines_db)
        self._ensure_kline_table()

    def close(self) -> None:
        """Close any owned resources."""
        try:
            conn = getattr(self._thread_local, "db", None)
            if conn:
                conn.close()
        except Exception:
            pass

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def _ensure_kline_table(self) -> None:
        """Create table for cached klines if not exists."""
        conn = sqlite3.connect(self._klines_db_path, timeout=30.0)
        with self._db_lock, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS klines (
                    symbol TEXT NOT NULL,
                    open_time_ms INTEGER NOT NULL,
                    close_time_ms INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    PRIMARY KEY(symbol, open_time_ms)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_klines_symbol_time ON klines(symbol, open_time_ms)"
            )
        conn.close()

    def _get_db(self) -> sqlite3.Connection:
        """Return a thread-local sqlite connection."""
        if getattr(self._thread_local, "db", None) is None:
            self._thread_local.db = sqlite3.connect(
                self._klines_db_path, check_same_thread=False, timeout=30.0
            )
        return self._thread_local.db

    def analyze_symbol(
        self,
        symbol_info: SymbolInfo,
        current_time: datetime,
    ) -> Optional[TradingSignal]:
        """Analyze a symbol and generate trading signal if conditions are met.

        Args:
            symbol_info: Information about the symbol
            current_time: Current timestamp

        Returns:
            TradingSignal if conditions are met, None otherwise
        """
        symbol = symbol_info.symbol

        # Check if symbol is disabled
        if self._state.is_symbol_disabled(
            symbol, current_time, self._config.disable_symbol_hours
        ):
            self._log.debug(f"Symbol {symbol} is disabled")
            return None

        # Check if position already exists
        if symbol in self._state.active_positions:
            self._log.debug(f"Position already exists for {symbol}")
            return None

        try:
            # Fetch recent candles for Heiken Ashi calculation
            # We need at least W+1 candles for reversal detection
            required_candles = self._config.heiken_ashi_candles_before + 1
            # Pull candles from Binance directly to ensure consistent format/data
            # Fetch atleast 20 candles to enhance Heiken Ashi calculation accuracy
            klines = self._fetch_binance_klines(
                symbol=symbol,
                interval=self._config.timeframe,
                limit=max(required_candles * 2, 20) + 1,  # extra for dropping last
            )

            if len(klines) < required_candles:
                self._log.debug(
                    f"Insufficient candles for {symbol}: {len(klines)} < {required_candles}"
                )
                return None

            # Convert to Candle objects
            candles = [
                Candle.from_binance(symbol, self._config.timeframe, kline)
                for kline in klines
            ]

            current_candle = candles[-1]
            candles_excluding_last_one = candles[
                :-1
            ]  # Drop last candle to avoid in-progress data

            # Calculate Heiken Ashi candles
            ha_candles_excluding_last_one = calculate_heiken_ashi(
                candles_excluding_last_one
            )

            last_closed_candle = candles_excluding_last_one[-1]
            last_closed_ha_candle = ha_candles_excluding_last_one[-1]
            self._log.debug(
                "%s latest original candle (%s): open=%.6f, high=%.6f, low=%.6f, close=%.6f",
                symbol,
                "Green" if last_closed_candle.is_bullish() else "Red",
                last_closed_candle.open,
                last_closed_candle.high,
                last_closed_candle.low,
                last_closed_candle.close,
            )
            self._log.debug(
                "%s latest HA candle (%s): open=%.6f, high=%.6f, low=%.6f, close=%.6f",
                symbol,
                "Green" if last_closed_ha_candle.is_bullish() else "Red",
                last_closed_ha_candle.ha_open,
                last_closed_ha_candle.ha_high,
                last_closed_ha_candle.ha_low,
                last_closed_ha_candle.ha_close,
            )
            self._log.debug(
                "%s HA candle[-2] (%s): open=%.6f, high=%.6f, low=%.6f, close=%.6f",
                symbol,
                "Green" if ha_candles_excluding_last_one[-2].is_bullish() else "Red",
                ha_candles_excluding_last_one[-2].ha_open,
                ha_candles_excluding_last_one[-2].ha_high,
                ha_candles_excluding_last_one[-2].ha_low,
                ha_candles_excluding_last_one[-2].ha_close,
            )
            self._log.debug(
                "%s HA candle[-3] (%s): open=%.6f, high=%.6f, low=%.6f, close=%.6f",
                symbol,
                "Green" if ha_candles_excluding_last_one[-3].is_bullish() else "Red",
                ha_candles_excluding_last_one[-3].ha_open,
                ha_candles_excluding_last_one[-3].ha_high,
                ha_candles_excluding_last_one[-3].ha_low,
                ha_candles_excluding_last_one[-3].ha_close,
            )

            if len(ha_candles_excluding_last_one) < required_candles:
                return None

            # Detect reversal signal
            signal_type = detect_reversal_signal(
                last_closed_candle,
                ha_candles_excluding_last_one,
                lookback_candles=self._config.heiken_ashi_candles_before,
            )

            if signal_type is None:
                return None

            # Determine position side based on price movement direction
            if signal_type == "LONG":
                # Price was falling (top losers), now reversing up
                # if symbol_info.price_change_pct >= 0: # TODO
                #     return None  # Symbol should be from losers list
                side = PositionSide.LONG
            else:  # SHORT
                # Price was rising (top gainers), now reversing down
                # if symbol_info.price_change_pct <= 0: # TODO
                #     return None  # Symbol should be from gainers list
                side = PositionSide.SHORT

            # Entry price is current market price
            entry_price = current_candle.close

            # # Stop loss is the open price of HA candle
            # stop_loss = last_closed_ha_candle.ha_open

            stop_loss = None

            # Calculate take profit
            if side == PositionSide.LONG:
                take_profit = entry_price * (1 + self._config.take_profit_pct / 100)
            else:
                take_profit = entry_price * (1 - self._config.take_profit_pct / 100)

            # Validate signal
            if side == PositionSide.LONG:
                if stop_loss is not None and stop_loss >= entry_price:
                    fallback_sl = entry_price * 0.99
                    self._log.warning(
                        "Invalid LONG stop loss for %s: stop_loss (%.6f) >= entry (%.6f); "
                        "using fallback 1%% below entry: %.6f",
                        symbol,
                        stop_loss,
                        entry_price,
                        fallback_sl,
                    )
                    stop_loss = fallback_sl
            else:
                if stop_loss is not None and stop_loss <= entry_price:
                    fallback_sl = entry_price * 0.99
                    self._log.warning(
                        "Invalid SHORT stop loss for %s: stop_loss (%.6f) <= entry (%.6f); "
                        "using fallback 1%% below entry: %.6f",
                        symbol,
                        stop_loss,
                        entry_price,
                        fallback_sl,
                    )
                    stop_loss = fallback_sl

            # Create signal
            signal = TradingSignal(
                timestamp=current_time,
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=self._config.leverage,
                margin_mode=self._config.margin_mode,
                reason=f"HA reversal: {signal_type} after {self._config.heiken_ashi_candles_before} candles",
                metadata={
                    "price_change_pct": symbol_info.price_change_pct,
                    "last_ha_candle": {
                        "open": last_closed_ha_candle.ha_open,
                        "high": last_closed_ha_candle.ha_high,
                        "low": last_closed_ha_candle.ha_low,
                        "close": last_closed_ha_candle.ha_close,
                    },
                },
            )

            self._log.info(
                f"Signal generated: {symbol} {side.value} @ {entry_price:.6f}, "
                f"SL: {stop_loss:.6f}, TP: {take_profit:.6f}"
            )

            return signal

        except Exception as e:
            self._log.error(f"Error analyzing {symbol}: {e}", exc_info=True)
            return None

    def analyze_symbol_v2(
        self,
        symbol_info: SymbolInfo,
        current_time: datetime,
    ) -> Optional[TradingSignal]:
        """Analyze a symbol using reconstructed candles from 1m klines."""
        symbol = symbol_info.symbol
        timeframe_minutes = self._interval_to_minutes(self._config.timeframe)
        if timeframe_minutes <= 0:
            self._log.error(
                "Invalid timeframe minutes for %s: %s", symbol, self._config.timeframe
            )
            return None

        required_candles = max(self._config.heiken_ashi_candles_before + 1, 3)
        lookback_periods = max(
            required_candles + 1, 20 + 1
        )  # fetch at least 21 periods

        end_boundary = current_time.replace(
            second=0, microsecond=0, tzinfo=timezone.utc
        )
        total_minutes = timeframe_minutes * lookback_periods
        start_boundary = end_boundary - timedelta(minutes=total_minutes)

        try:
            minute_candles = self._get_1m_candles(symbol, start_boundary, end_boundary)
            if len(minute_candles) == 0:
                self._log.warning("No 1m klines fetched for %s", symbol)
                return None

            aggregated = self._aggregate_candles(
                minute_candles,
                start_boundary=start_boundary,
                end_boundary=end_boundary,
                timeframe_minutes=timeframe_minutes,
            )

            if len(aggregated) < required_candles:
                self._log.debug(
                    "Insufficient aggregated candles for %s: got %s need %s",
                    symbol,
                    len(aggregated),
                    required_candles,
                )
                return None

            # Drop the most recent candle to avoid any partial period
            # candles = aggregated[:-1]
            # if len(candles) < required_candles:
            #     return None
            candles = aggregated

            ha_candles = calculate_heiken_ashi(candles)
            last_running_candle = candles[-1]
            last_running_ha_candle = ha_candles[-1]

            # Log last running candle
            self._log.info(
                "%s latest running candle: open=%.6f, high=%.6f, low=%.6f, close=%.6f",
                symbol,
                last_running_candle.open,
                last_running_candle.high,
                last_running_candle.low,
                last_running_candle.close,
            )

            signal_type = detect_reversal_signal(
                last_running_candle,
                ha_candles,
                lookback_candles=self._config.heiken_ashi_candles_before,
            )
            if signal_type is None:
                return None

            side = PositionSide.LONG if signal_type == "LONG" else PositionSide.SHORT
            entry_price = last_running_candle.close
            stop_loss = last_running_ha_candle.ha_open

            if side == PositionSide.LONG and stop_loss >= entry_price:
                stop_loss = entry_price * 0.99
            elif side == PositionSide.SHORT and stop_loss <= entry_price:
                stop_loss = entry_price * 0.99

            if side == PositionSide.LONG:
                take_profit = entry_price * (1 + self._config.take_profit_pct / 100)
            else:
                take_profit = entry_price * (1 - self._config.take_profit_pct / 100)

            signal = TradingSignal(
                timestamp=current_time,
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=self._config.leverage,
                margin_mode=self._config.margin_mode,
                reason=f"HA reversal v2: {signal_type} after {self._config.heiken_ashi_candles_before} candles",
                metadata={
                    "timeframe": self._config.timeframe,
                    "source": "1m_reconstructed",
                    "price_change_pct": symbol_info.price_change_pct,
                },
            )

            self._log.info(
                "V2 Signal generated: %s %s @ %.6f, SL: %.6f, TP: %.6f, Timestamp: %s",
                symbol,
                side.value,
                entry_price,
                stop_loss,
                take_profit,
                signal.timestamp.isoformat(),
            )

            return signal
        except Exception as exc:
            self._log.error(
                "Error in analyze_symbol_v2 for %s: %s", symbol, exc, exc_info=True
            )
            return None

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
                    break  # parsing error unlikely to succeed on retry for same host

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
                        with opener.open(
                            Request(url), timeout=10
                        ) as resp:
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
        """Return cached 1m candles, fetching missing segments and persisting to sqlite."""
        # Read cache snapshot without holding the lock
        with self._cache_lock:
            cache = list(self._one_min_cache.get(symbol, []))

        # Prune stale entries to keep cache compact
        cache = [c for c in cache if c.close_time >= start_time - timedelta(minutes=5)]
        # Also enforce memory retention of at most ~30 days
        thirty_days_ago = end_time - timedelta(days=30)
        cache = [c for c in cache if c.close_time >= thirty_days_ago]

        # Load from sqlite if cache is empty
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

        # Cache remains sorted by construction; store updated version (guarded)
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
        """Persist 1m candles to sqlite."""
        if not candles:
            return
        rows = [
            (
                c.symbol,
                int(c.open_time.timestamp() * 1000),
                int(c.close_time.timestamp() * 1000),
                c.open,
                c.high,
                c.low,
                c.close,
                c.volume,
            )
            for c in candles
        ]
        conn = self._get_db()
        with self._db_lock, conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO klines
                (symbol, open_time_ms, close_time_ms, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def _load_1m_from_db(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> List[Candle]:
        """Load 1m candles from sqlite."""
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        conn = self._get_db()
        with self._db_lock:
            cur = conn.execute(
                """
                SELECT open_time_ms, close_time_ms, open, high, low, close, volume
                FROM klines
                WHERE symbol = ?
                  AND open_time_ms >= ?
                  AND close_time_ms <= ?
                ORDER BY open_time_ms ASC
                """,
                (symbol, start_ms, end_ms),
            )
            rows = cur.fetchall()
        candles = []
        for row in rows:
            open_ms, close_ms, o, h, l, c, v = row
            candles.append(
                Candle(
                    symbol=symbol,
                    interval="1m",
                    open_time=datetime.fromtimestamp(open_ms / 1000, tz=timezone.utc),
                    close_time=datetime.fromtimestamp(close_ms / 1000, tz=timezone.utc),
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                    volume=v,
                )
            )
        return candles

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

        # Sort once for deterministic iteration
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
                # bucket collected from end to start; reverse to get chronological
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

        # We built from end to start; return in chronological order
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

    def generate_signals(
        self,
        top_gainers: List[SymbolInfo],
        top_losers: List[SymbolInfo],
        current_time: datetime,
    ) -> Tuple[List[TradingSignal], List[TradingSignal]]:
        """Generate trading signals for top movers.

        Args:
            top_gainers: Symbols with highest positive price change
            top_losers: Symbols with highest negative price change
            current_time: Current timestamp

        Returns:
            Tuple of (long_signals, short_signals)
        """
        long_signals: List[TradingSignal] = []
        short_signals: List[TradingSignal] = []

        from concurrent.futures import ThreadPoolExecutor, as_completed

        self._log.info(f"Analyzing {len(top_losers)} top losers for LONG signals")
        self._log.info(f"Analyzing {len(top_gainers)} top gainers for SHORT signals")

        def process(symbol_info: SymbolInfo, expect_side: PositionSide) -> Optional[TradingSignal]:
            signal = self.analyze_symbol_v2(symbol_info, current_time)
            if signal is None:
                return None
            # TODO:
            # if signal.side != expect_side:
            #     return None
            return signal

        tasks = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            for si in top_losers:
                tasks.append(executor.submit(process, si, PositionSide.LONG))
            for si in top_gainers:
                tasks.append(executor.submit(process, si, PositionSide.SHORT))

            for future in as_completed(tasks):
                sig = future.result()
                if sig is None:
                    continue
                if sig.side == PositionSide.LONG:
                    long_signals.append(sig)
                else:
                    short_signals.append(sig)

        self._log.info(
            "Generated %s trading signals (%s long, %s short)",
            len(long_signals) + len(short_signals),
            len(long_signals),
            len(short_signals),
        )
        for signal in long_signals + short_signals:
            self._log.info(
                "  Signal: %s %s @ %.6f (SL: %.6f, TP: %.6f) Reason: %s",
                signal.symbol,
                signal.side.value,
                signal.entry_price,
                signal.stop_loss,
                signal.take_profit,
                signal.reason,
            )
        return long_signals, short_signals
