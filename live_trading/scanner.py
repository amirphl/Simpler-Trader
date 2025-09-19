"""Symbol scanner for finding trading opportunities."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from json import loads
from typing import List, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, ProxyHandler, build_opener

from .exchange import Exchange
from .models import SymbolInfo


_TOP_TOKENS = set(
    [
        "btc",
        "eth",
        "bnb",
        # "xrp",
        "ada",
        "sol",
        "trx",
        "doge",
        "bch",
        "hype",
        "link",
        "leo",
        "xmr",
        "xlm",
        "zec",
        "ltc",
        "sui",
        "avax",
        "hbar",
        "shib",
        "mnt",
        "ton",
        "cro",
        "uni",
        "dot",
        "tao",
        "aave",
        "aster",
        "near",
        "ena",
        "pepe",
        "paxg",
        "arb",
        "algo",
        "atom",
        "fil",
        "sei",
        "cake",
        "dash",
    ]
)


class SymbolScanner:
    """Scans symbols to find top movers based on volume and price change."""

    def __init__(
        self, exchange: Exchange, logger: logging.Logger | None = None
    ) -> None:
        self._exchange = exchange
        self._log = logger or logging.getLogger(__name__)

        proxies = self._detect_proxies()
        handlers = [ProxyHandler(proxies)] if proxies else []
        self._opener = build_opener(*handlers)
        if proxies:
            self._log.debug(f"Using proxies for Binance requests: {proxies}")

    def scan_top_symbols_by_volume(
        self,
        current_time: datetime,
        limit: int = 100,
        timeframe: str = "1h",
    ) -> List[SymbolInfo]:
        """Get top symbols by quote volume for the most recent timeframe candle.

        This fetches USDT pairs from Binance, grabs the latest closed candle for
        the provided timeframe, and ranks symbols by that candle's quote volume.

        Args:
            limit: Maximum number of symbols to return
            timeframe: Binance kline interval (e.g., "15m", "1h", "4h")

        Returns:
            List of SymbolInfo sorted by volume (descending)
        """
        timeframe_minutes = self._interval_to_minutes(timeframe)
        if timeframe_minutes <= 0:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        # end_time = current_time.astimezone(timezone.utc)
        # start_time = end_time - timedelta(minutes=timeframe_minutes)

        try:
            tickers = self._fetch_binance_tickers()
            if not tickers:
                self._log.error("No tickers returned from Binance")
                raise RuntimeError("No tickers returned from Binance")

            usdt_tickers = [
                ticker
                for ticker in tickers
                if isinstance(ticker.get("symbol"), str)
                and ticker["symbol"].endswith("USDT")
            ]
            # Reduce HTTP calls by limiting to the highest 24h quote volume first
            usdt_tickers.sort(
                key=lambda t: float(t.get("quoteVolume", 0) or 0), reverse=True
            )
            candidate_count = limit
            candidates = usdt_tickers[:candidate_count]
            self._log.debug(
                f"Fetched {len(usdt_tickers)} USDT tickers; "
                f"using top {candidate_count} candidates for timeframe {timeframe}"
            )

            symbols: List[SymbolInfo] = []
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def process_ticker(ticker: dict) -> SymbolInfo | None:
                symbol = ticker.get("symbol")
                if not symbol:
                    return None
                s = symbol.lower().replace("usdt", "")
                if s not in _TOP_TOKENS:
                    return None
                candle = self._fetch_latest_closed_candle(symbol, timeframe)
                if not candle:
                    return None

                try:
                    open_price = float(candle[1])
                    close_price = float(candle[4])
                    volume = float(candle[5])
                    quote_volume = float(candle[7])
                    price_change_pct = (
                        ((close_price - open_price) / open_price) * 100
                        if open_price > 0
                        else 0.0
                    )
                    self._log.debug(
                        "%s %s price window open=%.6f close=%.6f volume=%.6f quote_volume=%.6f",
                        symbol,
                        timeframe,
                        open_price,
                        close_price,
                        volume,
                        quote_volume,
                    )
                    return SymbolInfo(
                        symbol=symbol,
                        current_price=close_price,
                        price_change_pct=price_change_pct,
                        volume=volume,
                        quote_volume=quote_volume,
                    )
                except (ValueError, TypeError, IndexError) as e:
                    self._log.warning(f"Failed to parse candle for {symbol}: {e}")
                    return None

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(process_ticker, t) for t in candidates]
                for fut in as_completed(futures):
                    res = fut.result()
                    if res:
                        symbols.append(res)

            symbols.sort(key=lambda x: x.quote_volume, reverse=True)
            result = symbols[:limit]
            self._log.info(
                f"Scanned {len(candidates)} candidates for timeframe {timeframe}, "
                f"returning top {len(result)} by quote volume"
            )
            return result

        except Exception as e:
            self._log.error(f"Failed to scan symbols by volume: {e}")
            raise RuntimeError(f"Symbol scanning failed: {e}") from e

    def _fetch_binance_tickers(self) -> List[dict]:
        """Fetch 24h tickers from Binance."""
        url = "https://api.binance.com/api/v3/ticker/24hr"
        delay = 1.0
        attempts = 3
        last_exc: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                with self._opener.open(Request(url), timeout=10) as resp:
                    payload = resp.read()
                    return loads(payload)
            except (HTTPError, URLError) as e:
                last_exc = e
                self._log.warning(
                    "Ticker request failed (attempt %s/%s): %s", attempt, attempts, e
                )
            except Exception as e:
                last_exc = e
                self._log.warning(
                    "Ticker parsing failed (attempt %s/%s): %s", attempt, attempts, e
                )

            if attempt < attempts:
                self._log.debug("Retrying tickers in %.1fs", delay)
                time.sleep(delay)
                delay = min(delay * 2, 8.0)

        raise RuntimeError(
            f"Ticker request failed after {attempts} attempts: {last_exc}"
        ) from last_exc

    # TODO: Not tested.
    def _fetch_latest_candle(self, symbol: str, interval: str) -> list | None:
        """Fetch the latest closed kline for a symbol."""
        LAST_INDEX = -1
        params = urlencode({"symbol": symbol, "interval": interval, "limit": 1})
        url = f"https://api.binance.com/api/v3/klines?{params}"
        delay = 1.0
        attempts = 3
        last_exc: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                with self._opener.open(Request(url), timeout=10) as resp:
                    klines = loads(resp.read())
                    if not klines:
                        return None

                    candle = klines[LAST_INDEX]
                    return candle
            except (HTTPError, URLError) as e:
                last_exc = e
                self._log.warning(
                    "Kline request failed for %s (attempt %s/%s): %s",
                    symbol,
                    attempt,
                    attempts,
                    e,
                )
            except Exception as e:
                last_exc = e
                self._log.warning(
                    "Kline parsing failed for %s (attempt %s/%s): %s",
                    symbol,
                    attempt,
                    attempts,
                    e,
                )

            if attempt < attempts:
                self._log.debug("Retrying %s in %.1fs", symbol, delay)
                time.sleep(delay)
                delay = min(delay * 2, 8.0)

        if last_exc:
            self._log.error(
                "Giving up fetching klines for %s after %s attempts: %s",
                symbol,
                attempts,
                last_exc,
            )
        return None

    def _fetch_latest_closed_candle(self, symbol: str, interval: str) -> list | None:
        """Fetch the latest closed kline for a symbol."""
        LAST_INDEX = -2
        params = urlencode({"symbol": symbol, "interval": interval, "limit": 2})
        url = f"https://api.binance.com/api/v3/klines?{params}"
        delay = 1.0
        attempts = 3
        last_exc: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                with self._opener.open(Request(url), timeout=10) as resp:
                    klines = loads(resp.read())
                    if not klines:
                        return None

                    candle = klines[LAST_INDEX]
                    return candle
            except (HTTPError, URLError) as e:
                last_exc = e
                self._log.warning(
                    "Kline request failed for %s (attempt %s/%s): %s",
                    symbol,
                    attempt,
                    attempts,
                    e,
                )
            except Exception as e:
                last_exc = e
                self._log.warning(
                    "Kline parsing failed for %s (attempt %s/%s): %s",
                    symbol,
                    attempt,
                    attempts,
                    e,
                )

            if attempt < attempts:
                self._log.debug("Retrying %s in %.1fs", symbol, delay)
                time.sleep(delay)
                delay = min(delay * 2, 8.0)

        if last_exc:
            self._log.error(
                "Giving up fetching klines for %s after %s attempts: %s",
                symbol,
                attempts,
                last_exc,
            )
        return None

    def _fetch_1m_candles_in_window(
        self, symbol: str, start_time: datetime, end_time: datetime, window_minutes: int
    ) -> list | None:
        """Fetch 1m klines for a symbol within a time window."""
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        limit = min(max(window_minutes + 1, 2), 1000)
        params = urlencode(
            {
                "symbol": symbol,
                "interval": "1m",
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": limit,
            }
        )
        url = f"https://api.binance.com/api/v3/klines?{params}"

        delay = 1.0
        attempts = 3
        last_exc: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                with self._opener.open(Request(url), timeout=10) as resp:
                    klines = loads(resp.read())
                    if not klines:
                        return None

                    filtered = [
                        kline for kline in klines if start_ms <= int(kline[0]) <= end_ms
                    ]
                    self._log.debug(
                        "%s 1m klines window (%s minutes): start=%s end=%s count=%s, last fetched=%s",
                        symbol,
                        window_minutes,
                        start_time.isoformat(),
                        end_time.isoformat(),
                        len(filtered),
                        klines[-1],
                    )
                    return filtered
            except (HTTPError, URLError) as e:
                last_exc = e
                self._log.warning(
                    "1m kline request failed for %s (attempt %s/%s): %s",
                    symbol,
                    attempt,
                    attempts,
                    e,
                )
            except Exception as e:
                last_exc = e
                self._log.warning(
                    "1m kline parsing failed for %s (attempt %s/%s): %s",
                    symbol,
                    attempt,
                    attempts,
                    e,
                )

            if attempt < attempts:
                self._log.debug("Retrying 1m klines for %s in %.1fs", symbol, delay)
                time.sleep(delay)
                delay = min(delay * 2, 8.0)

        if last_exc:
            self._log.error(
                "Giving up fetching 1m klines for %s after %s attempts: %s",
                symbol,
                attempts,
                last_exc,
            )
        return None

    def _detect_proxies(self) -> dict:
        """Detect proxy settings from environment to reuse for Binance calls."""
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
        return proxies

    def find_top_movers(
        self,
        symbols: List[SymbolInfo],
        top_n: int,
        min_change_pct: float,
    ) -> Tuple[List[SymbolInfo], List[SymbolInfo]]:
        """Find symbols with largest price changes (both directions).

        Args:
            symbols: List of symbols to analyze
            top_n: Number of top movers to return in each direction
            min_change_pct: Minimum absolute price change percentage

        Returns:
            Tuple of (top_gainers, top_losers) where each is a list of SymbolInfo
        """
        # Filter by minimum change threshold
        significant_movers = [
            s for s in symbols if abs(s.price_change_pct) >= min_change_pct
        ]

        # Separate gainers and losers
        gainers = [s for s in significant_movers if s.price_change_pct > 0]
        losers = [s for s in significant_movers if s.price_change_pct < 0]

        # Sort and take top N
        gainers.sort(key=lambda x: x.price_change_pct, reverse=True)
        losers.sort(key=lambda x: x.price_change_pct)  # ascending (most negative first)

        top_gainers = gainers[:top_n]
        top_losers = losers[:top_n]

        self._log.info(
            f"Found {len(top_gainers)} top gainers and {len(top_losers)} top losers (min change: {min_change_pct}%)"
        )
        self._log.info(
            "Top Gainers: "
            + ", ".join(f"{s.symbol} ({s.price_change_pct:.2f}%)" for s in top_gainers)
        )
        self._log.info(
            "Top Losers: "
            + ", ".join(f"{s.symbol} ({s.price_change_pct:.2f}%)" for s in top_losers)
        )

        return top_gainers, top_losers

    def _interval_to_minutes(self, interval: str) -> int:
        """Convert interval string to minutes."""
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
        return mapping.get(interval, 60)
