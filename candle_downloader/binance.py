from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, List
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener

from .models import Candle, normalize_symbol

BINANCE_BASE_URL = "https://api.binance.com"
MAX_BATCH = 10000


def interval_to_milliseconds(interval: str) -> int:
    """Translate Binance interval strings into millisecond durations."""
    normalized = interval.strip()
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
        "8h": 28_800_000,
        "12h": 43_200_000,
        "1d": 86_400_000,
        "3d": 259_200_000,
        "1w": 604_800_000,
        "1M": 2_592_000_000,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported interval: {interval}")
    return mapping[normalized]


@dataclass(frozen=True)
class BinanceClientConfig:
    base_url: str = BINANCE_BASE_URL
    timeout: float = 10.0
    proxies: Dict[str, str] | None = None
    max_retries: int = 5
    initial_retry_delay: float = 1.0  # seconds
    max_retry_delay: float = 60.0  # seconds
    retry_backoff_multiplier: float = 2.0


class BinanceClient:
    """Minimal Binance REST client for historical candle retrieval with retry logic."""

    def __init__(self, config: BinanceClientConfig, logger: logging.Logger | None = None) -> None:
        self._base_url = config.base_url.rstrip("/")
        self._timeout = config.timeout
        self._max_retries = config.max_retries
        self._initial_retry_delay = config.initial_retry_delay
        self._max_retry_delay = config.max_retry_delay
        self._retry_backoff_multiplier = config.retry_backoff_multiplier
        self._log = logger or logging.getLogger(__name__)
        handlers = []
        if config.proxies:
            handlers.append(ProxyHandler(config.proxies))
        self._opener = build_opener(*handlers)

    def fetch_klines(
        self,
        *,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        limit: int,
    ) -> List[Candle]:
        if end_ms <= start_ms:
            return []
        if limit <= 0 or limit > MAX_BATCH:
            raise ValueError(f"limit must be in 1..{MAX_BATCH}")
        params: Dict[str, str | int] = {
            "symbol": normalize_symbol(symbol),
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms - 1,
            "limit": limit,
        }
        query = urlencode(params)
        request = Request(f"{self._base_url}/api/v3/klines?{query}")

        last_exception: Exception | None = None
        delay = self._initial_retry_delay

        for attempt in range(self._max_retries):
            try:
                with self._opener.open(request, timeout=self._timeout) as response:
                    body = response.read()
                payload = json.loads(body)
                return [Candle.from_binance(symbol, interval, kline) for kline in payload]

            except HTTPError as exc:
                # Don't retry on client errors (4xx) except 429 (rate limit) and 408 (timeout)
                if 400 <= exc.code < 500 and exc.code not in (429, 408):
                    raise RuntimeError(f"Binance request failed with status {exc.code}: {exc.reason}") from exc

                last_exception = exc
                if attempt < self._max_retries - 1:
                    self._log.warning(
                        f"HTTP error {exc.code} on attempt {attempt + 1}/{self._max_retries}, "
                        f"retrying in {delay:.1f}s...",
                        extra={"symbol": symbol, "interval": interval, "status": exc.code},
                    )
                else:
                    raise RuntimeError(
                        f"Binance request failed after {self._max_retries} attempts with status {exc.code}: {exc.reason}"
                    ) from exc

            except URLError as exc:
                last_exception = exc
                error_msg = str(exc.reason) if exc.reason else str(exc)
                if attempt < self._max_retries - 1:
                    self._log.warning(
                        f"Connection error on attempt {attempt + 1}/{self._max_retries}, "
                        f"retrying in {delay:.1f}s: {error_msg}",
                        extra={"symbol": symbol, "interval": interval},
                    )
                else:
                    raise RuntimeError(
                        f"Binance request failed after {self._max_retries} attempts: {error_msg}"
                    ) from exc

            except (TimeoutError, OSError) as exc:
                last_exception = exc
                if attempt < self._max_retries - 1:
                    self._log.warning(
                        f"Timeout/OS error on attempt {attempt + 1}/{self._max_retries}, "
                        f"retrying in {delay:.1f}s: {exc}",
                        extra={"symbol": symbol, "interval": interval},
                    )
                else:
                    raise RuntimeError(
                        f"Binance request failed after {self._max_retries} attempts: {exc}"
                    ) from exc

            except Exception as exc:
                # Unexpected errors - don't retry
                raise RuntimeError(f"Unexpected error in Binance request: {exc}") from exc

            # Exponential backoff before retry
            if attempt < self._max_retries - 1:
                time.sleep(delay)
                delay = min(delay * self._retry_backoff_multiplier, self._max_retry_delay)

        # Should never reach here, but just in case
        if last_exception:
            raise RuntimeError(f"Binance request failed after {self._max_retries} attempts") from last_exception
        raise RuntimeError("Binance request failed for unknown reason")

    def close(self) -> None:
        # urllib opener does not require explicit closing; kept for symmetry.
        return

    def fetch_top_symbols(self, limit: int = 100) -> List[str]:
        """Return the most liquid symbols based on 24h quote volume."""
        request = Request(f"{self._base_url}/api/v3/ticker/24hr")
        with self._opener.open(request, timeout=self._timeout) as response:
            payload = json.loads(response.read())
        if not isinstance(payload, list):
            return []

        def parse_volume(item: dict) -> float:
            try:
                return float(item.get("quoteVolume", "0"))
            except (TypeError, ValueError):
                return 0.0

        # Prefer USDT pairs to keep things consistent
        sorted_items = sorted(payload, key=parse_volume, reverse=True)
        symbols: List[str] = []
        for item in sorted_items:
            symbol = item.get("symbol")
            if not isinstance(symbol, str):
                continue
            if not symbol.endswith("USDT"):
                continue
            symbols.append(symbol)
            if len(symbols) >= limit:
                break
        return symbols

