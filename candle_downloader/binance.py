from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import OpenerDirector, ProxyHandler, Request, build_opener

from .models import Candle, normalize_symbol

BINANCE_BASE_URL = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"
TICKER_24H_ENDPOINT = "/api/v3/ticker/24hr"
DEFAULT_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Simpler-Trader/1.0",
}
MAX_BATCH = 500

_INTERVAL_TO_MILLISECONDS = {
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


def interval_to_milliseconds(interval: str) -> int:
    """Translate Binance interval strings into millisecond durations."""
    normalized = interval.strip()
    try:
        return _INTERVAL_TO_MILLISECONDS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported interval: {interval}") from exc


@dataclass(frozen=True, slots=True)
class BinanceClientConfig:
    base_url: str = BINANCE_BASE_URL
    timeout: float = 10.0
    proxies: Mapping[str, str] | None = None
    max_retries: int = 5
    initial_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    retry_backoff_multiplier: float = 2.0

    def __post_init__(self) -> None:
        base_url = self.base_url.strip()
        if not base_url:
            raise ValueError("base_url cannot be empty")
        object.__setattr__(self, "base_url", base_url)
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.max_retries <= 0:
            raise ValueError("max_retries must be at least 1")
        if self.initial_retry_delay <= 0:
            raise ValueError("initial_retry_delay must be positive")
        if self.max_retry_delay <= 0:
            raise ValueError("max_retry_delay must be positive")
        if self.max_retry_delay < self.initial_retry_delay:
            raise ValueError(
                "max_retry_delay must be greater than or equal to initial_retry_delay"
            )
        if self.retry_backoff_multiplier < 1:
            raise ValueError("retry_backoff_multiplier must be at least 1")


class BinanceClient:
    """Minimal Binance REST client for historical candle retrieval with retry logic."""

    def __init__(
        self,
        config: BinanceClientConfig,
        logger: logging.Logger | None = None,
        opener: OpenerDirector | None = None,
    ) -> None:
        self._base_url = config.base_url.rstrip("/")
        self._timeout = config.timeout
        self._max_retries = config.max_retries
        self._initial_retry_delay = config.initial_retry_delay
        self._max_retry_delay = config.max_retry_delay
        self._retry_backoff_multiplier = config.retry_backoff_multiplier
        self._log = logger or logging.getLogger(__name__)
        self._opener = opener or self._build_opener(config.proxies)

    def fetch_klines(
        self,
        *,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        limit: int,
    ) -> List[Candle]:
        normalized_symbol = normalize_symbol(symbol)
        normalized_interval = self._normalize_interval(interval)
        validated_limit = self._validate_limit(limit)
        if end_ms <= start_ms:
            return []

        params = self._build_klines_params(
            symbol=normalized_symbol,
            interval=normalized_interval,
            start_ms=start_ms,
            end_ms=end_ms,
            limit=validated_limit,
        )
        payload = self._get_json(KLINES_ENDPOINT, params, context=params)
        rows = self._require_list_payload(payload, endpoint=KLINES_ENDPOINT)
        return self._parse_klines(
            rows,
            symbol=normalized_symbol,
            interval=normalized_interval,
        )

    def fetch_top_symbols(self, limit: int = 100) -> List[str]:
        """Return the most liquid USDT symbols based on 24h quote volume."""
        if limit <= 0:
            return []

        payload = self._get_json(
            TICKER_24H_ENDPOINT,
            context={"limit": limit},
        )
        rows = self._require_list_payload(payload, endpoint=TICKER_24H_ENDPOINT)
        return self._extract_top_symbols(rows, limit=limit)

    def close(self) -> None:
        close_method = getattr(self._opener, "close", None)
        if callable(close_method):
            close_method()

    def _build_opener(self, proxies: Mapping[str, str] | None) -> OpenerDirector:
        handlers = [ProxyHandler(dict(proxies))] if proxies else []
        return build_opener(*handlers)

    def _normalize_interval(self, interval: str) -> str:
        normalized_interval = interval.strip()
        interval_to_milliseconds(normalized_interval)
        return normalized_interval

    def _validate_limit(self, limit: int) -> int:
        if limit <= 0 or limit > MAX_BATCH:
            raise ValueError(f"limit must be in 1..{MAX_BATCH}")
        return limit

    def _build_klines_params(
        self,
        *,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        limit: int,
    ) -> Dict[str, str | int]:
        return {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms - 1,
            "limit": limit,
        }

    def _parse_klines(
        self,
        rows: Sequence[object],
        *,
        symbol: str,
        interval: str,
    ) -> List[Candle]:
        candles: List[Candle] = []
        for index, row in enumerate(rows):
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
                raise RuntimeError(
                    f"Binance returned invalid kline row at index {index}: {row!r}"
                )
            try:
                candles.append(Candle.from_binance(symbol, interval, row))
            except (IndexError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Failed to parse Binance kline row at index {index}: {row!r}"
                ) from exc
        return candles

    def _extract_top_symbols(self, rows: Sequence[object], *, limit: int) -> List[str]:
        def parse_volume(item: object) -> float:
            if not isinstance(item, Mapping):
                return 0.0
            try:
                return float(item.get("quoteVolume", "0"))
            except (TypeError, ValueError):
                return 0.0

        symbols: List[str] = []
        seen: set[str] = set()
        for item in sorted(rows, key=parse_volume, reverse=True):
            if not isinstance(item, Mapping):
                continue
            symbol = item.get("symbol")
            if not isinstance(symbol, str):
                continue
            normalized_symbol = normalize_symbol(symbol)
            if not normalized_symbol.endswith("USDT"):
                continue
            if normalized_symbol in seen:
                continue
            seen.add(normalized_symbol)
            symbols.append(normalized_symbol)
            if len(symbols) >= limit:
                break
        return symbols

    def _get_json(
        self,
        endpoint: str,
        params: Mapping[str, str | int] | None = None,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> object:
        url = self._build_url(endpoint, params)
        delay = self._initial_retry_delay
        last_exception: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            sleep_delay = delay
            request = self._build_request(url)
            try:
                with self._opener.open(request, timeout=self._timeout) as response:
                    return self._decode_json_response(response.read(), endpoint=endpoint)
            except HTTPError as exc:
                sleep_delay = self._resolve_retry_delay(exc, fallback=delay)
                self._raise_or_record_http_error(
                    exc=exc,
                    attempt=attempt,
                    endpoint=endpoint,
                    delay=sleep_delay,
                    context=context,
                )
                last_exception = exc
            except URLError as exc:
                last_exception = exc
                self._log_retry(
                    attempt=attempt,
                    delay=delay,
                    endpoint=endpoint,
                    message=f"Connection error: {self._format_url_error(exc)}",
                    context=context,
                )
            except (TimeoutError, OSError) as exc:
                last_exception = exc
                self._log_retry(
                    attempt=attempt,
                    delay=delay,
                    endpoint=endpoint,
                    message=f"Timeout/OS error: {exc}",
                    context=context,
                )
            except RuntimeError:
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"Unexpected Binance request failure for {endpoint}: {exc}"
                ) from exc

            if attempt < self._max_retries:
                time.sleep(sleep_delay)
                delay = min(
                    sleep_delay * self._retry_backoff_multiplier,
                    self._max_retry_delay,
                )

        if last_exception is not None:
            raise RuntimeError(
                f"Binance request failed after {self._max_retries} attempts for {endpoint}"
            ) from last_exception
        raise RuntimeError(f"Binance request failed for unknown reason: {endpoint}")

    def _build_request(self, url: str) -> Request:
        return Request(url, headers=DEFAULT_HEADERS)

    def _decode_json_response(self, raw_body: bytes, *, endpoint: str) -> object:
        try:
            return json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Binance returned invalid JSON for {endpoint}") from exc

    def _raise_or_record_http_error(
        self,
        *,
        exc: HTTPError,
        attempt: int,
        endpoint: str,
        delay: float,
        context: Mapping[str, Any] | None,
    ) -> None:
        if 400 <= exc.code < 500 and exc.code not in (408, 418, 429):
            details = self._extract_http_error_details(exc)
            raise RuntimeError(
                f"Binance request failed with status {exc.code} for {endpoint}: {details}"
            ) from exc

        details = self._extract_http_error_details(exc)
        self._log_retry(
            attempt=attempt,
            delay=delay,
            endpoint=endpoint,
            message=f"HTTP {exc.code}: {details}",
            context=context,
        )

    def _log_retry(
        self,
        *,
        attempt: int,
        delay: float,
        endpoint: str,
        message: str,
        context: Mapping[str, Any] | None,
    ) -> None:
        if attempt >= self._max_retries:
            return
        extra = {"endpoint": endpoint}
        if context:
            extra.update(dict(context))
        self._log.warning(
            "Binance request attempt %s/%s failed for %s, retrying in %.1fs: %s",
            attempt,
            self._max_retries,
            endpoint,
            delay,
            message,
            extra=extra,
        )

    def _build_url(self, endpoint: str, params: Mapping[str, str | int] | None) -> str:
        if params:
            return f"{self._base_url}{endpoint}?{urlencode(params)}"
        return f"{self._base_url}{endpoint}"

    def _resolve_retry_delay(self, exc: HTTPError, *, fallback: float) -> float:
        retry_after = exc.headers.get("Retry-After") if exc.headers else None
        if retry_after is None:
            return fallback
        try:
            return min(max(float(retry_after), 0.0), self._max_retry_delay)
        except (TypeError, ValueError):
            return fallback

    def _require_list_payload(self, payload: object, *, endpoint: str) -> List[object]:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, Mapping):
            code = payload.get("code")
            message = payload.get("msg") or payload.get("message") or repr(payload)
            raise RuntimeError(
                f"Binance returned an API error for {endpoint}: code={code} msg={message}"
            )
        raise RuntimeError(
            f"Binance returned unexpected payload type for {endpoint}: {type(payload).__name__}"
        )

    def _extract_http_error_details(self, exc: HTTPError) -> str:
        try:
            body = exc.read()
        except Exception:
            body = b""

        if body:
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                text = body.decode("utf-8", errors="replace").strip()
                return text or str(exc.reason)
            if isinstance(payload, Mapping):
                message = payload.get("msg") or payload.get("message")
                code = payload.get("code")
                if message is not None and code is not None:
                    return f"{message} (code={code})"
                if message is not None:
                    return str(message)
                return json.dumps(payload, sort_keys=True)

        return str(exc.reason)

    def _format_url_error(self, exc: URLError) -> str:
        reason = getattr(exc, "reason", None)
        return str(reason) if reason else str(exc)
