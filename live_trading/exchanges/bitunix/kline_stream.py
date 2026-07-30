"""Public Bitunix WebSocket stream for in-progress futures candles."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlparse

from ...exchange import KlineUpdate


_PUBLIC_WEBSOCKET_URL = "wss://fapi.bitunix.com/public/"
_APPLICATION_HEARTBEAT_INTERVAL_SECONDS = 20.0
_RECONNECT_BACKOFF_RESET_SECONDS = 15.0
_CHANNEL_INTERVAL_BY_STRATEGY_INTERVAL = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "60min",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "12h": "12h",
    "1d": "1day",
}


def kline_channel_for_interval(interval: str) -> str:
    """Return Bitunix's public market-kline channel for a strategy interval."""
    normalized = str(interval).strip().lower()
    try:
        channel_interval = _CHANNEL_INTERVAL_BY_STRATEGY_INTERVAL[normalized]
    except KeyError as exc:
        allowed = ", ".join(sorted(_CHANNEL_INTERVAL_BY_STRATEGY_INTERVAL))
        raise ValueError(
            f"Bitunix has no public kline stream mapping for {interval!r}; "
            f"supported intervals: {allowed}"
        ) from exc
    return f"market_kline_{channel_interval}"


class BitunixKlineStream:
    """Reconnectable public stream that emits only validated current candles."""

    def __init__(
        self,
        *,
        symbols: tuple[str, ...] | list[str],
        interval: str,
        on_kline: Callable[[KlineUpdate], None],
        logger: logging.Logger,
        proxies: Mapping[str, str] | None = None,
        heartbeat_interval_seconds: float = _APPLICATION_HEARTBEAT_INTERVAL_SECONDS,
        reconnect_backoff_reset_seconds: float = _RECONNECT_BACKOFF_RESET_SECONDS,
    ) -> None:
        normalized_symbols = tuple(
            symbol.strip().upper() for symbol in symbols if symbol.strip()
        )
        if not normalized_symbols:
            raise ValueError("symbols must contain at least one Bitunix symbol")
        if heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")
        if reconnect_backoff_reset_seconds <= 0:
            raise ValueError("reconnect_backoff_reset_seconds must be positive")
        self._symbols = normalized_symbols
        self._interval = str(interval).strip().lower()
        self._channel = kline_channel_for_interval(self._interval)
        self._on_kline = on_kline
        self._log = logger
        self._proxies = dict(proxies or {})
        self._stopping = threading.Event()
        self._socket_lock = threading.Lock()
        self._socket: Any | None = None
        self._thread: threading.Thread | None = None
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._reconnect_backoff_reset_seconds = reconnect_backoff_reset_seconds
        self._heartbeat_lock = threading.Lock()
        self._heartbeat_socket: Any | None = None
        self._heartbeat_stopping: threading.Event | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._subscription_started_at: float | None = None

    def start(self) -> None:
        """Start the stream thread, failing clearly if its declared dependency is absent."""
        try:
            import websocket  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Bitunix live kline streaming requires the websocket-client package"
            ) from exc

        if self._thread is not None and self._thread.is_alive():
            return
        self._stopping.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="bitunix-public-kline",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Close the active socket and wait briefly for its stream thread."""
        self._stopping.set()
        self._stop_application_heartbeat()
        with self._socket_lock:
            socket = self._socket
        if socket is not None:
            try:
                socket.close()
            except Exception:
                self._log.debug("Bitunix kline WebSocket close failed", exc_info=True)
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)

    def _run(self) -> None:
        import websocket

        reconnect_delay_seconds = 1.0
        while not self._stopping.is_set():
            app = websocket.WebSocketApp(
                _PUBLIC_WEBSOCKET_URL,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            with self._socket_lock:
                self._socket = app
                self._subscription_started_at = None
            try:
                app.run_forever(
                    ping_interval=20,
                    ping_timeout=10,
                    **self._proxy_options(),
                )
            except Exception:
                self._log.warning(
                    "Bitunix kline WebSocket runner failed; reconnecting",
                    exc_info=True,
                )
            finally:
                self._stop_application_heartbeat(socket=app)
                with self._socket_lock:
                    subscription_started_at = (
                        self._subscription_started_at
                        if self._socket is app
                        else None
                    )
                    if self._socket is app:
                        self._socket = None
                        self._subscription_started_at = None

            if self._stopping.is_set():
                break
            subscription_age_seconds = (
                time.monotonic() - subscription_started_at
                if subscription_started_at is not None
                else None
            )
            if (
                subscription_age_seconds is not None
                and subscription_age_seconds >= self._reconnect_backoff_reset_seconds
            ):
                reconnect_delay_seconds = 1.0
            self._log.warning(
                "Bitunix kline WebSocket disconnected; retrying in %.0fs",
                reconnect_delay_seconds,
            )
            self._stopping.wait(reconnect_delay_seconds)
            reconnect_delay_seconds = self._reconnect_delay_after_disconnect(
                reconnect_delay_seconds,
                subscription_age_seconds=subscription_age_seconds,
            )

    def _reconnect_delay_after_disconnect(
        self, current_delay_seconds: float, *, subscription_age_seconds: float | None
    ) -> float:
        """Reset after a stable subscription; back off only rapid failures."""
        if (
            subscription_age_seconds is not None
            and subscription_age_seconds >= self._reconnect_backoff_reset_seconds
        ):
            return 1.0
        return min(max(current_delay_seconds, 1.0) * 2, 30.0)

    def _proxy_options(self) -> dict[str, Any]:
        proxy_url = self._proxies.get("https") or self._proxies.get("http")
        if not proxy_url:
            return {}
        parsed = urlparse(proxy_url)
        if not parsed.hostname:
            self._log.warning("Bitunix kline WebSocket ignoring invalid proxy URL")
            return {}
        try:
            port = parsed.port
        except ValueError:
            self._log.warning("Bitunix kline WebSocket ignoring invalid proxy port")
            return {}
        options: dict[str, Any] = {"http_proxy_host": parsed.hostname}
        if port is not None:
            options["http_proxy_port"] = port
        if parsed.scheme.lower().startswith("socks"):
            options["proxy_type"] = parsed.scheme.lower()
        else:
            options["proxy_type"] = "http"
        return options

    def _on_open(self, socket: Any) -> None:
        request = {
            "op": "subscribe",
            "args": [
                {"symbol": symbol, "ch": self._channel} for symbol in self._symbols
            ],
        }
        try:
            socket.send(json.dumps(request, separators=(",", ":")))
        except Exception:
            self._log.warning("Bitunix kline WebSocket subscription failed", exc_info=True)
            try:
                socket.close()
            except Exception:
                pass
            return
        with self._socket_lock:
            if self._socket is socket:
                self._subscription_started_at = time.monotonic()
        self._start_application_heartbeat(socket)
        self._log.info(
            "Bitunix kline WebSocket subscribed (channel=%s symbols=%s)",
            self._channel,
            ",".join(self._symbols),
        )

    def _start_application_heartbeat(self, socket: Any) -> None:
        """Send Bitunix's required JSON heartbeat independently of WS pings."""
        self._stop_application_heartbeat()
        stopping = threading.Event()
        thread = threading.Thread(
            target=self._application_heartbeat_loop,
            args=(socket, stopping),
            name="bitunix-public-kline-heartbeat",
            daemon=True,
        )
        with self._heartbeat_lock:
            self._heartbeat_socket = socket
            self._heartbeat_stopping = stopping
            self._heartbeat_thread = thread
        thread.start()

    def _stop_application_heartbeat(self, socket: Any | None = None) -> None:
        """Stop the heartbeat for ``socket`` without affecting a newer stream."""
        with self._heartbeat_lock:
            if socket is not None and self._heartbeat_socket is not socket:
                return
            stopping = self._heartbeat_stopping
            thread = self._heartbeat_thread
            self._heartbeat_socket = None
            self._heartbeat_stopping = None
            self._heartbeat_thread = None
        if stopping is not None:
            stopping.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def _application_heartbeat_loop(
        self, socket: Any, stopping: threading.Event
    ) -> None:
        while not stopping.is_set() and not self._stopping.is_set():
            try:
                socket.send(
                    json.dumps(
                        {"op": "ping", "ping": int(time.time())},
                        separators=(",", ":"),
                    )
                )
            except Exception:
                if not self._stopping.is_set() and not stopping.is_set():
                    self._log.warning(
                        "Bitunix kline WebSocket application heartbeat failed; "
                        "reconnecting",
                        exc_info=True,
                    )
                    try:
                        socket.close()
                    except Exception:
                        self._log.debug(
                            "Bitunix kline WebSocket close after heartbeat failure "
                            "failed",
                            exc_info=True,
                        )
                return
            stopping.wait(self._heartbeat_interval_seconds)

    def _on_message(self, _socket: Any, message: str | bytes) -> None:
        update = self._parse_update(message)
        if update is None:
            return
        try:
            self._on_kline(update)
        except Exception:
            self._log.exception(
                "Bitunix kline WebSocket callback failed for %s", update.symbol
            )

    def _on_error(self, _socket: Any, error: Exception | str) -> None:
        if not self._stopping.is_set():
            self._log.debug("Bitunix kline WebSocket error: %s", error)

    def _on_close(self, _socket: Any, status_code: int | None, reason: str | None) -> None:
        self._stop_application_heartbeat(socket=_socket)
        if not self._stopping.is_set():
            self._log.debug(
                "Bitunix kline WebSocket closed (code=%s reason=%s)",
                status_code,
                reason,
            )

    def _parse_update(self, message: str | bytes) -> KlineUpdate | None:
        if isinstance(message, bytes):
            try:
                message = message.decode("utf-8")
            except UnicodeDecodeError:
                return None
        try:
            payload = json.loads(message)
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict) or payload.get("ch") != self._channel:
            return None

        symbol = str(payload.get("symbol", "")).strip().upper()
        data = payload.get("data")
        if symbol not in self._symbols or not isinstance(data, dict):
            return None
        try:
            update = KlineUpdate(
                symbol=symbol,
                interval=self._interval,
                event_time_ms=int(payload["ts"]),
                open=float(data["o"]),
                high=float(data["h"]),
                low=float(data["l"]),
                close=float(data["c"]),
                base_volume=float(data["b"]),
                quote_volume=float(data["q"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
        if not self._is_valid_update(update):
            return None
        return update

    @staticmethod
    def _is_valid_update(update: KlineUpdate) -> bool:
        values = (
            update.open,
            update.high,
            update.low,
            update.close,
            update.base_volume,
            update.quote_volume,
        )
        if update.event_time_ms <= 0 or not all(math.isfinite(value) for value in values):
            return False
        if min(update.open, update.high, update.low, update.close) <= 0:
            return False
        if update.base_volume < 0 or update.quote_volume < 0:
            return False
        return (
            update.low <= min(update.open, update.close)
            and update.high >= max(update.open, update.close)
            and update.low <= update.high
        )
