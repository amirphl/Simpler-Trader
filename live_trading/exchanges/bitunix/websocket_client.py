"""Dedicated Bitunix websocket market-data client.

This module is intentionally separate from REST transport and owns:
- websocket connect/reconnect lifecycle
- channel subscriptions
- latest market price cache
- latest kline cache
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set

from ...exchange import ExchangeConfig

try:
    import websocket
except Exception:  # pragma: no cover - dependency/runtime environment specific
    websocket = None  # type: ignore[assignment]

from .utils import floor_timestamp_to_interval_start_ms, interval_to_kline_channel_suffix


@dataclass(frozen=True)
class _KlineKey:
    symbol: str
    interval: str
    price_type: str  # market|mark


class BitunixWebsocketClient:
    """Bitunix public websocket client for price and kline channels."""

    MAINNET_PUBLIC_URL = "wss://fapi.bitunix.com/public/"
    TESTNET_PUBLIC_URL = "wss://fapi.bitunix.com/public/"
    MAX_SUBSCRIPTIONS = 300
    _BACKOFF_MIN_SECONDS = 1.0
    _BACKOFF_MAX_SECONDS = 30.0
    _BACKOFF_JITTER_PCT = 0.25
    _MAX_PENDING_MESSAGES = 1000
    _SEND_RETRIES = 2

    def __init__(
        self, config: ExchangeConfig, logger: Optional[logging.Logger] = None
    ) -> None:
        if websocket is None:
            raise RuntimeError(
                "websocket-client is required for Bitunix websocket support"
            )
        self._log = logger or logging.getLogger(__name__)
        self._url = (
            self.TESTNET_PUBLIC_URL if config.testnet else self.MAINNET_PUBLIC_URL
        )
        self._thread: Optional[threading.Thread] = None
        self._ws_app: Optional[Any] = None
        self._stop_event = threading.Event()
        self._state_lock = threading.RLock()
        self._connected_event = threading.Event()
        self._opened_once = False
        self._connect_attempt = 0
        self._last_error: Optional[str] = None

        self._price_symbols: Set[str] = set()
        self._kline_subscriptions: Set[_KlineKey] = set()
        self._ticker_symbols: Set[str] = set()
        self._tickers_symbols: Set[str] = set()
        self._trade_symbols: Set[str] = set()

        self._latest_prices: Dict[str, Dict[str, Any]] = {}
        self._latest_klines: Dict[_KlineKey, Dict[str, Any]] = {}
        self._latest_ticker_by_symbol: Dict[str, Dict[str, Any]] = {}
        self._latest_tickers_by_symbol: Dict[str, Dict[str, Any]] = {}
        self._recent_trades_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
        self._max_trades_per_symbol = 200
        self._pending_messages: List[Dict[str, Any]] = []

    def start(self) -> None:
        with self._state_lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_forever_loop,
                name="BitunixWebsocketClient",
                daemon=True,
            )
            self._thread.start()
            self._log.info("Bitunix WS: background worker started")

    def stop(self) -> None:
        self._stop_event.set()
        app: Optional[Any]
        with self._state_lock:
            app = self._ws_app
        if app is not None:
            try:
                app.close()
            except Exception:
                pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        with self._state_lock:
            self._ws_app = None
            self._thread = None
            self._connected_event.clear()
            self._pending_messages.clear()
        self._log.info("Bitunix WS: stopped")

    def subscribe_price(self, symbols: Iterable[str]) -> None:
        normalized = {
            str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()
        }
        if not normalized:
            return
        self.start()
        with self._state_lock:
            to_add = normalized - self._price_symbols
            if not to_add:
                return
            to_add = self._cap_new_subscriptions(to_add)
            if not to_add:
                return
            self._price_symbols.update(to_add)
            args = [{"symbol": symbol, "ch": "price"} for symbol in sorted(to_add)]
        self._send_subscribe(args)

    def unsubscribe_price(self, symbols: Iterable[str]) -> None:
        normalized = {
            str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()
        }
        if not normalized:
            return
        with self._state_lock:
            to_remove = normalized & self._price_symbols
            if not to_remove:
                return
            self._price_symbols.difference_update(to_remove)
            for symbol in to_remove:
                self._latest_prices.pop(symbol, None)
        self._send_unsubscribe(
            [{"symbol": symbol, "ch": "price"} for symbol in sorted(to_remove)]
        )

    def subscribe_kline(
        self, symbol: str, interval: str, price_type: str = "market"
    ) -> None:
        normalized_symbol = str(symbol).strip().upper()
        normalized_interval = str(interval).strip()
        normalized_type = str(price_type).strip().lower()
        if not normalized_symbol or not normalized_interval:
            return
        if normalized_type not in {"market", "mark"}:
            raise ValueError(f"Unsupported kline price_type: {price_type}")
        suffix = interval_to_kline_channel_suffix(normalized_interval)
        if suffix is None:
            raise ValueError(f"Unsupported Bitunix websocket interval: {interval}")
        channel = f"{normalized_type}_kline_{suffix}"
        key = _KlineKey(
            symbol=normalized_symbol,
            interval=normalized_interval,
            price_type=normalized_type,
        )
        self.start()
        stale_keys: List[_KlineKey] = []
        with self._state_lock:
            if key in self._kline_subscriptions:
                return
            # Spec requires unsubscribe-before-subscribe when switching kline interval.
            stale_keys = [
                item
                for item in self._kline_subscriptions
                if item.symbol == normalized_symbol
                and item.price_type == normalized_type
                and item.interval != normalized_interval
            ]
            if (
                not stale_keys
                and self._current_subscription_count() >= self.MAX_SUBSCRIPTIONS
            ):
                self._log.warning(
                    "Bitunix WS: subscription limit reached (%s). "
                    "Skip subscribe %s %s %s",
                    self.MAX_SUBSCRIPTIONS,
                    normalized_symbol,
                    normalized_type,
                    normalized_interval,
                )
                return
            for stale in stale_keys:
                self._kline_subscriptions.discard(stale)
                self._latest_klines.pop(stale, None)
            self._kline_subscriptions.add(key)
        for stale in stale_keys:
            stale_suffix = interval_to_kline_channel_suffix(stale.interval)
            if stale_suffix is None:
                continue
            self._send_unsubscribe(
                [
                    {
                        "symbol": stale.symbol,
                        "ch": f"{stale.price_type}_kline_{stale_suffix}",
                    }
                ]
            )
        self._send_subscribe([{"symbol": normalized_symbol, "ch": channel}])

    def unsubscribe_kline(
        self, symbol: str, interval: str, price_type: str = "market"
    ) -> None:
        normalized_symbol = str(symbol).strip().upper()
        normalized_interval = str(interval).strip()
        normalized_type = str(price_type).strip().lower()
        suffix = interval_to_kline_channel_suffix(normalized_interval)
        if suffix is None:
            return
        channel = f"{normalized_type}_kline_{suffix}"
        key = _KlineKey(
            symbol=normalized_symbol,
            interval=normalized_interval,
            price_type=normalized_type,
        )
        with self._state_lock:
            if key not in self._kline_subscriptions:
                return
            self._kline_subscriptions.discard(key)
            self._latest_klines.pop(key, None)
        self._send_unsubscribe([{"symbol": normalized_symbol, "ch": channel}])

    def subscribe_ticker(self, symbols: Iterable[str]) -> None:
        normalized = {
            str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()
        }
        if not normalized:
            return
        self.start()
        with self._state_lock:
            to_add = normalized - self._ticker_symbols
            if not to_add:
                return
            to_add = self._cap_new_subscriptions(to_add)
            if not to_add:
                return
            self._ticker_symbols.update(to_add)
        self._send_subscribe(
            [{"symbol": symbol, "ch": "ticker"} for symbol in sorted(to_add)]
        )

    def unsubscribe_ticker(self, symbols: Iterable[str]) -> None:
        normalized = {
            str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()
        }
        if not normalized:
            return
        with self._state_lock:
            to_remove = normalized & self._ticker_symbols
            if not to_remove:
                return
            self._ticker_symbols.difference_update(to_remove)
            for symbol in to_remove:
                self._latest_ticker_by_symbol.pop(symbol, None)
        self._send_unsubscribe(
            [{"symbol": symbol, "ch": "ticker"} for symbol in sorted(to_remove)]
        )

    def subscribe_tickers(self, symbols: Iterable[str]) -> None:
        normalized = {
            str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()
        }
        if not normalized:
            return
        self.start()
        with self._state_lock:
            to_add = normalized - self._tickers_symbols
            if not to_add:
                return
            to_add = self._cap_new_subscriptions(to_add)
            if not to_add:
                return
            self._tickers_symbols.update(to_add)
        self._send_subscribe(
            [{"symbol": symbol, "ch": "tickers"} for symbol in sorted(to_add)]
        )

    def unsubscribe_tickers(self, symbols: Iterable[str]) -> None:
        normalized = {
            str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()
        }
        if not normalized:
            return
        with self._state_lock:
            to_remove = normalized & self._tickers_symbols
            if not to_remove:
                return
            self._tickers_symbols.difference_update(to_remove)
            for symbol in to_remove:
                self._latest_tickers_by_symbol.pop(symbol, None)
        self._send_unsubscribe(
            [{"symbol": symbol, "ch": "tickers"} for symbol in sorted(to_remove)]
        )

    def subscribe_trade(self, symbols: Iterable[str]) -> None:
        normalized = {
            str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()
        }
        if not normalized:
            return
        self.start()
        with self._state_lock:
            to_add = normalized - self._trade_symbols
            if not to_add:
                return
            to_add = self._cap_new_subscriptions(to_add)
            if not to_add:
                return
            self._trade_symbols.update(to_add)
        self._send_subscribe(
            [{"symbol": symbol, "ch": "trade"} for symbol in sorted(to_add)]
        )

    def unsubscribe_trade(self, symbols: Iterable[str]) -> None:
        normalized = {
            str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()
        }
        if not normalized:
            return
        with self._state_lock:
            to_remove = normalized & self._trade_symbols
            if not to_remove:
                return
            self._trade_symbols.difference_update(to_remove)
            for symbol in to_remove:
                self._recent_trades_by_symbol.pop(symbol, None)
        self._send_unsubscribe(
            [{"symbol": symbol, "ch": "trade"} for symbol in sorted(to_remove)]
        )

    def get_latest_price(self, symbol: str) -> Optional[float]:
        normalized = str(symbol).strip().upper()
        if not normalized:
            return None
        with self._state_lock:
            payload = self._latest_prices.get(normalized)
        if not payload:
            with self._state_lock:
                ticker_item = self._latest_ticker_by_symbol.get(normalized)
                tickers_item = self._latest_tickers_by_symbol.get(normalized)
            for item in (ticker_item, tickers_item):
                if isinstance(item, dict):
                    raw_last = item.get("la")
                    try:
                        return float(raw_last) if raw_last is not None else None
                    except (TypeError, ValueError):
                        continue
            return None
        data = payload.get("data")
        if not isinstance(data, dict):
            return None
        raw = data.get("mp")
        try:
            return float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    def get_latest_kline(
        self, symbol: str, interval: str, price_type: str = "market"
    ) -> Optional[Dict[str, Any]]:
        key = _KlineKey(
            symbol=str(symbol).strip().upper(),
            interval=str(interval).strip(),
            price_type=str(price_type).strip().lower(),
        )
        with self._state_lock:
            payload = self._latest_klines.get(key)
        if not payload:
            return None

        data = payload.get("data")
        ts_val = payload.get("ts")
        if not isinstance(data, dict):
            return None
        try:
            ts = int(ts_val)
        except (TypeError, ValueError):
            return None
        open_time = floor_timestamp_to_interval_start_ms(ts, key.interval)
        if open_time is None:
            return None
        return {
            "symbol": key.symbol,
            "interval": key.interval,
            "priceType": key.price_type.upper(),
            "time": open_time,
            "open": str(data.get("o", "")),
            "high": str(data.get("h", "")),
            "low": str(data.get("l", "")),
            "close": str(data.get("c", "")),
            "baseVol": str(data.get("b", "0")),
            "quoteVol": str(data.get("q", "0")),
            "ts": ts,
        }

    def get_latest_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        normalized = str(symbol).strip().upper()
        if not normalized:
            return None
        with self._state_lock:
            payload = self._latest_ticker_by_symbol.get(normalized)
            if payload is None:
                payload = self._latest_tickers_by_symbol.get(normalized)
        if payload is None:
            return None
        return dict(payload)

    def get_latest_tickers(
        self, symbols: Optional[Iterable[str]] = None
    ) -> List[Dict[str, Any]]:
        with self._state_lock:
            items = {
                key: dict(value)
                for key, value in self._latest_tickers_by_symbol.items()
                if isinstance(value, dict)
            }
        if symbols is None:
            return list(items.values())
        wanted = {str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}
        return [item for sym, item in items.items() if sym in wanted]

    def get_recent_trades(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        normalized = str(symbol).strip().upper()
        if not normalized:
            return []
        with self._state_lock:
            rows = list(self._recent_trades_by_symbol.get(normalized, []))
        max_items = max(1, int(limit))
        return rows[-max_items:]

    def _run_forever_loop(self) -> None:
        assert websocket is not None  # guarded in __init__
        backoff = self._BACKOFF_MIN_SECONDS
        while not self._stop_event.is_set():
            self._connect_attempt += 1
            self._opened_once = False
            try:
                app = websocket.WebSocketApp(
                    self._url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                with self._state_lock:
                    self._ws_app = app
                app.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as exc:
                self._last_error = str(exc)
                self._log.warning(
                    "Bitunix WS: run_forever crash on attempt=%s error=%s",
                    self._connect_attempt,
                    exc,
                )
            finally:
                self._connected_event.clear()
                with self._state_lock:
                    self._ws_app = None

            if self._stop_event.is_set():
                break
            if self._opened_once:
                backoff = self._BACKOFF_MIN_SECONDS
            else:
                backoff = min(backoff * 2.0, self._BACKOFF_MAX_SECONDS)
            jitter = backoff * self._BACKOFF_JITTER_PCT * random.random()
            delay = backoff + jitter
            self._log.warning(
                "Bitunix WS: disconnected (attempt=%s opened=%s), reconnecting in %.2fs",
                self._connect_attempt,
                self._opened_once,
                delay,
            )
            time.sleep(delay)

    def _on_open(self, ws: Any) -> None:
        self._connected_event.set()
        self._opened_once = True
        self._last_error = None
        self._log.info(
            "Bitunix WS: connected to %s (attempt=%s)",
            self._url,
            self._connect_attempt,
        )
        self._send_ping()
        self._resubscribe_all()
        self._flush_pending_messages()

    def _on_message(self, ws: Any, message: str) -> None:
        try:
            payload = json.loads(message)
            if not isinstance(payload, dict):
                self._log.debug("Bitunix WS: ignored non-dict frame")
                return

            if payload.get("op") == "ping" and payload.get("ping") is not None:
                self._send_ping()
                return
            if payload.get("op") and payload.get("msg"):
                # Generic server response / error message payload.
                self._log.debug("Bitunix WS: op message=%s", payload)

            channel = str(payload.get("ch") or "")
            symbol = str(payload.get("symbol") or "").upper()
            if channel == "price" and symbol:
                with self._state_lock:
                    self._latest_prices[symbol] = payload
                return

            if channel == "ticker" and symbol:
                data = payload.get("data")
                if isinstance(data, dict):
                    normalized = self._normalize_ticker_dict(
                        symbol=symbol,
                        data=data,
                        ts=payload.get("ts"),
                        channel=channel,
                    )
                    if normalized:
                        with self._state_lock:
                            self._latest_ticker_by_symbol[symbol] = normalized
                            self._latest_tickers_by_symbol[symbol] = normalized
                return

            if channel == "tickers":
                data_list = payload.get("data")
                if isinstance(data_list, list):
                    ts_val = payload.get("ts")
                    updated: Dict[str, Dict[str, Any]] = {}
                    for row in data_list:
                        if not isinstance(row, dict):
                            continue
                        row_symbol = str(row.get("s") or "").upper()
                        if not row_symbol:
                            continue
                        normalized = self._normalize_ticker_dict(
                            symbol=row_symbol,
                            data=row,
                            ts=ts_val,
                            channel=channel,
                        )
                        if normalized:
                            updated[row_symbol] = normalized
                    if updated:
                        with self._state_lock:
                            self._latest_tickers_by_symbol.update(updated)
                return

            if channel == "trade" and symbol:
                data_list = payload.get("data")
                if not isinstance(data_list, list):
                    return
                trades: List[Dict[str, Any]] = []
                for row in data_list:
                    if not isinstance(row, dict):
                        continue
                    trades.append(
                        {
                            "symbol": symbol,
                            "price": str(row.get("p", "")),
                            "volume": str(row.get("v", "")),
                            "side": str(row.get("s", "")).lower(),
                            "time": str(row.get("t", "")),
                            "ts": payload.get("ts"),
                        }
                    )
                if not trades:
                    return
                with self._state_lock:
                    existing = self._recent_trades_by_symbol.setdefault(symbol, [])
                    existing.extend(trades)
                    if len(existing) > self._max_trades_per_symbol:
                        del existing[0 : len(existing) - self._max_trades_per_symbol]
                return

            if "_kline_" in channel and symbol:
                key = self._key_from_channel(symbol, channel)
                if key is None:
                    self._log.debug(
                        "Bitunix WS: unsupported kline channel=%s symbol=%s",
                        channel,
                        symbol,
                    )
                    return
                with self._state_lock:
                    self._latest_klines[key] = payload
                return
            if channel:
                self._log.debug("Bitunix WS: unhandled channel=%s payload=%s", channel, payload)
        except Exception as exc:
            self._log.warning("Bitunix WS: failed to handle message error=%s", exc)

    def _on_error(self, ws: Any, error: Any) -> None:
        self._last_error = str(error)
        self._log.warning("Bitunix WS: error=%s", error)

    def _on_close(
        self, ws: Any, close_status_code: Optional[int], close_msg: Optional[str]
    ) -> None:
        self._connected_event.clear()
        self._log.info(
            "Bitunix WS: closed code=%s msg=%s", close_status_code, close_msg or ""
        )

    def _send_json(self, payload: Dict[str, Any], *, enqueue_on_fail: bool = True) -> bool:
        if not self._connected_event.is_set():
            if enqueue_on_fail:
                self._enqueue_message(payload)
            return False
        encoded = json.dumps(payload, separators=(",", ":"))
        for attempt in range(1, self._SEND_RETRIES + 1):
            with self._state_lock:
                app = self._ws_app
            if app is None:
                if enqueue_on_fail:
                    self._enqueue_message(payload)
                return False
            try:
                app.send(encoded)
                return True
            except Exception as exc:
                self._last_error = str(exc)
                self._log.warning(
                    "Bitunix WS: send failed attempt=%s/%s payload=%s error=%s",
                    attempt,
                    self._SEND_RETRIES,
                    payload,
                    exc,
                )
                if attempt < self._SEND_RETRIES:
                    time.sleep(0.05 * attempt)
        if enqueue_on_fail:
            self._enqueue_message(payload)
        return False

    def _send_ping(self) -> None:
        self._send_json({"op": "ping", "ping": int(time.time())})

    def _send_subscribe(self, args: list[Dict[str, Any]]) -> None:
        if not args:
            return
        self._send_json({"op": "subscribe", "args": args})

    def _send_unsubscribe(self, args: list[Dict[str, Any]]) -> None:
        if not args:
            return
        self._send_json({"op": "unsubscribe", "args": args})

    def _enqueue_message(self, payload: Dict[str, Any]) -> None:
        with self._state_lock:
            self._pending_messages.append(payload)
            if len(self._pending_messages) > self._MAX_PENDING_MESSAGES:
                drop_count = len(self._pending_messages) - self._MAX_PENDING_MESSAGES
                del self._pending_messages[:drop_count]
                self._log.warning(
                    "Bitunix WS: pending queue overflow, dropped %s oldest messages",
                    drop_count,
                )

    def _flush_pending_messages(self) -> None:
        with self._state_lock:
            pending = list(self._pending_messages)
            self._pending_messages.clear()
        if not pending:
            return
        self._log.info("Bitunix WS: flushing %s queued messages", len(pending))
        for idx, payload in enumerate(pending):
            if self._stop_event.is_set():
                return
            if not self._send_json(payload, enqueue_on_fail=False):
                # Connection may be unstable; keep remaining messages queued order-safe.
                with self._state_lock:
                    self._pending_messages[0:0] = pending[idx:]
                return

    def _resubscribe_all(self) -> None:
        with self._state_lock:
            price_symbols = sorted(self._price_symbols)
            kline_subs = list(self._kline_subscriptions)
            ticker_symbols = sorted(self._ticker_symbols)
            tickers_symbols = sorted(self._tickers_symbols)
            trade_symbols = sorted(self._trade_symbols)

        if price_symbols:
            self._send_subscribe(
                [{"symbol": symbol, "ch": "price"} for symbol in price_symbols]
            )
        if ticker_symbols:
            self._send_subscribe(
                [{"symbol": symbol, "ch": "ticker"} for symbol in ticker_symbols]
            )
        if tickers_symbols:
            self._send_subscribe(
                [{"symbol": symbol, "ch": "tickers"} for symbol in tickers_symbols]
            )
        if trade_symbols:
            self._send_subscribe(
                [{"symbol": symbol, "ch": "trade"} for symbol in trade_symbols]
            )
        for sub in kline_subs:
            suffix = interval_to_kline_channel_suffix(sub.interval)
            if suffix is None:
                continue
            self._send_subscribe(
                [
                    {
                        "symbol": sub.symbol,
                        "ch": f"{sub.price_type}_kline_{suffix}",
                    }
                ]
            )

    def _normalize_ticker_dict(
        self, symbol: str, data: Dict[str, Any], ts: Any, channel: str
    ) -> Optional[Dict[str, Any]]:
        normalized_symbol = str(symbol).strip().upper()
        if not normalized_symbol:
            return None
        normalized: Dict[str, Any] = {
            "symbol": normalized_symbol,
            "open": str(data.get("o", "")),
            "high": str(data.get("h", "")),
            "low": str(data.get("l", "")),
            "lastPrice": str(data.get("la", "")),
            "baseVolume": str(data.get("b", "")),
            "quoteVolume": str(data.get("q", "")),
            "priceChangePercent": str(data.get("r", "")),
            "bestBidPrice": str(data.get("bd", "")),
            "bestAskPrice": str(data.get("ak", "")),
            "bestBidVolume": str(data.get("bv", "")),
            "bestAskVolume": str(data.get("av", "")),
            "sourceChannel": channel,
            "ts": ts,
            # Preserve raw-compatible aliases for convenience.
            "o": str(data.get("o", "")),
            "h": str(data.get("h", "")),
            "l": str(data.get("l", "")),
            "la": str(data.get("la", "")),
            "b": str(data.get("b", "")),
            "q": str(data.get("q", "")),
            "r": str(data.get("r", "")),
        }
        return normalized

    def _current_subscription_count(self) -> int:
        return (
            len(self._price_symbols)
            + len(self._kline_subscriptions)
            + len(self._ticker_symbols)
            + len(self._tickers_symbols)
            + len(self._trade_symbols)
        )

    def _cap_new_subscriptions(self, symbols: Set[str]) -> Set[str]:
        """Bound new subscriptions to exchange max channels per connection."""
        available = self.MAX_SUBSCRIPTIONS - self._current_subscription_count()
        if available <= 0:
            self._log.warning(
                "Bitunix WS: subscription limit reached (%s).", self.MAX_SUBSCRIPTIONS
            )
            return set()
        ordered = sorted(symbols)
        kept = set(ordered[:available])
        dropped = ordered[available:]
        if dropped:
            self._log.warning(
                "Bitunix WS: dropped %s subscriptions due to limit %s",
                len(dropped),
                self.MAX_SUBSCRIPTIONS,
            )
        return kept

    def _key_from_channel(self, symbol: str, channel: str) -> Optional[_KlineKey]:
        if "_kline_" not in channel:
            return None
        if channel.startswith("market_kline_"):
            price_type = "market"
            suffix = channel.replace("market_kline_", "", 1)
        elif channel.startswith("mark_kline_"):
            price_type = "mark"
            suffix = channel.replace("mark_kline_", "", 1)
        else:
            return None

        interval_map = {
            "1min": "1m",
            "3min": "3m",
            "5min": "5m",
            "15min": "15m",
            "30min": "30m",
            "60min": "1h",
            "2h": "2h",
            "4h": "4h",
            "6h": "6h",
            "8h": "8h",
            "12h": "12h",
            "1day": "1d",
            "3day": "3d",
            "1week": "1w",
            "1month": "1M",
        }
        interval = interval_map.get(suffix)
        if interval is None:
            return None
        return _KlineKey(symbol=symbol, interval=interval, price_type=price_type)
