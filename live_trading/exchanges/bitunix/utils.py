"""Shared helpers for Bitunix exchange integration."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_query_concat(params: Optional[Dict[str, Any]]) -> str:
    """Sort query params by key and concatenate key+value pairs per Bitunix spec."""
    if not params:
        return ""
    items = sorted(params.items(), key=lambda kv: str(kv[0]))
    parts: List[str] = []
    for key, val in items:
        parts.append(str(key))
        parts.append(str(val))
    return "".join(parts)


def infer_margin_coin_from_symbol(symbol: str) -> str:
    symbol_upper = (symbol or "").upper()
    for coin in ("USDT", "USD", "USDC", "BTC", "ETH"):
        if symbol_upper.endswith(coin):
            return coin
    return "USDT"


def interval_to_milliseconds(interval: str) -> Optional[int]:
    """Convert exchange interval strings (e.g. 1m, 1h, 1d) to milliseconds."""
    if not interval:
        return None
    value = interval.strip()
    if len(value) < 2:
        return None
    unit = value[-1]
    try:
        amount = int(value[:-1])
    except ValueError:
        return None
    if amount <= 0:
        return None

    if unit == "m":
        return amount * 60 * 1000
    if unit == "h":
        return amount * 60 * 60 * 1000
    if unit == "d":
        return amount * 24 * 60 * 60 * 1000
    if unit == "w":
        return amount * 7 * 24 * 60 * 60 * 1000
    if unit == "M":
        # Calendar months vary; 30d approximation is sufficient for close_time synthesis.
        return amount * 30 * 24 * 60 * 60 * 1000
    return None


def interval_to_kline_channel_suffix(interval: str) -> Optional[str]:
    """Convert internal interval (1m, 1h, ...) to Bitunix websocket kline suffix."""
    mapping = {
        "1m": "1min",
        "3m": "3min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "60min",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "8h": "8h",
        "12h": "12h",
        "1d": "1day",
        "3d": "3day",
        "1w": "1week",
        "1M": "1month",
    }
    return mapping.get(str(interval).strip())


def floor_timestamp_to_interval_start_ms(
    timestamp_ms: int, interval: str
) -> Optional[int]:
    """Floor a millisecond timestamp to the interval open time."""
    interval_ms = interval_to_milliseconds(interval)
    if interval_ms is None or interval_ms <= 0:
        return None
    ts = int(timestamp_ms)
    if ts <= 0:
        return None
    return (ts // interval_ms) * interval_ms
