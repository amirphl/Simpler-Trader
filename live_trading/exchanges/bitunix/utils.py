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
