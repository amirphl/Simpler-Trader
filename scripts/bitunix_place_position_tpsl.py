#!/usr/bin/env python3
"""CLI helper to place TP/SL for an existing Bitunix position (positionId-based)."""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Optional

from live_trading.exchange import ExchangeConfig
from live_trading.exchanges.bitunix.client import BitunixClient


def _read_credentials(
    explicit_key: Optional[str], explicit_secret: Optional[str]
) -> tuple[str, str]:
    api_key = explicit_key or os.getenv("BITUNIX_API_KEY") or os.getenv("BITUNIX_KEY")
    api_secret = (
        explicit_secret
        or os.getenv("BITUNIX_API_SECRET")
        or os.getenv("BITUNIX_SECRET")
    )
    if not api_key or not api_secret:
        raise SystemExit(
            "API credentials missing. Provide --key/--secret or set BITUNIX_API_KEY / BITUNIX_API_SECRET."
        )
    return api_key, api_secret


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Place TP/SL for an open position using positionId."
    )
    parser.add_argument("--symbol", required=True, help="Symbol, e.g. BTCUSDT")
    parser.add_argument("--position-id", required=True, help="Position ID to protect")
    parser.add_argument("--tp-price", type=float, help="Take-profit price")
    parser.add_argument("--sl-price", type=float, help="Stop-loss price")
    parser.add_argument(
        "--tp-stop-type", choices=["LAST_PRICE", "MARK_PRICE"], help="TP trigger type"
    )
    parser.add_argument(
        "--sl-stop-type", choices=["LAST_PRICE", "MARK_PRICE"], help="SL trigger type"
    )
    parser.add_argument("--key", help="Bitunix API key (falls back to env)")
    parser.add_argument("--secret", help="Bitunix API secret (falls back to env)")
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="HTTP timeout seconds (default: 10)"
    )
    parser.add_argument(
        "--max-retries", type=int, default=1, help="Request retries (default: 1)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tp_price is None and args.sl_price is None:
        raise SystemExit("Provide at least one of --tp-price or --sl-price")

    api_key, api_secret = _read_credentials(args.key, args.secret)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    log = logging.getLogger("bitunix-cli")

    config = ExchangeConfig(
        api_key=api_key,
        api_secret=api_secret,
        testnet=False,
        timeout=args.timeout,
        max_retries=max(args.max_retries, 1),
        base_url="https://fapi.bitunix.com",
    )

    client = BitunixClient(config, log)

    response = client.place_position_tpsl_order(
        symbol=args.symbol,
        position_id=args.position_id,
        tp_price=args.tp_price,
        tp_stop_type=args.tp_stop_type,
        sl_price=args.sl_price,
        sl_stop_type=args.sl_stop_type,
    )

    print(json.dumps(response, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
