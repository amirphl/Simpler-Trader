#!/usr/bin/env python3
"""
Tiny helper to play with Bitunix stop-loss updates outside the main app.

It mirrors BitunixExchange.update_stop_loss_order by calling
BitunixClient.modify_tpsl_order with MARK_PRICE/LIMIT params.
"""

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
    """Resolve API key/secret from CLI args or environment."""
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
        description="Update an existing Bitunix stop-loss order (TP/SL endpoint)."
    )
    parser.add_argument(
        "--order-id", required=True, help="Existing TP/SL order id to update"
    )
    parser.add_argument(
        "--stop-price",
        type=float,
        required=True,
        help="New stop-loss trigger/limit price",
    )
    parser.add_argument(
        "--qty",
        type=float,
        required=True,
        help="Stop-loss quantity (usually the position size)",
    )
    parser.add_argument("--key", help="Bitunix API key (falls back to env)")
    parser.add_argument("--secret", help="Bitunix API secret (falls back to env)")
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout seconds (default: 10)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Number of retries for the request (default: 1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key, api_secret = _read_credentials(args.key, args.secret)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
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

    log.info(
        "Updating stop-loss: order_id=%s stop_price=%s qty=%s testnet=%s",
        args.order_id,
        args.stop_price,
        args.qty,
        config.testnet,
    )

    response = client.modify_tpsl_order(
        order_id=args.order_id,
        sl_price=args.stop_price,
        sl_stop_type="MARK_PRICE",
        sl_order_type="LIMIT",
        sl_order_price=args.stop_price,
        sl_qty=args.qty,
    )

    print(json.dumps(response, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
