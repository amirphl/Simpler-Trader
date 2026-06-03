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

if __package__:
    from ._bitunix_cli import add_connection_args, build_client
else:  # pragma: no cover - direct script execution
    from _bitunix_cli import add_connection_args, build_client


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
    add_connection_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stop_price <= 0:
        raise SystemExit("--stop-price must be > 0")
    if args.qty <= 0:
        raise SystemExit("--qty must be > 0")
    client = build_client(args)
    log = logging.getLogger("bitunix-cli")

    log.info(
        "Updating stop-loss: order_id=%s stop_price=%s qty=%s",
        args.order_id,
        args.stop_price,
        args.qty,
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
