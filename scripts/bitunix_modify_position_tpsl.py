#!/usr/bin/env python3
"""CLI helper to modify TP/SL for an existing Bitunix position."""

from __future__ import annotations

import argparse
import json

if __package__:
    from ._bitunix_cli import add_connection_args, build_client, normalize_symbol
else:  # pragma: no cover - direct script execution
    from _bitunix_cli import add_connection_args, build_client, normalize_symbol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modify TP/SL for an existing position."
    )
    parser.add_argument("--symbol", required=True, help="Symbol, e.g. BTCUSDT")
    parser.add_argument("--position-id", required=True, help="Position ID to modify")
    parser.add_argument("--tp-price", type=float, help="New take-profit price")
    parser.add_argument("--sl-price", type=float, help="New stop-loss price")
    parser.add_argument(
        "--tp-stop-type",
        choices=["LAST_PRICE", "MARK_PRICE"],
        help="TP trigger type (LAST_PRICE/MARK_PRICE)",
    )
    parser.add_argument(
        "--sl-stop-type",
        choices=["LAST_PRICE", "MARK_PRICE"],
        help="SL trigger type (LAST_PRICE/MARK_PRICE)",
    )
    add_connection_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tp_price is None and args.sl_price is None:
        raise SystemExit("Provide at least one of --tp-price or --sl-price")
    client = build_client(args)

    response = client.modify_position_tpsl_order(
        symbol=normalize_symbol(args.symbol),
        position_id=args.position_id,
        tp_price=args.tp_price,
        tp_stop_type=args.tp_stop_type,
        sl_price=args.sl_price,
        sl_stop_type=args.sl_stop_type,
    )

    print(json.dumps(response, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
