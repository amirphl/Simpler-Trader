#!/usr/bin/env python3
"""Quick CLI to list current Bitunix open positions."""

from __future__ import annotations

import argparse
import json

if __package__:
    from ._bitunix_cli import add_connection_args, build_client, normalize_symbol
else:  # pragma: no cover - direct script execution
    from _bitunix_cli import add_connection_args, build_client, normalize_symbol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List current Bitunix positions (pending/open)."
    )
    parser.add_argument("--symbol", help="Optional symbol filter, e.g. BTCUSDT")
    parser.add_argument("--position-id", help="Optional positionId filter")
    add_connection_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = build_client(args)

    positions = client.get_pending_positions(
        symbol=normalize_symbol(args.symbol) if args.symbol else None,
        position_id=args.position_id,
    )
    print(json.dumps(positions, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
