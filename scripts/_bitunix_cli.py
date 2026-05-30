"""Shared Bitunix CLI helpers."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

if __package__:
    from ._bootstrap import ensure_project_root_on_path
else:  # pragma: no cover - direct script execution
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from live_trading.exchange import ExchangeConfig


DEFAULT_BITUNIX_BASE_URL = "https://fapi.bitunix.com"


def read_credentials(
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


def add_connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--key", help="Bitunix API key (falls back to env)")
    parser.add_argument("--secret", help="Bitunix API secret (falls back to env)")
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="HTTP timeout seconds (default: 10)"
    )
    parser.add_argument(
        "--max-retries", type=int, default=1, help="Request retries (default: 1)"
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("BITUNIX_BASE_URL", DEFAULT_BITUNIX_BASE_URL),
        help=f"Bitunix futures API base URL (default: {DEFAULT_BITUNIX_BASE_URL})",
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Mark the exchange config as testnet. Use --base-url for a testnet endpoint.",
    )


def build_client(args: argparse.Namespace, logger_name: str = "bitunix-cli"):
    from live_trading.exchanges.bitunix.client import BitunixClient

    api_key, api_secret = read_credentials(args.key, args.secret)
    if args.timeout <= 0:
        raise SystemExit("--timeout must be > 0")
    if args.max_retries < 1:
        raise SystemExit("--max-retries must be >= 1")
    base_url = str(args.base_url).strip()
    if not base_url:
        raise SystemExit("--base-url cannot be empty")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    log = logging.getLogger(logger_name)
    config = ExchangeConfig(
        api_key=api_key,
        api_secret=api_secret,
        testnet=bool(args.testnet),
        timeout=args.timeout,
        max_retries=args.max_retries,
        base_url=base_url,
    )
    return BitunixClient(config, log)


def normalize_symbol(symbol: str) -> str:
    normalized = str(symbol).strip().upper()
    if not normalized:
        raise SystemExit("--symbol cannot be empty")
    return normalized
