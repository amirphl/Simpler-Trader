from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from candle_downloader.binance import BinanceClient, BinanceClientConfig
from candle_downloader.downloader import CandleDownloader, DownloadRequest
from candle_downloader.storage import build_store


def parse_datetime(value: str) -> datetime:
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        dt = datetime.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - argparse path
        raise argparse.ArgumentTypeError(f"Invalid datetime: {value}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download Binance candles into local storage.")
    parser.add_argument("--symbol", required=True, help="Trading pair symbol, e.g. BTCUSDT.")
    parser.add_argument("--interval", required=True, help="Binance interval, e.g. 1m, 5m, 1h.")
    parser.add_argument("--start", required=True, type=parse_datetime, help="Inclusive ISO8601 datetime (UTC).")
    parser.add_argument("--end", required=True, type=parse_datetime, help="Exclusive ISO8601 datetime (UTC).")
    parser.add_argument(
        "--mode",
        choices=("normal", "override"),
        default="normal",
        help="normal downloads only missing candles; override re-fetches everything.",
    )
    parser.add_argument("--store-kind", choices=("sqlite", "csv"), default="sqlite")
    parser.add_argument(
        "--store-path",
        type=Path,
        default=Path("./data/candles.db"),
        help="Path to the sqlite database or csv file.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1000,
        help="Maximum number of candles to request per Binance API call.",
    )
    parser.add_argument("--http-proxy", dest="http_proxy", help="HTTP proxy URL.")
    parser.add_argument("--https-proxy", dest="https_proxy", help="HTTPS proxy URL.")
    parser.add_argument("--proxy", dest="proxy", help="Shortcut to set both HTTP and HTTPS proxy.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser


def resolve_proxies(args: argparse.Namespace) -> Dict[str, str]:
    proxies: Dict[str, str] = {}
    if args.proxy:
        proxies["http"] = args.proxy
        proxies["https"] = args.proxy
    if args.http_proxy:
        proxies["http"] = args.http_proxy
    if args.https_proxy:
        proxies["https"] = args.https_proxy
    if proxies:
        return proxies
    # Fall back to environment variables so the CLI mirrors requests' defaults.
    env_http = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
    env_https = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    if env_http:
        proxies["http"] = env_http
    if env_https:
        proxies["https"] = env_https
    return proxies


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    store_path: Path = args.store_path
    store = build_store(args.store_kind, store_path)
    proxies = resolve_proxies(args)
    client_logger = logging.getLogger("candle_downloader.binance")
    client = BinanceClient(BinanceClientConfig(proxies=proxies or None), logger=client_logger)
    downloader = CandleDownloader(client=client, store=store)
    request = DownloadRequest(
        symbol=args.symbol,
        interval=args.interval,
        start=args.start,
        end=args.end,
        override=args.mode == "override",
        max_batch=args.batch,
    )
    try:
        stats = downloader.sync(request)
        print(json.dumps(stats.__dict__, indent=2))
    finally:
        store.close()
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

