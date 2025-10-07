#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Dict

from candle_downloader.models import to_milliseconds
from experiments.candle_csv import write_candles_to_csv
from experiments.pivot_detection import download_candles


def parse_datetime(value: str) -> datetime:
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    moment = datetime.fromisoformat(cleaned)
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Binance candles and save them to a CSV file."
    )
    parser.add_argument("--symbol", required=True, help="Symbol, e.g. ETHUSDT")
    parser.add_argument("--timeframe", required=True, help="Interval, e.g. 1h")
    parser.add_argument(
        "--start",
        required=True,
        help="Start datetime in ISO format, e.g. 2026-01-01T00:00:00Z",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End datetime in ISO format, e.g. 2026-01-10T00:00:00Z",
    )
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--http-proxy", default=None, help="Optional HTTP proxy URL")
    parser.add_argument("--https-proxy", default=None, help="Optional HTTPS proxy URL")
    return parser.parse_args()


def build_proxies(args: argparse.Namespace) -> Dict[str, str] | None:
    proxies: Dict[str, str] = {}
    if args.http_proxy:
        proxies["http"] = args.http_proxy
    if args.https_proxy:
        proxies["https"] = args.https_proxy
    return proxies or None


def main() -> int:
    args = parse_args()
    start = parse_datetime(args.start)
    end = parse_datetime(args.end)
    if start >= end:
        raise ValueError("start must be before end")

    candles = download_candles(
        symbol=args.symbol.strip().upper(),
        interval=args.timeframe.strip(),
        start_ms=to_milliseconds(start),
        end_ms=to_milliseconds(end),
        proxies=build_proxies(args),
    )
    write_candles_to_csv(args.output, candles)
    print(f"Saved {len(candles)} candles to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
