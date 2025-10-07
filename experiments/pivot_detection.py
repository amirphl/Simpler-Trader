from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Sequence

from candle_downloader.binance import BinanceClient, BinanceClientConfig, interval_to_milliseconds, MAX_BATCH
from candle_downloader.models import Candle, to_milliseconds
from experiments.candle_csv import read_candles_from_csv


@dataclass
class Pivot:
    index: int               # P
    type: str                # "bullish" or "bearish"
    high: float
    low: float
    reference_index: int     # R (the candle j where search started)
    trigger_index: int       # p (crosses candidate level)
    invalidation_index: int | None  # q (crosses invalidation), may be None
    haunted: bool = False
    next_bullish_index: int | None = None
    next_bearish_index: int | None = None
    previous_bullish_index: int | None = None
    previous_bearish_index: int | None = None


def _prev_match(candles: Sequence[Candle], start: int, predicate) -> int | None:
    for i in range(start - 1, -1, -1):
        if predicate(candles[i]):
            return i
    return None


def _next_match(candles: Sequence[Candle], start: int, predicate, end: int) -> int | None:
    for i in range(start + 1, end + 1):
        if predicate(candles[i]):
            return i
    return None


def _argmax_high(candles: Sequence[Candle], a: int, b: int) -> int:
    best = a
    best_val = candles[a].high
    for i in range(a + 1, b + 1):
        if candles[i].high > best_val:
            best_val = candles[i].high
            best = i
    return best


def _argmin_low(candles: Sequence[Candle], a: int, b: int) -> int:
    best = a
    best_val = candles[a].low
    for i in range(a + 1, b + 1):
        if candles[i].low < best_val:
            best_val = candles[i].low
            best = i
    return best


def detect_pivots(candles: Sequence[Candle], scan_length: int) -> List[Pivot]:
    """Detect pivots using the provided pseudocode."""
    if not candles:
        return []

    M = len(candles)
    scan_start = max(0, M - scan_length)
    scan_end = M - 1
    pivots: List[Pivot] = []

    cursor = scan_start
    while cursor <= scan_end:
        c = candles[cursor]
        if c.is_bearish():
            j = cursor
            k = _next_match(candles, j, Candle.is_bullish, scan_end)
            if k is None:
                break

            target_low = min(candles[t].low for t in range(j, k + 1))
            pb = _prev_match(candles, j, Candle.is_bearish)
            left_bound = j if pb is None else pb
            invalidation_high = max(candles[t].high for t in range(left_bound, j + 1))

            p = None
            q = None
            x = k + 1
            while x <= scan_end:
                if candles[x].high > invalidation_high:
                    q = x
                    break
                if candles[x].low < target_low:
                    p = x
                    break
                x += 1

            if p is None:
                if q is not None:
                    cursor = q
                else:
                    break
            else:
                pivot_index = _argmax_high(candles, j + 1, p)
                pivots.append(
                    Pivot(
                        index=pivot_index,
                        type="bearish",
                        high=candles[pivot_index].high,
                        low=candles[pivot_index].low,
                        reference_index=j,
                        trigger_index=p,
                        invalidation_index=q,
                        haunted=False,
                        next_bullish_index=k,
                        previous_bearish_index=pb,
                    )
                )
                cursor = p
        elif c.is_bullish():
            j = cursor
            k = _next_match(candles, j, Candle.is_bearish, scan_end)
            if k is None:
                break

            candidate_high = max(candles[t].high for t in range(j, k + 1))
            pb = _prev_match(candles, j, Candle.is_bullish)
            left_bound = j if pb is None else pb
            target_low = min(candles[t].low for t in range(left_bound, j + 1))

            p = None
            q = None
            x = k + 1
            while x <= scan_end:
                if candles[x].low < target_low:
                    q = x
                    break
                if candles[x].high > candidate_high:
                    p = x
                    break
                x += 1

            if p is None:
                if q is not None:
                    cursor = q
                else:
                    break
            else:
                pivot_index = _argmin_low(candles, j + 1, p)
                pivots.append(
                    Pivot(
                        index=pivot_index,
                        type="bullish",
                        high=candles[pivot_index].high,
                        low=candles[pivot_index].low,
                        reference_index=j,
                        trigger_index=p,
                        invalidation_index=q,
                        haunted=False,
                        next_bearish_index=k,
                        previous_bullish_index=pb,
                    )
                )
                cursor = p
        else:
            cursor += 1

    # Haunted marking loop
    for pivot in pivots:
        if pivot.type == "bearish":
            for x in range(pivot.index + 1, scan_end + 1):
                if candles[x].high > pivot.high:
                    pivot.haunted = True  # type: ignore[misc]
                    break
        else:
            for x in range(pivot.index + 1, scan_end + 1):
                if candles[x].low < pivot.low:
                    pivot.haunted = True  # type: ignore[misc]
                    break

    return pivots


def download_candles(
    *,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    proxies: dict | None = None,
    logger=None,
) -> List[Candle]:
    """Lightweight candle downloader without persistence."""
    client = BinanceClient(BinanceClientConfig(proxies=proxies or None), logger=logger)
    step = interval_to_milliseconds(interval) * MAX_BATCH
    cursor = start_ms
    candles: List[Candle] = []
    while cursor < end_ms:
        batch_end = min(end_ms, cursor + step)
        fetched = client.fetch_klines(
            symbol=symbol,
            interval=interval,
            start_ms=cursor,
            end_ms=batch_end,
            limit=MAX_BATCH,
        )
        if not fetched:
            break
        candles.extend(fetched)
        cursor = fetched[-1].open_time_ms + interval_to_milliseconds(interval)
    return candles


def get_candles(
    *,
    source: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    csv_path: str | None = None,
    proxies: dict | None = None,
    logger=None,
) -> List[Candle]:
    source_normalized = source.strip().lower()
    if source_normalized == "binance":
        return download_candles(
            symbol=symbol,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
            proxies=proxies,
            logger=logger,
        )
    if source_normalized == "csv":
        if not csv_path:
            raise ValueError("csv_path is required when source='csv'")
        return read_candles_from_csv(
            csv_path,
            symbol=symbol,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
        )
    raise ValueError("source must be either 'binance' or 'csv'")


def parse_datetime(value: str) -> datetime:
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    moment = datetime.fromisoformat(cleaned)
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pivot detection on candle data.")
    parser.add_argument("--symbol", required=True, help="Symbol, e.g. ETHUSDT")
    parser.add_argument("--timeframe", required=True, help="Timeframe, e.g. 1h")
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
    parser.add_argument("--scan-length", type=int, default=500, help="Pivot scan length")
    parser.add_argument(
        "--source",
        choices=["binance", "csv"],
        default="binance",
        help="Candle source",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Required when --source csv; path to candle CSV file",
    )
    parser.add_argument("--http-proxy", default=None, help="Optional HTTP proxy URL")
    parser.add_argument("--https-proxy", default=None, help="Optional HTTPS proxy URL")
    args = parser.parse_args()

    proxies = {}
    if args.http_proxy:
        proxies["http"] = args.http_proxy
    if args.https_proxy:
        proxies["https"] = args.https_proxy

    candles = get_candles(
        source=args.source,
        symbol=args.symbol.strip().upper(),
        interval=args.timeframe.strip(),
        start_ms=to_milliseconds(parse_datetime(args.start)),
        end_ms=to_milliseconds(parse_datetime(args.end)),
        csv_path=args.csv_path,
        proxies=proxies or None,
    )
    pivots = detect_pivots(candles, args.scan_length)
    print(f"Candles: {len(candles)}")
    print(f"Pivots: {len(pivots)}")
    for pivot in pivots:
        print(
            f"{pivot.type} pivot index={pivot.index} "
            f"ref={pivot.reference_index} trigger={pivot.trigger_index} "
            f"invalidation={pivot.invalidation_index} haunted={pivot.haunted}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
