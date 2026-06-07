from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Sequence

from candle_downloader.binance import (
    BinanceClient,
    BinanceClientConfig,
    MAX_BATCH,
    interval_to_milliseconds,
)
from candle_downloader.models import Candle, to_milliseconds
from experiments.candle_csv import read_candles_from_csv


@dataclass
class PivotV2Entry:
    candle_index: int
    pivot_index: int | None
    pivot_type: str | None  # "bearish" or "bullish"
    hunt_index: int | None


@dataclass
class PivotConfigV2:
    min_swing_pct: float = 0.0


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


def _swing_pct(high: float, low: float) -> float:
    mid = (high + low) / 2.0
    if mid == 0.0:
        return 0.0
    return (high - low) / mid * 100.0


def detect_pivots_v2(
    candles: Sequence[Candle],
    scan_length: int,
    config: PivotConfigV2 | None = None,
) -> List[PivotV2Entry]:
    """Detect bearish and bullish pivots using the v2 per-candle algorithm.

    For each bullish candle i: find the previous valid bullish candle (with at
    least one bearish candle in between), compute range [start, i], then scan
    forward — if the range low is taken out first, the bearish pivot is the
    argmax high in [start, hunt_index].

    The reverse logic applies for each bearish candle to find bullish pivots.
    """
    if scan_length <= 0:
        raise ValueError("scan_length must be > 0")
    if not candles:
        return []

    cfg = config or PivotConfigV2()
    M = len(candles)
    scan_start = max(0, M - scan_length)
    scan_end = M - 1
    entries: List[PivotV2Entry] = []

    for i in range(scan_start, scan_end + 1):
        c = candles[i]

        if c.is_bullish():
            seen_negative = False
            prev_positive_idx: int | None = None
            j = i - 1
            while j >= scan_start:
                cj = candles[j]
                if not seen_negative:
                    if cj.is_bearish():
                        seen_negative = True
                else:
                    if cj.is_bullish():
                        prev_positive_idx = j
                        break
                j -= 1

            if prev_positive_idx is None:
                entries.append(PivotV2Entry(candle_index=i, pivot_index=None, pivot_type=None, hunt_index=None))
                continue

            start = prev_positive_idx
            end = i
            range_min = min(candles[t].low for t in range(start, end + 1))
            range_max = max(candles[t].high for t in range(start, end + 1))

            if cfg.min_swing_pct > 0.0 and _swing_pct(range_max, range_min) < cfg.min_swing_pct:
                entries.append(PivotV2Entry(candle_index=i, pivot_index=None, pivot_type=None, hunt_index=None))
                continue

            min_hunted = False
            hunt_k: int | None = None
            k = i + 1
            while k <= scan_end:
                if candles[k].low <= range_min:
                    min_hunted = True
                    hunt_k = k
                    break
                if candles[k].high >= range_max:
                    hunt_k = k
                    break
                k += 1

            if min_hunted and hunt_k is not None:
                pivot_idx = _argmax_high(candles, i, hunt_k)
                entries.append(PivotV2Entry(candle_index=i, pivot_index=pivot_idx, pivot_type="bearish", hunt_index=hunt_k))
            else:
                entries.append(PivotV2Entry(candle_index=i, pivot_index=None, pivot_type=None, hunt_index=hunt_k))

        elif c.is_bearish():
            seen_positive = False
            prev_negative_idx: int | None = None
            j = i - 1
            while j >= scan_start:
                cj = candles[j]
                if not seen_positive:
                    if cj.is_bullish():
                        seen_positive = True
                else:
                    if cj.is_bearish():
                        prev_negative_idx = j
                        break
                j -= 1

            if prev_negative_idx is None:
                entries.append(PivotV2Entry(candle_index=i, pivot_index=None, pivot_type=None, hunt_index=None))
                continue

            start = prev_negative_idx
            end = i
            range_max = max(candles[t].high for t in range(start, end + 1))
            range_min = min(candles[t].low for t in range(start, end + 1))

            if cfg.min_swing_pct > 0.0 and _swing_pct(range_max, range_min) < cfg.min_swing_pct:
                entries.append(PivotV2Entry(candle_index=i, pivot_index=None, pivot_type=None, hunt_index=None))
                continue

            max_hunted = False
            hunt_k = None
            k = i + 1
            while k <= scan_end:
                if candles[k].high >= range_max:
                    max_hunted = True
                    hunt_k = k
                    break
                if candles[k].low <= range_min:
                    hunt_k = k
                    break
                k += 1

            if max_hunted and hunt_k is not None:
                pivot_idx = _argmin_low(candles, i, hunt_k)
                entries.append(PivotV2Entry(candle_index=i, pivot_index=pivot_idx, pivot_type="bullish", hunt_index=hunt_k))
            else:
                entries.append(PivotV2Entry(candle_index=i, pivot_index=None, pivot_type=None, hunt_index=hunt_k))

        else:
            entries.append(PivotV2Entry(candle_index=i, pivot_index=None, pivot_type=None, hunt_index=None))

    return entries


def download_candles(
    *,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    proxies: dict | None = None,
    logger=None,
) -> List[Candle]:
    if start_ms >= end_ms:
        raise ValueError("start_ms must be before end_ms")
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
    if start_ms >= end_ms:
        raise ValueError("start_ms must be before end_ms")

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
    parser = argparse.ArgumentParser(description="Run pivot detection v2 on candle data.")
    parser.add_argument("--symbol", required=True, help="Symbol, e.g. ETHUSDT")
    parser.add_argument("--timeframe", required=True, help="Timeframe, e.g. 1h")
    parser.add_argument("--start", required=True, help="Start datetime in ISO format")
    parser.add_argument("--end", required=True, help="End datetime in ISO format")
    parser.add_argument("--scan-length", type=int, default=500, help="Pivot scan length")
    parser.add_argument(
        "--min-swing-pct",
        type=float,
        default=0.0,
        help="Drop pivots whose (high-low)/mid*100 < this value (0 = disabled)",
    )
    parser.add_argument(
        "--source",
        choices=["binance", "csv"],
        default="binance",
        help="Candle source",
    )
    parser.add_argument("--csv-path", default=None, help="Required when --source csv")
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
    cfg = PivotConfigV2(min_swing_pct=args.min_swing_pct)
    entries = detect_pivots_v2(candles, args.scan_length, cfg)
    with_pivot = [e for e in entries if e.pivot_index is not None]
    print(f"Candles: {len(candles)}")
    print(f"Scanned entries: {len(entries)}")
    print(f"Entries with pivot: {len(with_pivot)}")
    for e in with_pivot:
        print(
            f"candle={e.candle_index} pivot={e.pivot_index} type={e.pivot_type} hunt={e.hunt_index}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
