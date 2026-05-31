from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, List, Literal, Sequence

from candle_downloader.binance import (
    BinanceClient,
    BinanceClientConfig,
    MAX_BATCH,
    interval_to_milliseconds,
)
from candle_downloader.models import Candle, to_milliseconds
from experiments.candle_csv import read_candles_from_csv

CandlePredicate = Callable[[Candle], bool]


@dataclass
class Pivot:
    index: int               # P
    type: str                # "bullish" or "bearish"
    high: float
    low: float
    reference_index: int     # R (the candle j where search started)
    trigger_index: int       # p (crosses candidate level)
    invalidation_index: int | None  # q (crosses invalidation), may be None
    invalidation_level: float | None = None  # price level that invalidates the setup
    haunted: bool = False
    next_bullish_index: int | None = None
    next_bearish_index: int | None = None
    previous_bullish_index: int | None = None
    previous_bearish_index: int | None = None

    @property
    def hunted(self) -> bool:
        """Preferred spelling; kept alongside `haunted` for API compatibility."""
        return self.haunted

    @hunted.setter
    def hunted(self, value: bool) -> None:
        self.haunted = value


@dataclass
class PivotConfig:
    # On invalidation advance to j+1 instead of q, eliminating gaps between pivots.
    restart_on_invalidation: bool = False
    # Minimum (high-low)/mid*100 threshold; pivots below this are dropped. 0 = disabled.
    min_swing_pct: float = 0.0
    # Use the most recent confirmed same-type pivot as left bound for the invalidation
    # zone instead of the nearest raw same-direction candle.  May widen or narrow the
    # zone depending on how recent the prior structural pivot was.
    use_structural_left_bound: bool = False
    # Include the reference candle j itself in the argmax/argmin pivot search range.
    # When True the structural extreme at j is not missed if j holds the highest high
    # (bearish) or lowest low (bullish).  Default True (corrects the off-by-one).
    include_reference_candle: bool = True


@dataclass
class PivotDetectionConfig:
    """Unified pivot-detection settings used by callers that run pivot detection
    internally (e.g. DetectionConfig, LiquidityZoneConfig)."""

    version: Literal["v1", "v2"] = "v1"
    # scan_length=0 means pivot detection is disabled at the caller level.
    scan_length: int = 500
    min_swing_pct: float = 0.0
    # v1-only options (ignored when version=="v2")
    restart_on_invalidation: bool = False
    use_structural_left_bound: bool = False
    include_reference_candle: bool = True

    def as_v1_config(self) -> PivotConfig:
        return PivotConfig(
            min_swing_pct=self.min_swing_pct,
            restart_on_invalidation=self.restart_on_invalidation,
            use_structural_left_bound=self.use_structural_left_bound,
            include_reference_candle=self.include_reference_candle,
        )


def _prev_match(
    candles: Sequence[Candle],
    start: int,
    predicate: CandlePredicate,
    lower_bound: int = 0,
) -> int | None:
    for i in range(start - 1, lower_bound - 1, -1):
        if predicate(candles[i]):
            return i
    return None


def _next_match(
    candles: Sequence[Candle],
    start: int,
    predicate: CandlePredicate,
    end: int,
) -> int | None:
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


def _swing_pct(pivot: Pivot) -> float:
    mid = (pivot.high + pivot.low) / 2.0
    if mid == 0.0:
        return 0.0
    return (pivot.high - pivot.low) / mid * 100.0


def detect_pivots(
    candles: Sequence[Candle],
    scan_length: int,
    config: PivotConfig | None = None,
) -> List[Pivot]:
    """Detect bullish and bearish structure pivots.

    The scan is intentionally bounded to the most recent ``scan_length`` candles.
    A reference candle R starts a candidate, the first opposite candle provides
    confirmation context, and the trigger p breaks the candidate level. The
    pivot P is the pre-trigger extreme between R and p, so the breakout candle
    itself is not mislabeled as the swing pivot.
    """
    if scan_length <= 0:
        raise ValueError("scan_length must be > 0")
    if not candles:
        return []

    cfg = config or PivotConfig()
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
            pb = _prev_match(candles, j, Candle.is_bullish, scan_start)
            if cfg.use_structural_left_bound:
                prev_same = next((pv for pv in reversed(pivots) if pv.type == "bearish"), None)
                left_bound = j if prev_same is None else prev_same.index
            else:
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
                    cursor = j + 1 if cfg.restart_on_invalidation else q
                else:
                    break
            else:
                # pivot_start = j if cfg.include_reference_candle else j + 1
                pivot_index = _argmax_high(candles, left_bound, k + 1)
                candidate = Pivot(
                    index=pivot_index,
                    type="bearish",
                    high=candles[pivot_index].high,
                    low=candles[pivot_index].low,
                    reference_index=j,
                    trigger_index=p,
                    invalidation_index=q,
                    invalidation_level=invalidation_high,
                    haunted=False,
                    next_bullish_index=k,
                    previous_bearish_index=pb,
                )
                if cfg.min_swing_pct <= 0.0 or _swing_pct(candidate) >= cfg.min_swing_pct:
                    pivots.append(candidate)
                cursor = p
        elif c.is_bullish():
            j = cursor
            k = _next_match(candles, j, Candle.is_bearish, scan_end)
            if k is None:
                break

            candidate_high = max(candles[t].high for t in range(j, k + 1))
            pb = _prev_match(candles, j, Candle.is_bearish, scan_start)
            if cfg.use_structural_left_bound:
                prev_same = next((pv for pv in reversed(pivots) if pv.type == "bullish"), None)
                left_bound = j if prev_same is None else prev_same.index
            else:
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
                    cursor = j + 1 if cfg.restart_on_invalidation else q
                else:
                    break
            else:
                # pivot_start = j if cfg.include_reference_candle else j + 1
                pivot_start = left_bound
                pivot_index = _argmin_low(candles, pivot_start, k + 1)
                candidate = Pivot(
                    index=pivot_index,
                    type="bullish",
                    high=candles[pivot_index].high,
                    low=candles[pivot_index].low,
                    reference_index=j,
                    trigger_index=p,
                    invalidation_index=q,
                    invalidation_level=target_low,
                    haunted=False,
                    next_bearish_index=k,
                    previous_bullish_index=pb,
                )
                if cfg.min_swing_pct <= 0.0 or _swing_pct(candidate) >= cfg.min_swing_pct:
                    pivots.append(candidate)
                cursor = p
        else:
            cursor += 1

    # Hunted marking loop. The response field is still named `haunted` for
    # backwards compatibility with the existing web UI.
    for pivot in pivots:
        if pivot.type == "bearish":
            for x in range(pivot.index + 1, scan_end + 1):
                if candles[x].high > pivot.high:
                    pivot.hunted = True
                    break
        else:
            for x in range(pivot.index + 1, scan_end + 1):
                if candles[x].low < pivot.low:
                    pivot.hunted = True
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
    parser.add_argument(
        "--restart-on-invalidation",
        action="store_true",
        help="On invalidation advance cursor to j+1 instead of q, filling gaps",
    )
    parser.add_argument(
        "--min-swing-pct",
        type=float,
        default=0.0,
        help="Drop pivots whose (high-low)/mid*100 < this value (0 = disabled)",
    )
    parser.add_argument(
        "--use-structural-left-bound",
        action="store_true",
        help="Use previous confirmed same-type pivot as left bound for invalidation zone",
    )
    parser.add_argument(
        "--no-include-reference-candle",
        action="store_true",
        help="Exclude the reference candle j from the pivot search range (legacy behaviour)",
    )
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
    cfg = PivotConfig(
        restart_on_invalidation=args.restart_on_invalidation,
        min_swing_pct=args.min_swing_pct,
        use_structural_left_bound=args.use_structural_left_bound,
        include_reference_candle=not args.no_include_reference_candle,
    )
    pivots = detect_pivots(candles, args.scan_length, cfg)
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
