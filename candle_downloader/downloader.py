from __future__ import annotations

import logging
from bisect import bisect_right, insort
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Sequence, Set, Tuple

from .binance import BinanceClient, interval_to_milliseconds, MAX_BATCH
from .models import Candle, normalize_symbol, to_milliseconds
from .storage import CandleStore


@dataclass(frozen=True)
class DownloadRequest:
    symbol: str
    interval: str
    start: datetime
    end: datetime
    override: bool = False
    max_batch: int = MAX_BATCH

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError("start must be before end")
        if self.max_batch <= 0 or self.max_batch > MAX_BATCH:
            raise ValueError(f"max_batch must be in 1..{MAX_BATCH}")


@dataclass
class DownloadStats:
    requested: int = 0
    fetched_candles: int = 0
    saved_candles: int = 0
    batches: int = 0


@dataclass
class DownloadResult:
    stats: DownloadStats
    candles: List[Candle]


class CandleDownloader:
    """Coordinates retrieving candles from Binance and persisting them."""

    def __init__(
        self,
        *,
        client: BinanceClient,
        store: CandleStore,
        logger: logging.Logger | None = None,
    ) -> None:
        self._client = client
        self._store = store
        self._log = logger or logging.getLogger(__name__)

    def sync(self, request: DownloadRequest) -> DownloadResult:
        symbol = normalize_symbol(request.symbol)
        interval_ms = interval_to_milliseconds(request.interval)
        start_dt = _align_down_to_interval(_ensure_utc(request.start), interval_ms)
        end_dt = _align_down_to_interval(_ensure_utc(request.end), interval_ms)
        if end_dt <= start_dt:
            raise ValueError("aligned start must be before aligned end")
        stats = DownloadStats()

        candles_by_time: Dict[int, Candle] = {}
        existing_times: Set[int] = set()
        if not request.override:
            existing_candles = self._store.load(
                symbol=symbol,
                interval=request.interval,
                start=start_dt,
                end=end_dt,
            )
            for candle in existing_candles:
                ts = candle.open_time_ms
                candles_by_time[ts] = candle
                existing_times.add(ts)
        else:
            existing_candles = []

        pending = _build_missing_times(
            start_ms=to_milliseconds(start_dt),
            end_ms=to_milliseconds(end_dt),
            step_ms=interval_ms,
            existing=existing_times if not request.override else set(),
        )
        stats.requested = len(pending)
        if not pending and existing_candles:
            self._log.info(
                "Range already populated",
                extra={"symbol": symbol, "interval": request.interval},
            )
            return DownloadResult(stats=stats, candles=existing_candles)

        _fetch_pending_windows(
            pending=pending,
            symbol=symbol,
            interval=request.interval,
            interval_ms=interval_ms,
            request=request,
            client=self._client,
            store=self._store,
            stats=stats,
            sink=candles_by_time,
            logger=self._log,
        )

        final_candles = self._store.load(
            symbol=symbol,
            interval=request.interval,
            start=start_dt,
            end=end_dt,
        )
        final_times = {candle.open_time_ms for candle in final_candles}
        remaining = _build_missing_times(
            to_milliseconds(start_dt), to_milliseconds(end_dt), interval_ms, final_times
        )
        if remaining:
            self._log.warning(
                "Download incomplete; generating synthetic candles for missing slots",
                extra={
                    "symbol": symbol,
                    "interval": request.interval,
                    "missing": len(remaining),
                    "requested_start": start_dt.isoformat(),
                    "requested_end": end_dt.isoformat(),
                },
            )
            synthetic = _synthesize_missing_candles(
                symbol=symbol,
                interval=request.interval,
                interval_ms=interval_ms,
                missing=remaining,
                known=final_candles,
            )
            inserted = self._store.save(synthetic)
            stats.saved_candles += inserted
            final_candles = self._store.load(
                symbol=symbol,
                interval=request.interval,
                start=start_dt,
                end=end_dt,
            )

        return DownloadResult(stats=stats, candles=final_candles)


def _ensure_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _align_down_to_interval(moment: datetime, interval_ms: int) -> datetime:
    """Floor a datetime to the nearest lower interval boundary."""
    ms = to_milliseconds(moment)
    aligned_ms = (ms // interval_ms) * interval_ms
    return datetime.fromtimestamp(aligned_ms / 1000, tz=timezone.utc)


def _build_missing_times(
    start_ms: int,
    end_ms: int,
    step_ms: int,
    existing: Set[int],
) -> Set[int]:
    pending: Set[int] = set()
    cursor = start_ms
    while cursor < end_ms:
        if cursor not in existing:
            pending.add(cursor)
        cursor += step_ms
    return pending


def _coalesce_windows(open_times: Sequence[int], step_ms: int) -> List[Tuple[int, int]]:
    if not open_times:
        return []
    windows: List[Tuple[int, int]] = []
    start = prev = open_times[0]
    for ts in open_times[1:]:
        if ts == prev + step_ms:
            prev = ts
            continue
        windows.append((start, prev + step_ms))
        start = prev = ts
    windows.append((start, prev + step_ms))
    return windows


def _fetch_pending_windows(
    *,
    pending: Set[int],
    symbol: str,
    interval: str,
    interval_ms: int,
    request: DownloadRequest,
    client: BinanceClient,
    store: CandleStore,
    stats: DownloadStats,
    sink: Dict[int, Candle],
    logger: logging.Logger | None = None,
) -> None:
    windows = _coalesce_windows(sorted(pending), interval_ms)
    for window_start, window_end in windows:
        cursor = window_start
        while cursor < window_end:
            remaining_candles = (window_end - cursor) // interval_ms
            limit = min(request.max_batch, max(remaining_candles, 1))
            chunk_end = cursor + limit * interval_ms
            if logger:
                logger.info(
                    "Fetching candles",
                    extra={
                        "symbol": symbol,
                        "interval": interval,
                        "start_time": _ms_to_iso(cursor),
                        "end_time": _ms_to_iso(chunk_end),
                        "limit": limit,
                    },
                )
            candles = client.fetch_klines(
                symbol=symbol,
                interval=interval,
                start_ms=cursor,
                end_ms=chunk_end,
                limit=limit,
            )
            stats.batches += 1
            stats.fetched_candles += len(candles)
            if not candles:
                cursor = chunk_end
                continue
            filtered = [candle for candle in candles if candle.open_time_ms in pending]
            if filtered:
                inserted = store.save(filtered)
                stats.saved_candles += inserted
                for candle in filtered:
                    pending.discard(candle.open_time_ms)
                    sink[candle.open_time_ms] = candle
            cursor = candles[-1].open_time_ms + interval_ms if candles else chunk_end


def _ms_to_iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()


# NOTE: This function is a best-effort attempt to fill in missing candles when the downloader fails to retrieve them after retries.
# NOTE: Needs review to ensure it doesn't introduce misleading data. The synthesized candles are marked with zero volume and open/high/low/close all set to the same price, which is derived from the nearest known candles.
def _synthesize_missing_candles(
    *,
    symbol: str,
    interval: str,
    interval_ms: int,
    missing: Set[int],
    known: Sequence[Candle],
) -> List[Candle]:
    if not missing:
        return []
    known_by_time = {candle.open_time_ms: candle for candle in known}
    known_times = sorted(known_by_time)

    fallback_price = known_by_time[known_times[0]].close if known_times else 1.0
    generated: List[Candle] = []
    for open_time_ms in sorted(missing):
        price = _anchor_price(
            open_time_ms=open_time_ms,
            known_by_time=known_by_time,
            known_times=known_times,
            fallback_price=fallback_price,
        )
        candle = Candle(
            symbol=symbol,
            interval=interval,
            open_time=datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc),
            close_time=datetime.fromtimestamp(
                (open_time_ms + interval_ms - 1) / 1000, tz=timezone.utc
            ),
            open=price,
            high=price,
            low=price,
            close=price,
            volume=0.0,
        )
        generated.append(candle)
        known_by_time[open_time_ms] = candle
        insort(known_times, open_time_ms)
    return generated


def _anchor_price(
    *,
    open_time_ms: int,
    known_by_time: Dict[int, Candle],
    known_times: Sequence[int],
    fallback_price: float,
) -> float:
    idx = bisect_right(known_times, open_time_ms)
    if idx > 0:
        prev_ts = known_times[idx - 1]
        return known_by_time[prev_ts].close
    if idx < len(known_times):
        next_ts = known_times[idx]
        return known_by_time[next_ts].open
    return fallback_price
