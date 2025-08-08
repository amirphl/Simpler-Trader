from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Sequence, Set, Tuple

from .binance import BinanceClient, interval_to_milliseconds, MAX_BATCH
from .models import normalize_symbol, to_milliseconds
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

    def sync(self, request: DownloadRequest) -> DownloadStats:
        symbol = normalize_symbol(request.symbol)
        interval_ms = interval_to_milliseconds(request.interval)
        start_ms = to_milliseconds(_ensure_utc(request.start))
        end_ms = to_milliseconds(_ensure_utc(request.end))
        stats = DownloadStats()

        existing = set()
        if not request.override:
            existing = self._store.list_open_times(
                symbol=symbol,
                interval=request.interval,
                start=request.start,
                end=request.end,
            )

        pending = _build_missing_times(start_ms, end_ms, interval_ms, existing if not request.override else set())
        stats.requested = len(pending)
        if not pending:
            self._log.info("No missing candles detected", extra={"symbol": symbol, "interval": request.interval})
            return stats

        windows = _coalesce_windows(sorted(pending), interval_ms)
        for window_start, window_end in windows:
            cursor = window_start
            while cursor < window_end:
                chunk_end = min(window_end, cursor + request.max_batch * interval_ms)
                candles = self._client.fetch_klines(
                    symbol=symbol,
                    interval=request.interval,
                    start_ms=cursor,
                    end_ms=chunk_end,
                    limit=min(request.max_batch, (chunk_end - cursor) // interval_ms or 1),
                )
                stats.batches += 1
                stats.fetched_candles += len(candles)
                if not candles:
                    self._log.warning(
                        "Binance returned no candles",
                        extra={"symbol": symbol, "interval": request.interval, "start": cursor, "end": chunk_end},
                    )
                    cursor = chunk_end
                    continue
                filtered = [c for c in candles if c.open_time_ms in pending]
                inserted = self._store.save(filtered)
                stats.saved_candles += inserted
                for candle in filtered:
                    pending.discard(candle.open_time_ms)
                last_open_ms = candles[-1].open_time_ms
                cursor = last_open_ms + interval_ms

        if pending:
            self._log.warning("Some candles remained missing after sync", extra={"missing": len(pending)})
        return stats


def _ensure_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


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

