from __future__ import annotations

import csv
import sqlite3
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Set

from .models import Candle, to_datetime, to_milliseconds


class CandleStore(ABC):
    """Abstract persistence boundary for storing and querying candles."""

    @abstractmethod
    def save(self, candles: Sequence[Candle]) -> int:
        """Persist candles and return the number of newly inserted rows."""

    @abstractmethod
    def list_open_times(self, symbol: str, interval: str, start: datetime, end: datetime) -> Set[int]:
        """Return open times (in ms) for candles already stored in [start, end)."""

    @abstractmethod
    def load(self, symbol: str, interval: str, start: datetime, end: datetime) -> List[Candle]:
        """Load all candles for (symbol, interval) within [start, end)."""

    def close(self) -> None:  # pragma: no cover - optional hook
        """Allow stores with resources to clean up."""


@dataclass(frozen=True)
class SQLiteConfig:
    path: Path


class SQLiteCandleStore(CandleStore):
    """SQLite-backed persistence with a narrow, test-friendly interface."""

    def __init__(self, config: SQLiteConfig) -> None:
        self._path = config.path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                open_time INTEGER NOT NULL,
                close_time INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                PRIMARY KEY (symbol, interval, open_time)
            )
            """
        )
        self._conn.commit()

    def save(self, candles: Sequence[Candle]) -> int:
        if not candles:
            return 0
        rows = [
            (
                candle.symbol,
                candle.interval,
                candle.open_time_ms,
                to_milliseconds(candle.close_time),
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
            )
            for candle in candles
        ]
        before = self._conn.total_changes
        self._conn.executemany(
            """
            INSERT OR IGNORE INTO candles (
                symbol, interval, open_time, close_time,
                open, high, low, close, volume
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._conn.commit()
        return self._conn.total_changes - before

    def list_open_times(self, symbol: str, interval: str, start: datetime, end: datetime) -> Set[int]:
        cursor = self._conn.execute(
            """
            SELECT open_time
            FROM candles
            WHERE symbol = ? AND interval = ? AND open_time >= ? AND open_time < ?
            """,
            (symbol, interval, to_milliseconds(start), to_milliseconds(end)),
        )
        return {row[0] for row in cursor.fetchall()}

    def load(self, symbol: str, interval: str, start: datetime, end: datetime) -> List[Candle]:
        cursor = self._conn.execute(
            """
            SELECT symbol, interval, open_time, close_time, open, high, low, close, volume
            FROM candles
            WHERE symbol = ? AND interval = ? AND open_time >= ? AND open_time < ?
            ORDER BY open_time ASC
            """,
            (symbol, interval, to_milliseconds(start), to_milliseconds(end)),
        )
        rows = cursor.fetchall()
        return [
            Candle(
                symbol=row[0],
                interval=row[1],
                open_time=to_datetime(row[2]),
                close_time=to_datetime(row[3]),
                open=float(row[4]),
                high=float(row[5]),
                low=float(row[6]),
                close=float(row[7]),
                volume=float(row[8]),
            )
            for row in rows
        ]

    def close(self) -> None:
        self._conn.close()


class CSVFileCandleStore(CandleStore):
    """Lightweight CSV storage suited for quick inspection or portability."""

    header = (
        "symbol",
        "interval",
        "open_time",
        "close_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    )

    def __init__(self, file_path: Path) -> None:
        self._path = file_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._index: dict[tuple[str, str], Set[int]] = defaultdict(set)
        self._initialize_file()
        self._hydrate_index()

    def _initialize_file(self) -> None:
        if self._path.exists():
            return
        with self._path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.header)

    def _hydrate_index(self) -> None:
        with self._path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key = (row["symbol"], row["interval"])
                self._index[key].add(int(row["open_time"]))

    def save(self, candles: Sequence[Candle]) -> int:
        if not candles:
            return 0
        inserted = 0
        with self._path.open("a", newline="") as handle:
            writer = csv.writer(handle)
            for candle in candles:
                key = (candle.symbol, candle.interval)
                if candle.open_time_ms in self._index[key]:
                    continue
                writer.writerow(candle.as_row())
                self._index[key].add(candle.open_time_ms)
                inserted += 1
        return inserted

    def list_open_times(self, symbol: str, interval: str, start: datetime, end: datetime) -> Set[int]:
        key = (symbol, interval)
        start_ms, end_ms = to_milliseconds(start), to_milliseconds(end)
        return {ts for ts in self._index.get(key, set()) if start_ms <= ts < end_ms}

    def load(self, symbol: str, interval: str, start: datetime, end: datetime) -> List[Candle]:
        start_ms, end_ms = to_milliseconds(start), to_milliseconds(end)
        candles: List[Candle] = []
        with self._path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row["symbol"] != symbol or row["interval"] != interval:
                    continue
                open_time_ms = int(row["open_time"])
                if open_time_ms < start_ms or open_time_ms >= end_ms:
                    continue
                candles.append(
                    Candle(
                        symbol=row["symbol"],
                        interval=row["interval"],
                        open_time=to_datetime(open_time_ms),
                        close_time=to_datetime(int(row["close_time"])),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                    )
                )
        candles.sort(key=lambda c: c.open_time)
        return candles


def build_store(kind: str, location: Path) -> CandleStore:
    """Factory helper to reduce boilerplate in callers/CLI."""
    if kind == "sqlite":
        return SQLiteCandleStore(SQLiteConfig(path=location))
    if kind == "csv":
        return CSVFileCandleStore(location)
    raise ValueError(f"Unsupported store kind: {kind}")

