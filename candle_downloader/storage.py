from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Set

from .models import Candle, to_datetime, to_milliseconds

try:
    from psycopg_pool import ConnectionPool
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    ConnectionPool = None  # type: ignore[assignment]
    _PSYCOPG_IMPORT_ERROR = exc
else:
    _PSYCOPG_IMPORT_ERROR = None


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
class PostgresConfig:
    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = "postgres"
    database: str = "scalp_test"
    sslmode: str = "prefer"
    min_pool_size: int = 2
    max_pool_size: int = 20
    connect_timeout_seconds: int = 10
    max_idle_seconds: int = 300
    conninfo: str | None = None

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "PostgresConfig":
        file_values = _load_env_file(env_file)

        def get(name: str, default: str | None = None) -> str | None:
            return os.getenv(name, file_values.get(name, default))

        conninfo = get("CANDLE_DATABASE_URL") or get("DATABASE_URL")
        if conninfo:
            return cls(
                conninfo=conninfo,
                min_pool_size=int(get("CANDLE_DB_MIN_POOL_SIZE", get("POSTGRES_MIN_POOL_SIZE", "2")) or "2"),
                max_pool_size=int(get("CANDLE_DB_MAX_POOL_SIZE", get("POSTGRES_MAX_POOL_SIZE", "20")) or "20"),
                connect_timeout_seconds=int(
                    get("CANDLE_DB_CONNECT_TIMEOUT", get("POSTGRES_CONNECT_TIMEOUT", "10")) or "10"
                ),
                max_idle_seconds=int(
                    get("CANDLE_DB_MAX_IDLE_SECONDS", get("POSTGRES_MAX_IDLE_SECONDS", "300")) or "300"
                ),
            )

        return cls(
            host=get("CANDLE_DB_HOST", get("POSTGRES_HOST", "localhost")) or "localhost",
            port=int(get("CANDLE_DB_PORT", get("POSTGRES_PORT", "5432")) or "5432"),
            user=get("CANDLE_DB_USER", get("POSTGRES_USER", "postgres")) or "postgres",
            password=get("CANDLE_DB_PASSWORD", get("POSTGRES_PASSWORD", "postgres")) or "postgres",
            database=get("CANDLE_DB_NAME", get("POSTGRES_DB", "scalp_test")) or "scalp_test",
            sslmode=get("CANDLE_DB_SSLMODE", get("POSTGRES_SSLMODE", "prefer")) or "prefer",
            min_pool_size=int(get("CANDLE_DB_MIN_POOL_SIZE", get("POSTGRES_MIN_POOL_SIZE", "2")) or "2"),
            max_pool_size=int(get("CANDLE_DB_MAX_POOL_SIZE", get("POSTGRES_MAX_POOL_SIZE", "20")) or "20"),
            connect_timeout_seconds=int(
                get("CANDLE_DB_CONNECT_TIMEOUT", get("POSTGRES_CONNECT_TIMEOUT", "10")) or "10"
            ),
            max_idle_seconds=int(
                get("CANDLE_DB_MAX_IDLE_SECONDS", get("POSTGRES_MAX_IDLE_SECONDS", "300")) or "300"
            ),
        )

    def to_conninfo(self) -> str:
        if self.conninfo:
            return self.conninfo
        return (
            f"host={self.host} "
            f"port={self.port} "
            f"dbname={self.database} "
            f"user={self.user} "
            f"password={self.password} "
            f"sslmode={self.sslmode} "
            f"connect_timeout={self.connect_timeout_seconds}"
        )


class PostgresCandleStore(CandleStore):
    """PostgreSQL-backed persistence optimized for concurrent backtest workloads."""

    def __init__(self, config: PostgresConfig) -> None:
        if ConnectionPool is None:
            raise RuntimeError(
                "PostgreSQL dependencies are missing. Install 'psycopg[binary]' and 'psycopg_pool'."
            ) from _PSYCOPG_IMPORT_ERROR

        self._pool = ConnectionPool(
            conninfo=config.to_conninfo(),
            min_size=max(config.min_pool_size, 1),
            max_size=max(config.max_pool_size, max(config.min_pool_size, 1)),
            kwargs={"autocommit": True},
            max_idle=config.max_idle_seconds,
            open=True,
        )
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS candles (
                        symbol TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        open_time BIGINT NOT NULL,
                        close_time BIGINT NOT NULL,
                        open DOUBLE PRECISION NOT NULL,
                        high DOUBLE PRECISION NOT NULL,
                        low DOUBLE PRECISION NOT NULL,
                        close DOUBLE PRECISION NOT NULL,
                        volume DOUBLE PRECISION NOT NULL,
                        PRIMARY KEY (symbol, interval, open_time)
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS candles_symbol_interval_time_idx
                    ON candles (symbol, interval, open_time)
                    """
                )

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

        inserted = 0
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO candles (
                        symbol, interval, open_time, close_time,
                        open, high, low, close, volume
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, interval, open_time) DO NOTHING
                    """,
                    rows,
                )
                inserted = max(cur.rowcount, 0)
        return inserted

    def list_open_times(self, symbol: str, interval: str, start: datetime, end: datetime) -> Set[int]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT open_time
                    FROM candles
                    WHERE symbol = %s AND interval = %s AND open_time >= %s AND open_time < %s
                    """,
                    (symbol, interval, to_milliseconds(start), to_milliseconds(end)),
                )
                rows = cur.fetchall()
        return {int(row[0]) for row in rows}

    def load(self, symbol: str, interval: str, start: datetime, end: datetime) -> List[Candle]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT symbol, interval, open_time, close_time, open, high, low, close, volume
                    FROM candles
                    WHERE symbol = %s AND interval = %s AND open_time >= %s AND open_time < %s
                    ORDER BY open_time ASC
                    """,
                    (symbol, interval, to_milliseconds(start), to_milliseconds(end)),
                )
                rows = cur.fetchall()

        return [
            Candle(
                symbol=row[0],
                interval=row[1],
                open_time=to_datetime(int(row[2])),
                close_time=to_datetime(int(row[3])),
                open=float(row[4]),
                high=float(row[5]),
                low=float(row[6]),
                close=float(row[7]),
                volume=float(row[8]),
            )
            for row in rows
        ]

    def close(self) -> None:
        self._pool.close()


def build_store(kind: str, location: Path | None = None) -> CandleStore:
    """Factory helper for candle storage backend.

    Only PostgreSQL is supported.
    - Environment variables override env-file values.
    - Set env-file using CANDLE_DB_ENV_FILE or POSTGRES_ENV_FILE.
    - Optionally pass `location` pointing to a .env file.
    """
    normalized = (kind or "").strip().lower()
    if normalized != "postgres":
        raise ValueError("Unsupported store kind: only 'postgres' is supported")

    env_file: Path | None = None
    explicit_env_file = os.getenv("CANDLE_DB_ENV_FILE") or os.getenv("POSTGRES_ENV_FILE")
    if explicit_env_file:
        candidate = Path(explicit_env_file)
        if candidate.exists():
            env_file = candidate
    elif location and location.suffix == ".env" and location.exists():
        env_file = location

    return PostgresCandleStore(PostgresConfig.from_env(env_file))


def _load_env_file(path: Path | None) -> Dict[str, str]:
    if path is None or not path.exists() or not path.is_file():
        return {}

    values: Dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values
